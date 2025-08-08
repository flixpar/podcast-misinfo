#!/usr/bin/env python3
"""
Podcast Transcription Pipeline
Main orchestration script for fetching, downloading, and transcribing top health podcasts
"""

import os
import sys
import json
import logging
import argparse
import sqlite3
import zstandard as zstd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('podcast_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Project structure
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
AUDIO_DIR = DATA_DIR / "audio"
TRANSCRIPT_DIR = DATA_DIR / "transcripts"
DB_PATH = DATA_DIR / "podcast_metadata.db"
CONFIG_FILE = PROJECT_ROOT / "config.json"

# Create directories
for dir_path in [DATA_DIR, AUDIO_DIR, TRANSCRIPT_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

class PodcastPipeline:
    """Main pipeline orchestrator for podcast transcription"""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize the pipeline with configuration"""
        self.config = self._load_config(config_path or CONFIG_FILE)
        self.db_conn = self._init_database()
        
        # Initialize components (imported from separate modules)
        from rss_parser import RSSParser
        from audio_downloader import AudioDownloader
        from transcript_processor import TranscriptProcessor
        
        # Initialize podcast fetcher based on configuration
        fetcher_type = self.config.get('fetcher', {}).get('type', 'podchaser')
        if fetcher_type.lower() == 'apple':
            from apple_podcast_fetcher import ApplePodcastFetcher
            self.fetcher = ApplePodcastFetcher(self.config.get('fetcher', {}))
            logger.info("Using Apple RSS API for podcast fetching")
        else:
            from podcast_fetcher import PodcastFetcher
            self.fetcher = PodcastFetcher(self.config.get('podchaser', {}))
            logger.info("Using Podchaser API for podcast fetching")
        self.rss_parser = RSSParser()
        self.downloader = AudioDownloader(
            output_dir=AUDIO_DIR,
            max_workers=self.config.get('download', {}).get('max_workers', 4)
        )
        self.transcript_processor = TranscriptProcessor(
            transcript_dir=TRANSCRIPT_DIR,
            db_conn=self.db_conn
        )
        
        # Transcription will be initialized only when needed
        self.transcriber = None
        
    def _load_config(self, config_path: Path) -> dict:
        """Load configuration from JSON file"""
        if not config_path.exists():
            logger.warning(f"Config file not found at {config_path}, using defaults")
            return self._get_default_config()
        
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def _get_default_config(self) -> dict:
        """Return default configuration"""
        return {
            "fetcher": {
                "type": "apple"  # Changed default to Apple RSS API (can be 'apple' or 'podchaser')
            },
            "podchaser": {
                "client_id": os.getenv("PODCHASER_CLIENT_ID", ""),
                "client_secret": os.getenv("PODCHASER_CLIENT_SECRET", ""),
                "api_url": "https://api.podchaser.com/graphql"
            },
            "download": {
                "max_workers": 4,
                "chunk_size": 8192,
                "timeout": 300,
                "max_episodes_per_podcast": 5
            },
            "transcription": {
                "batch_size": 8,
                "num_gpus": 4,
                "model_name": "nvidia/parakeet-tdt-0.6b-v2",
                "max_duration_seconds": 1440,  # 24 minutes
                "enable_timestamps": True,
                "language": "en"
            },
            "processing": {
                "max_parallel_podcasts": 4,
                "checkpoint_interval": 10
            }
        }
    
    def _init_database(self) -> sqlite3.Connection:
        """Initialize SQLite database with required tables"""
        conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        cursor = conn.cursor()
        
        # Create podcasts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS podcasts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                podchaser_id TEXT UNIQUE,
                title TEXT NOT NULL,
                description TEXT,
                publisher TEXT,
                rss_url TEXT,
                apple_podcasts_id TEXT,
                spotify_id TEXT,
                categories TEXT,
                episode_count INTEGER,
                latest_episode_date TEXT,
                fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                processed_at TIMESTAMP,
                status TEXT DEFAULT 'pending',
                metadata TEXT
            )
        """)
        
        # Create episodes table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                podcast_id INTEGER,
                episode_guid TEXT UNIQUE,
                title TEXT NOT NULL,
                description TEXT,
                audio_url TEXT,
                duration_seconds INTEGER,
                published_date TEXT,
                transcript_url TEXT,
                has_rss_transcript BOOLEAN DEFAULT 0,
                audio_file_path TEXT,
                transcript_file_path TEXT,
                transcribed_at TIMESTAMP,
                status TEXT DEFAULT 'pending',
                error_message TEXT,
                metadata TEXT,
                FOREIGN KEY (podcast_id) REFERENCES podcasts(id)
            )
        """)
        
        # Create transcripts table for storing transcript metadata
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS transcripts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                episode_id INTEGER UNIQUE,
                format TEXT,
                compression TEXT DEFAULT 'zstd',
                file_path TEXT,
                word_count INTEGER,
                duration_seconds REAL,
                confidence_score REAL,
                has_timestamps BOOLEAN DEFAULT 1,
                has_speakers BOOLEAN DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT,
                FOREIGN KEY (episode_id) REFERENCES episodes(id)
            )
        """)
        
        # Create processing_logs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS processing_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                phase TEXT NOT NULL,
                podcast_id INTEGER,
                episode_id INTEGER,
                status TEXT,
                message TEXT,
                error_details TEXT,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                duration_seconds REAL,
                metadata TEXT
            )
        """)
        
        # Create indices for better query performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_podcasts_status ON podcasts(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_episodes_status ON episodes(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_episodes_podcast ON episodes(podcast_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_transcripts_episode ON transcripts(episode_id)")
        
        conn.commit()
        return conn
    
    def phase1_fetch_metadata(self, limit: int = 100) -> List[Dict]:
        """
        Phase 1: Fetch podcast metadata from Podchaser API
        
        Args:
            limit: Number of top podcasts to fetch
            
        Returns:
            List of podcast metadata dictionaries
        """
        logger.info(f"Starting Phase 1: Fetching top {limit} health podcasts")
        
        try:
            # Fetch top health podcasts from Podchaser
            podcasts = self.fetcher.get_top_health_podcasts(limit=limit)
            
            # Store in database
            cursor = self.db_conn.cursor()
            for podcast in podcasts:
                cursor.execute("""
                    INSERT OR REPLACE INTO podcasts 
                    (podchaser_id, title, description, publisher, rss_url, 
                     apple_podcasts_id, spotify_id, categories, episode_count, 
                     latest_episode_date, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    podcast.get('id'),
                    podcast.get('title'),
                    podcast.get('description'),
                    podcast.get('publisher'),
                    podcast.get('rss_url'),
                    podcast.get('apple_podcasts_id'),
                    podcast.get('spotify_id'),
                    json.dumps(podcast.get('categories', [])),
                    podcast.get('episode_count'),
                    podcast.get('latest_episode_date'),
                    json.dumps(podcast)
                ))
            
            self.db_conn.commit()
            logger.info(f"Successfully stored {len(podcasts)} podcasts in database")
            
            return podcasts
            
        except Exception as e:
            logger.error(f"Error in Phase 1: {str(e)}")
            raise
    
    def phase2_download_audio(self, max_episodes_per_podcast: Optional[int] = None) -> Dict:
        """
        Phase 2: Parse RSS feeds and download audio files
        
        Args:
            max_episodes_per_podcast: Maximum episodes to download per podcast
            
        Returns:
            Dictionary with download statistics
        """
        logger.info("Starting Phase 2: Parsing RSS and downloading audio")
        
        if max_episodes_per_podcast is None:
            max_episodes_per_podcast = self.config['download']['max_episodes_per_podcast']
        
        cursor = self.db_conn.cursor()
        cursor.execute("""
            SELECT id, podchaser_id, title, rss_url 
            FROM podcasts 
            WHERE rss_url IS NOT NULL
              AND status IN ('pending', 'downloaded')
        """)
        podcasts = cursor.fetchall()
        
        stats = {
            'total_podcasts': len(podcasts),
            'total_episodes': 0,
            'downloaded': 0,
            'has_transcript': 0,
            'errors': 0
        }
        
        with ThreadPoolExecutor(max_workers=self.config['processing']['max_parallel_podcasts']) as executor:
            futures = []
            for podcast_id, podchaser_id, title, rss_url in podcasts:
                future = executor.submit(
                    self._process_podcast_rss,
                    podcast_id, title, rss_url, max_episodes_per_podcast
                )
                futures.append(future)
            
            for future in futures:
                try:
                    result = future.result(timeout=600)
                    stats['total_episodes'] += result['total_episodes']
                    stats['downloaded'] += result['downloaded']
                    stats['has_transcript'] += result['has_transcript']
                except Exception as e:
                    logger.error(f"Error processing podcast: {str(e)}")
                    stats['errors'] += 1
        
        logger.info(f"Phase 2 complete. Stats: {stats}")
        return stats
    
    def _process_podcast_rss(self, podcast_id: int, title: str, rss_url: str, 
                            max_episodes: int) -> Dict:
        """Process a single podcast's RSS feed"""
        logger.info(f"Processing RSS for podcast: {title}")
        
        result = {
            'total_episodes': 0,
            'downloaded': 0,
            'has_transcript': 0
        }
        
        try:
            # Parse RSS feed
            feed_data = self.rss_parser.parse_feed(rss_url)
            episodes = feed_data.get('episodes', [])[:max_episodes]
            result['total_episodes'] = len(episodes)
            
            cursor = self.db_conn.cursor()
            
            for episode in episodes:
                # Check if episode has transcript in RSS
                has_transcript = bool(episode.get('transcript_url'))
                result['has_transcript'] += int(has_transcript)
                
                # Store episode in database
                cursor.execute("""
                    INSERT OR IGNORE INTO episodes
                    (podcast_id, episode_guid, title, description, audio_url,
                     duration_seconds, published_date, transcript_url, 
                     has_rss_transcript, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    podcast_id,
                    episode.get('guid'),
                    episode.get('title'),
                    episode.get('description'),
                    episode.get('audio_url'),
                    episode.get('duration_seconds'),
                    episode.get('published_date'),
                    episode.get('transcript_url'),
                    has_transcript,
                    json.dumps(episode)
                ))
                
                # Download audio if no transcript exists
                if not has_transcript and episode.get('audio_url'):
                    try:
                        file_path = self.downloader.download_episode(
                            episode['audio_url'],
                            podcast_title=title,
                            episode_title=episode.get('title', 'unknown')
                        )
                        
                        cursor.execute("""
                            UPDATE episodes 
                            SET audio_file_path = ?, status = 'downloaded'
                            WHERE episode_guid = ?
                        """, (str(file_path), episode['guid']))
                        
                        result['downloaded'] += 1
                        
                    except Exception as e:
                        logger.error(f"Failed to download episode: {str(e)}")
                        cursor.execute("""
                            UPDATE episodes 
                            SET status = 'error', error_message = ?
                            WHERE episode_guid = ?
                        """, (str(e), episode['guid']))
            
            # Update podcast status
            cursor.execute("""
                UPDATE podcasts 
                SET status = 'downloaded', processed_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (podcast_id,))
            
            self.db_conn.commit()
            
        except Exception as e:
            logger.error(f"Error processing RSS for {title}: {str(e)}")
            cursor = self.db_conn.cursor()
            cursor.execute("""
                UPDATE podcasts 
                SET status = 'error', processed_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (podcast_id,))
            self.db_conn.commit()
        
        return result
    
    def phase3_transcribe(self, batch_size: Optional[int] = None, 
                         use_gpu_ids: Optional[List[int]] = None) -> Dict:
        """
        Phase 3: Transcribe downloaded audio files using NVIDIA Parakeet
        
        Args:
            batch_size: Batch size for transcription
            use_gpu_ids: List of GPU IDs to use (default: [0,1,2,3])
            
        Returns:
            Dictionary with transcription statistics
        """
        logger.info("Starting Phase 3: Transcribing audio files")
        
        # Initialize transcriber if not already done
        if self.transcriber is None:
            from transcriber import ParakeetTranscriber
            self.transcriber = ParakeetTranscriber(
                model_name=self.config['transcription']['model_name'],
                batch_size=batch_size or self.config['transcription']['batch_size'],
                gpu_ids=use_gpu_ids or list(range(self.config['transcription']['num_gpus']))
            )
        
        # Get episodes that need transcription
        cursor = self.db_conn.cursor()
        cursor.execute("""
            SELECT e.id, e.episode_guid, e.title, e.audio_file_path, 
                   p.title as podcast_title
            FROM episodes e
            JOIN podcasts p ON e.podcast_id = p.id
            WHERE e.status = 'downloaded' 
              AND e.audio_file_path IS NOT NULL
              AND e.has_rss_transcript = 0
            ORDER BY e.published_date DESC
        """)
        episodes = cursor.fetchall()
        
        logger.info(f"Found {len(episodes)} episodes to transcribe")
        
        stats = {
            'total': len(episodes),
            'successful': 0,
            'failed': 0,
            'skipped': 0
        }
        
        # Process in batches for efficient GPU utilization
        batch_size = self.config['transcription']['batch_size']
        for i in range(0, len(episodes), batch_size):
            batch = episodes[i:i+batch_size]
            
            try:
                # Prepare batch data
                batch_data = []
                for episode_id, guid, title, audio_path, podcast_title in batch:
                    if not Path(audio_path).exists():
                        logger.warning(f"Audio file not found: {audio_path}")
                        stats['skipped'] += 1
                        continue
                    
                    batch_data.append({
                        'id': episode_id,
                        'guid': guid,
                        'title': title,
                        'audio_path': audio_path,
                        'podcast_title': podcast_title
                    })
                
                if not batch_data:
                    continue
                
                # Transcribe batch
                results = self.transcriber.transcribe_batch(batch_data)
                
                # Process results
                for result in results:
                    if result['success']:
                        # Save transcript
                        transcript_path = self._save_transcript(
                            result['episode_id'],
                            result['transcript'],
                            result['metadata']
                        )
                        
                        # Update database
                        cursor.execute("""
                            UPDATE episodes 
                            SET status = 'transcribed',
                                transcript_file_path = ?,
                                transcribed_at = CURRENT_TIMESTAMP
                            WHERE id = ?
                        """, (transcript_path, result['episode_id']))
                        
                        # Add to transcripts table
                        cursor.execute("""
                            INSERT INTO transcripts
                            (episode_id, format, file_path, word_count, 
                             duration_seconds, confidence_score, has_timestamps)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, (
                            result['episode_id'],
                            'jsonl',
                            transcript_path,
                            result['metadata'].get('word_count'),
                            result['metadata'].get('duration'),
                            result['metadata'].get('confidence'),
                            True
                        ))
                        
                        stats['successful'] += 1
                        
                    else:
                        # Handle failure
                        cursor.execute("""
                            UPDATE episodes 
                            SET status = 'error',
                                error_message = ?
                            WHERE id = ?
                        """, (result.get('error', 'Unknown error'), result['episode_id']))
                        
                        stats['failed'] += 1
                
                self.db_conn.commit()
                
                # Log progress
                logger.info(f"Progress: {i+len(batch)}/{len(episodes)} episodes processed")
                
            except Exception as e:
                logger.error(f"Error processing batch: {str(e)}")
                stats['failed'] += len(batch)
        
        logger.info(f"Phase 3 complete. Stats: {stats}")
        return stats
    
    def _save_transcript(self, episode_id: int, transcript: Dict, 
                        metadata: Dict) -> str:
        """Save transcript to compressed JSONL file"""
        
        # Create filename based on episode ID
        filename = f"episode_{episode_id}.jsonl.zst"
        file_path = TRANSCRIPT_DIR / filename
        
        # Prepare JSONL data
        jsonl_data = []
        
        # Add metadata as first line
        jsonl_data.append({
            'type': 'metadata',
            'episode_id': episode_id,
            'created_at': datetime.now().isoformat(),
            **metadata
        })
        
        # Add transcript segments
        for segment in transcript.get('segments', []):
            jsonl_data.append({
                'type': 'segment',
                'start': segment.get('start'),
                'end': segment.get('end'),
                'text': segment.get('text'),
                'words': segment.get('words', []),
                'confidence': segment.get('confidence')
            })
        
        # Convert to JSONL string
        jsonl_string = '\n'.join(json.dumps(line, ensure_ascii=False) 
                                for line in jsonl_data)
        
        # Compress with zstandard
        cctx = zstd.ZstdCompressor(level=3)
        compressed = cctx.compress(jsonl_string.encode('utf-8'))
        
        # Save to file
        with open(file_path, 'wb') as f:
            f.write(compressed)
        
        logger.debug(f"Saved transcript to {file_path}")
        return str(file_path)
    
    def get_statistics(self) -> Dict:
        """Get current pipeline statistics from database"""
        cursor = self.db_conn.cursor()
        
        stats = {}
        
        # Podcast statistics
        cursor.execute("""
            SELECT status, COUNT(*) 
            FROM podcasts 
            GROUP BY status
        """)
        stats['podcasts'] = dict(cursor.fetchall())
        
        # Episode statistics
        cursor.execute("""
            SELECT status, COUNT(*) 
            FROM episodes 
            GROUP BY status
        """)
        stats['episodes'] = dict(cursor.fetchall())
        
        # Transcript statistics
        cursor.execute("""
            SELECT COUNT(*), SUM(word_count), AVG(confidence_score)
            FROM transcripts
        """)
        result = cursor.fetchone()
        stats['transcripts'] = {
            'total': result[0] or 0,
            'total_words': result[1] or 0,
            'avg_confidence': result[2] or 0
        }
        
        # Storage statistics
        audio_size = sum(f.stat().st_size for f in AUDIO_DIR.glob('**/*') if f.is_file())
        transcript_size = sum(f.stat().st_size for f in TRANSCRIPT_DIR.glob('**/*') if f.is_file())
        
        stats['storage'] = {
            'audio_gb': round(audio_size / (1024**3), 2),
            'transcript_mb': round(transcript_size / (1024**2), 2)
        }
        
        return stats
    
    def run_full_pipeline(self, limit: int = 100, max_episodes: int = 5):
        """Run the complete pipeline from start to finish"""
        logger.info("Starting full pipeline execution")
        
        try:
            # Phase 1: Fetch metadata
            podcasts = self.phase1_fetch_metadata(limit=limit)
            logger.info(f"Phase 1 complete: Fetched {len(podcasts)} podcasts")
            
            # Phase 2: Download audio
            download_stats = self.phase2_download_audio(max_episodes_per_podcast=max_episodes)
            logger.info(f"Phase 2 complete: {download_stats}")
            
            # Phase 3: Transcribe (optional - can be run separately)
            if input("Run transcription phase now? (y/n): ").lower() == 'y':
                transcribe_stats = self.phase3_transcribe()
                logger.info(f"Phase 3 complete: {transcribe_stats}")
            
            # Final statistics
            final_stats = self.get_statistics()
            logger.info(f"Pipeline complete. Final statistics: {json.dumps(final_stats, indent=2)}")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise
    
    def cleanup(self):
        """Clean up resources"""
        if self.db_conn:
            self.db_conn.close()
        if self.transcriber:
            self.transcriber.cleanup()


def main():
    """Main entry point for the script"""
    parser = argparse.ArgumentParser(description='Podcast Transcription Pipeline')
    parser.add_argument('--phase', choices=['1', '2', '3', 'all'], default='all',
                       help='Which phase to run (1=fetch, 2=download, 3=transcribe, all=complete pipeline)')
    parser.add_argument('--limit', type=int, default=100,
                       help='Number of top podcasts to fetch')
    parser.add_argument('--max-episodes', type=int, default=5,
                       help='Maximum episodes per podcast to download')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--stats', action='store_true', help='Show current statistics')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    config_path = Path(args.config) if args.config else None
    pipeline = PodcastPipeline(config_path=config_path)
    
    try:
        if args.stats:
            # Just show statistics
            stats = pipeline.get_statistics()
            print(json.dumps(stats, indent=2))
            
        elif args.phase == '1':
            # Run Phase 1 only
            pipeline.phase1_fetch_metadata(limit=args.limit)
            
        elif args.phase == '2':
            # Run Phase 2 only
            pipeline.phase2_download_audio(max_episodes_per_podcast=args.max_episodes)
            
        elif args.phase == '3':
            # Run Phase 3 only
            pipeline.phase3_transcribe()
            
        elif args.phase == 'all':
            # Run complete pipeline
            pipeline.run_full_pipeline(limit=args.limit, max_episodes=args.max_episodes)
            
    finally:
        pipeline.cleanup()


if __name__ == "__main__":
    main()

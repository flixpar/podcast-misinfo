"""
Transcript Processor Module
Handles storage and retrieval of compressed JSONL transcripts
"""

import json
import logging
import sqlite3
import zstandard as zstd
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class TranscriptProcessor:
    """Handles transcript storage, compression, and retrieval"""
    
    def __init__(self, transcript_dir: Path, db_conn: sqlite3.Connection,
                 compression_level: int = 3):
        """
        Initialize the transcript processor
        
        Args:
            transcript_dir: Directory to store transcript files
            db_conn: SQLite database connection
            compression_level: Zstandard compression level (1-22, default 3)
        """
        self.transcript_dir = Path(transcript_dir)
        self.transcript_dir.mkdir(parents=True, exist_ok=True)
        self.db_conn = db_conn
        self.compression_level = compression_level
        
        # Initialize compressor and decompressor
        self.compressor = zstd.ZstdCompressor(level=compression_level)
        self.decompressor = zstd.ZstdDecompressor()
        
        # Statistics
        self.stats = {
            'transcripts_saved': 0,
            'transcripts_loaded': 0,
            'total_compressed_size': 0,
            'total_uncompressed_size': 0
        }
    
    def save_transcript(self, episode_id: int, transcript: Dict, 
                       metadata: Optional[Dict] = None) -> str:
        """
        Save transcript to compressed JSONL file
        
        Args:
            episode_id: Episode ID
            transcript: Transcript data with text and segments
            metadata: Optional metadata to include
            
        Returns:
            Path to saved transcript file
        """
        logger.debug(f"Saving transcript for episode {episode_id}")
        
        # Create filename
        filename = f"episode_{episode_id}.jsonl.zst"
        file_path = self.transcript_dir / filename
        
        # Prepare JSONL data
        jsonl_lines = []
        
        # Add metadata line
        meta_line = {
            'type': 'metadata',
            'episode_id': episode_id,
            'created_at': datetime.now().isoformat(),
            'version': '1.0',
            'format': 'jsonl_compressed',
            'compression': 'zstd',
            'compression_level': self.compression_level
        }
        
        if metadata:
            meta_line.update(metadata)
        
        jsonl_lines.append(meta_line)
        
        # Add summary line with full text
        summary_line = {
            'type': 'summary',
            'text': transcript.get('text', ''),
            'language': transcript.get('language', 'en'),
            'word_count': len(transcript.get('text', '').split()),
            'segment_count': len(transcript.get('segments', []))
        }
        jsonl_lines.append(summary_line)
        
        # Add segment lines
        for i, segment in enumerate(transcript.get('segments', [])):
            segment_line = {
                'type': 'segment',
                'index': i,
                'start': segment.get('start'),
                'end': segment.get('end'),
                'text': segment.get('text', ''),
                'confidence': segment.get('confidence')
            }
            
            # Add words if available
            if segment.get('words'):
                segment_line['words'] = segment['words']
            
            jsonl_lines.append(segment_line)
        
        # Convert to JSONL string
        jsonl_string = '\n'.join(
            json.dumps(line, ensure_ascii=False, separators=(',', ':'))
            for line in jsonl_lines
        )
        
        # Track sizes
        uncompressed_size = len(jsonl_string.encode('utf-8'))
        
        # Compress
        compressed_data = self.compressor.compress(jsonl_string.encode('utf-8'))
        compressed_size = len(compressed_data)
        
        # Save to file
        with open(file_path, 'wb') as f:
            f.write(compressed_data)
        
        # Update statistics
        self.stats['transcripts_saved'] += 1
        self.stats['total_uncompressed_size'] += uncompressed_size
        self.stats['total_compressed_size'] += compressed_size
        
        # Log compression ratio
        compression_ratio = uncompressed_size / compressed_size if compressed_size > 0 else 0
        logger.info(f"Saved transcript to {file_path} "
                   f"(compression ratio: {compression_ratio:.2f}x, "
                   f"size: {compressed_size / 1024:.1f}KB)")
        
        return str(file_path)
    
    def load_transcript(self, file_path: str) -> Optional[Dict]:
        """
        Load transcript from compressed JSONL file
        
        Args:
            file_path: Path to transcript file
            
        Returns:
            Dictionary with transcript data or None if failed
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.error(f"Transcript file not found: {file_path}")
            return None
        
        try:
            # Read compressed data
            with open(file_path, 'rb') as f:
                compressed_data = f.read()
            
            # Decompress
            jsonl_string = self.decompressor.decompress(compressed_data).decode('utf-8')
            
            # Parse JSONL
            lines = jsonl_string.strip().split('\n')
            
            # Parse each line
            metadata = None
            summary = None
            segments = []
            
            for line in lines:
                if not line:
                    continue
                
                data = json.loads(line)
                line_type = data.get('type')
                
                if line_type == 'metadata':
                    metadata = data
                elif line_type == 'summary':
                    summary = data
                elif line_type == 'segment':
                    segments.append(data)
            
            # Construct transcript object
            transcript = {
                'metadata': metadata,
                'text': summary.get('text', '') if summary else '',
                'language': summary.get('language', 'en') if summary else 'en',
                'segments': segments
            }
            
            # Update statistics
            self.stats['transcripts_loaded'] += 1
            
            logger.debug(f"Loaded transcript from {file_path}")
            return transcript
            
        except Exception as e:
            logger.error(f"Failed to load transcript from {file_path}: {str(e)}")
            return None
    
    def search_transcripts(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Search transcripts for a query string
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of matching transcript segments
        """
        logger.info(f"Searching transcripts for: {query}")
        
        query_lower = query.lower()
        results = []
        
        # Get all transcript files from database
        cursor = self.db_conn.cursor()
        cursor.execute("""
            SELECT e.id, e.title, e.transcript_file_path, p.title as podcast_title
            FROM episodes e
            JOIN podcasts p ON e.podcast_id = p.id
            WHERE e.transcript_file_path IS NOT NULL
            LIMIT 1000
        """)
        
        for episode_id, episode_title, file_path, podcast_title in cursor.fetchall():
            if not file_path or not Path(file_path).exists():
                continue
            
            # Load transcript
            transcript = self.load_transcript(file_path)
            if not transcript:
                continue
            
            # Search in text
            text = transcript.get('text', '').lower()
            if query_lower in text:
                # Find matching segments
                for segment in transcript.get('segments', []):
                    segment_text = segment.get('text', '').lower()
                    if query_lower in segment_text:
                        results.append({
                            'episode_id': episode_id,
                            'episode_title': episode_title,
                            'podcast_title': podcast_title,
                            'segment': segment,
                            'match_context': self._get_context(segment_text, query_lower)
                        })
                        
                        if len(results) >= limit:
                            break
            
            if len(results) >= limit:
                break
        
        logger.info(f"Found {len(results)} matching segments")
        return results
    
    def _get_context(self, text: str, query: str, context_length: int = 100) -> str:
        """Get context around search query match"""
        query_pos = text.lower().find(query.lower())
        if query_pos == -1:
            return text[:200]
        
        start = max(0, query_pos - context_length)
        end = min(len(text), query_pos + len(query) + context_length)
        
        context = text[start:end]
        if start > 0:
            context = '...' + context
        if end < len(text):
            context = context + '...'
        
        return context
    
    def export_transcript(self, episode_id: int, format: str = 'txt') -> Optional[str]:
        """
        Export transcript in various formats
        
        Args:
            episode_id: Episode ID
            format: Export format (txt, srt, vtt, json)
            
        Returns:
            Exported transcript string or None if failed
        """
        # Get transcript file path from database
        cursor = self.db_conn.cursor()
        cursor.execute("""
            SELECT transcript_file_path 
            FROM episodes 
            WHERE id = ?
        """, (episode_id,))
        
        result = cursor.fetchone()
        if not result or not result[0]:
            logger.error(f"No transcript found for episode {episode_id}")
            return None
        
        # Load transcript
        transcript = self.load_transcript(result[0])
        if not transcript:
            return None
        
        # Export based on format
        if format == 'txt':
            return self._export_as_text(transcript)
        elif format == 'srt':
            return self._export_as_srt(transcript)
        elif format == 'vtt':
            return self._export_as_vtt(transcript)
        elif format == 'json':
            return json.dumps(transcript, indent=2, ensure_ascii=False)
        else:
            logger.error(f"Unsupported export format: {format}")
            return None
    
    def _export_as_text(self, transcript: Dict) -> str:
        """Export transcript as plain text"""
        return transcript.get('text', '')
    
    def _export_as_srt(self, transcript: Dict) -> str:
        """Export transcript as SRT format"""
        srt_lines = []
        
        for i, segment in enumerate(transcript.get('segments', []), 1):
            # Index
            srt_lines.append(str(i))
            
            # Timestamps
            start = segment.get('start', 0) or 0
            end = segment.get('end', start + 5) or start + 5
            
            start_time = self._seconds_to_srt_time(start)
            end_time = self._seconds_to_srt_time(end)
            srt_lines.append(f"{start_time} --> {end_time}")
            
            # Text
            srt_lines.append(segment.get('text', ''))
            
            # Empty line
            srt_lines.append('')
        
        return '\n'.join(srt_lines)
    
    def _export_as_vtt(self, transcript: Dict) -> str:
        """Export transcript as WebVTT format"""
        vtt_lines = ['WEBVTT', '']
        
        for segment in transcript.get('segments', []):
            # Timestamps
            start = segment.get('start', 0) or 0
            end = segment.get('end', start + 5) or start + 5
            
            start_time = self._seconds_to_vtt_time(start)
            end_time = self._seconds_to_vtt_time(end)
            vtt_lines.append(f"{start_time} --> {end_time}")
            
            # Text
            vtt_lines.append(segment.get('text', ''))
            
            # Empty line
            vtt_lines.append('')
        
        return '\n'.join(vtt_lines)
    
    def _seconds_to_srt_time(self, seconds: float) -> str:
        """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    
    def _seconds_to_vtt_time(self, seconds: float) -> str:
        """Convert seconds to WebVTT timestamp format (HH:MM:SS.mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
    
    def get_statistics(self) -> Dict:
        """Get transcript processing statistics"""
        stats = self.stats.copy()
        
        # Calculate compression ratio
        if stats['total_uncompressed_size'] > 0:
            stats['compression_ratio'] = (
                stats['total_uncompressed_size'] / 
                stats['total_compressed_size']
            )
        else:
            stats['compression_ratio'] = 0
        
        # Convert sizes to human-readable format
        stats['total_uncompressed_mb'] = stats['total_uncompressed_size'] / (1024 * 1024)
        stats['total_compressed_mb'] = stats['total_compressed_size'] / (1024 * 1024)
        
        # Get transcript count from database
        cursor = self.db_conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM transcripts")
        stats['total_transcripts_in_db'] = cursor.fetchone()[0]
        
        return stats
    
    def validate_transcript(self, file_path: str) -> bool:
        """
        Validate that a transcript file is properly formatted
        
        Args:
            file_path: Path to transcript file
            
        Returns:
            True if valid, False otherwise
        """
        try:
            transcript = self.load_transcript(file_path)
            
            if not transcript:
                return False
            
            # Check required fields
            if not transcript.get('text'):
                logger.warning(f"Transcript missing text: {file_path}")
                return False
            
            if not transcript.get('segments'):
                logger.warning(f"Transcript missing segments: {file_path}")
                return False
            
            # Check metadata
            metadata = transcript.get('metadata', {})
            if not metadata.get('episode_id'):
                logger.warning(f"Transcript missing episode_id: {file_path}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to validate transcript {file_path}: {str(e)}")
            return False
    
    def cleanup_orphaned_files(self):
        """Remove transcript files that are not referenced in the database"""
        logger.info("Cleaning up orphaned transcript files")
        
        # Get all transcript files from database
        cursor = self.db_conn.cursor()
        cursor.execute("""
            SELECT transcript_file_path 
            FROM episodes 
            WHERE transcript_file_path IS NOT NULL
        """)
        
        db_files = set(Path(row[0]).name for row in cursor.fetchall() if row[0])
        
        # Get all files in transcript directory
        disk_files = set(f.name for f in self.transcript_dir.glob('*.jsonl.zst'))
        
        # Find orphaned files
        orphaned = disk_files - db_files
        
        # Remove orphaned files
        for filename in orphaned:
            file_path = self.transcript_dir / filename
            logger.info(f"Removing orphaned file: {file_path}")
            file_path.unlink()
        
        logger.info(f"Removed {len(orphaned)} orphaned files")

"""
Audio Downloader Module
Handles parallel downloading of podcast audio files with resume support
"""

import os
import logging
import hashlib
import time
from pathlib import Path
from typing import Optional, Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse, unquote
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm
from utils import normalize_id

logger = logging.getLogger(__name__)


class AudioDownloader:
    """Handles downloading podcast audio files with parallelization and resume support"""
    
    def __init__(self, output_dir: Path, max_workers: int = 4):
        """
        Initialize the audio downloader
        
        Args:
            output_dir: Directory to save audio files
            max_workers: Maximum number of parallel downloads
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers
        
        # Setup session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=max_workers,
            pool_maxsize=max_workers * 2
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Set headers
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; PodcastTranscriber/1.0)'
        })
        
        # Download statistics
        self.stats = {
            'total_downloads': 0,
            'successful': 0,
            'failed': 0,
            'skipped': 0,
            'total_bytes': 0,
            'total_time': 0
        }
    
    def download_episode(self, audio_url: str, podcast_title: str, 
                        episode_title: str, episode_guid: Optional[str] = None) -> Optional[Path]:
        """
        Download a single episode
        
        Args:
            audio_url: URL of the audio file
            podcast_title: Title of the podcast
            episode_title: Title of the episode
            episode_guid: Optional unique identifier for the episode
            
        Returns:
            Path to downloaded file or None if failed
        """
        # Create normalized IDs for folder and filename base
        podcast_id = normalize_id(podcast_title)
        episode_id = normalize_id(episode_title)
        
        # Create podcast directory
        podcast_dir = self.output_dir / podcast_id
        podcast_dir.mkdir(exist_ok=True)
        
        # Generate filename
        if episode_guid:
            # Use hash of GUID for uniqueness
            guid_hash = hashlib.md5(episode_guid.encode()).hexdigest()[:8]
            filename = f"{episode_id}_{guid_hash}.mp3"
        else:
            filename = f"{episode_id}.mp3"
        
        file_path = podcast_dir / filename
        
        # Check if already downloaded
        if file_path.exists() and file_path.stat().st_size > 0:
            logger.debug(f"File already exists: {file_path}")
            self.stats['skipped'] += 1
            return file_path
        
        try:
            logger.info(f"Downloading: {episode_title} from {podcast_title}")
            
            # Download with resume support
            downloaded = self._download_with_resume(audio_url, file_path)
            
            if downloaded:
                self.stats['successful'] += 1
                logger.info(f"Successfully downloaded: {file_path}")
                return file_path
            else:
                self.stats['failed'] += 1
                return None
                
        except Exception as e:
            logger.error(f"Failed to download {episode_title}: {str(e)}")
            self.stats['failed'] += 1
            
            # Clean up partial file
            if file_path.exists():
                file_path.unlink()
            
            return None
    
    def download_episodes_batch(self, episodes: List[Dict]) -> List[Dict]:
        """
        Download multiple episodes in parallel
        
        Args:
            episodes: List of episode dictionaries with required fields:
                - audio_url: URL of the audio file
                - podcast_title: Title of the podcast
                - episode_title: Title of the episode
                - episode_guid: Optional unique identifier
                
        Returns:
            List of results with download status
        """
        logger.info(f"Starting batch download of {len(episodes)} episodes")
        
        results = []
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all download tasks
            future_to_episode = {}
            for episode in episodes:
                if not episode.get('audio_url'):
                    logger.warning(f"Skipping episode without audio URL: {episode.get('episode_title')}")
                    continue
                
                future = executor.submit(
                    self.download_episode,
                    episode['audio_url'],
                    episode['podcast_title'],
                    episode['episode_title'],
                    episode.get('episode_guid')
                )
                future_to_episode[future] = episode
            
            # Process completed downloads
            with tqdm(total=len(future_to_episode), desc="Downloading episodes") as pbar:
                for future in as_completed(future_to_episode):
                    episode = future_to_episode[future]
                    
                    try:
                        file_path = future.result(timeout=300)
                        results.append({
                            'episode': episode,
                            'file_path': file_path,
                            'success': file_path is not None
                        })
                    except Exception as e:
                        logger.error(f"Download failed for {episode['episode_title']}: {str(e)}")
                        results.append({
                            'episode': episode,
                            'file_path': None,
                            'success': False,
                            'error': str(e)
                        })
                    
                    pbar.update(1)
        
        self.stats['total_time'] += time.time() - start_time
        self.stats['total_downloads'] += len(episodes)
        
        # Log summary
        logger.info(f"Batch download complete: {self.stats['successful']} successful, "
                   f"{self.stats['failed']} failed, {self.stats['skipped']} skipped")
        
        return results
    
    def _download_with_resume(self, url: str, file_path: Path, 
                             chunk_size: int = 8192, timeout: int = 300) -> bool:
        """
        Download file with resume support
        
        Args:
            url: URL to download from
            file_path: Path to save file
            chunk_size: Size of chunks to download
            timeout: Request timeout in seconds
            
        Returns:
            True if successful, False otherwise
        """
        # Check if partial download exists
        resume_header = {}
        mode = 'wb'
        resume_pos = 0
        
        if file_path.exists():
            resume_pos = file_path.stat().st_size
            resume_header = {'Range': f'bytes={resume_pos}-'}
            mode = 'ab'
            logger.debug(f"Resuming download from byte {resume_pos}")
        
        try:
            # Make request
            response = self.session.get(
                url,
                headers=resume_header,
                stream=True,
                timeout=timeout
            )
            
            # Check if server supports resume
            if resume_pos > 0:
                if response.status_code == 206:
                    logger.debug("Server supports resume, continuing download")
                elif response.status_code == 200:
                    logger.warning("Server doesn't support resume, restarting download")
                    mode = 'wb'
                    resume_pos = 0
                else:
                    response.raise_for_status()
            else:
                response.raise_for_status()
            
            # Get total size
            total_size = int(response.headers.get('content-length', 0))
            if resume_pos > 0 and response.status_code == 206:
                # For resumed downloads, add the already downloaded size
                content_range = response.headers.get('content-range', '')
                if content_range:
                    total_size = int(content_range.split('/')[-1])
            
            # Download with progress bar
            with open(file_path, mode) as f:
                with tqdm(total=total_size, initial=resume_pos, 
                         unit='B', unit_scale=True, unit_divisor=1024,
                         desc=file_path.name[:30]) as pbar:
                    
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
                            self.stats['total_bytes'] += len(chunk)
            
            # Verify download
            final_size = file_path.stat().st_size
            if total_size > 0 and final_size < total_size * 0.95:
                logger.warning(f"Downloaded file may be incomplete: {final_size}/{total_size} bytes")
                return False
            
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {str(e)}")
            return False
        except IOError as e:
            logger.error(f"File I/O error: {str(e)}")
            return False
    
    def _sanitize_filename(self, filename: str, max_length: int = 100) -> str:
        """
        Sanitize filename for safe file system usage
        
        Args:
            filename: Original filename
            max_length: Maximum length for filename
            
        Returns:
            Sanitized filename
        """
        # Remove or replace invalid characters
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '_')
        
        # Remove control characters
        filename = ''.join(char for char in filename if ord(char) >= 32)
        
        # Trim whitespace
        filename = filename.strip()
        
        # Limit length
        if len(filename) > max_length:
            filename = filename[:max_length]
        
        # Fallback if empty
        if not filename:
            filename = 'unknown'
        
        return filename
    
    def clean_incomplete_downloads(self):
        """Remove incomplete download files"""
        logger.info("Cleaning up incomplete downloads")
        
        cleaned = 0
        for file_path in self.output_dir.rglob('*.mp3'):
            # Check for very small files (likely incomplete)
            if file_path.stat().st_size < 1000:
                logger.debug(f"Removing incomplete file: {file_path}")
                file_path.unlink()
                cleaned += 1
        
        logger.info(f"Cleaned {cleaned} incomplete files")
    
    def get_download_stats(self) -> Dict:
        """Get download statistics"""
        stats = self.stats.copy()
        
        # Calculate additional metrics
        if stats['total_time'] > 0:
            stats['avg_speed_mbps'] = (stats['total_bytes'] / stats['total_time']) / (1024 * 1024)
        else:
            stats['avg_speed_mbps'] = 0
        
        stats['total_size_gb'] = stats['total_bytes'] / (1024 ** 3)
        
        return stats
    
    def verify_audio_file(self, file_path: Path) -> bool:
        """
        Verify that an audio file is valid
        
        Args:
            file_path: Path to audio file
            
        Returns:
            True if file appears valid, False otherwise
        """
        if not file_path.exists():
            return False
        
        # Check minimum size (1KB)
        if file_path.stat().st_size < 1024:
            return False
        
        # Check file header for common audio formats
        try:
            with open(file_path, 'rb') as f:
                header = f.read(12)
                
                # Check for MP3
                if header[:3] == b'ID3' or header[:2] == b'\xff\xfb':
                    return True
                
                # Check for MP4/M4A
                if header[4:8] == b'ftyp':
                    return True
                
                # Check for OGG
                if header[:4] == b'OggS':
                    return True
                
                # Check for WAV
                if header[:4] == b'RIFF' and header[8:12] == b'WAVE':
                    return True
                
                # Check for FLAC
                if header[:4] == b'fLaC':
                    return True
                
            # If no recognized header, might still be valid
            return True
            
        except Exception as e:
            logger.error(f"Error verifying audio file {file_path}: {str(e)}")
            return False

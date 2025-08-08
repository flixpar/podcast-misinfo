"""
RSS Feed Parser Module
Handles parsing podcast RSS feeds and detecting existing transcripts
"""

import logging
import re
from typing import Dict, List, Optional
from datetime import datetime
from urllib.parse import urlparse
import feedparser
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class RSSParser:
    """Parser for podcast RSS feeds with transcript detection"""
    
    def __init__(self):
        """Initialize the RSS parser"""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; PodcastTranscriber/1.0)'
        })
        
        # Podcast 2.0 namespace for transcript support
        self.podcast_namespace = 'https://podcastindex.org/namespace/1.0'

    def _to_lower_str(self, value) -> str:
        """Safely convert any value to a lowercase string (None -> '')"""
        if value is None:
            return ''
        try:
            return str(value).lower()
        except Exception:
            return ''

    def _is_explicit(self, value) -> bool:
        """Normalize explicit flag that may be bool or various strings"""
        if isinstance(value, bool):
            return value
        lowered = self._to_lower_str(value).strip()
        return lowered in {'yes', 'true', '1', 'explicit'}
        
    def parse_feed(self, rss_url: str) -> Dict:
        """
        Parse an RSS feed and extract podcast and episode information
        
        Args:
            rss_url: URL of the RSS feed
            
        Returns:
            Dictionary containing podcast metadata and episodes list
        """
        logger.info(f"Parsing RSS feed: {rss_url}")
        
        try:
            # Parse the feed
            feed = feedparser.parse(rss_url)
            
            if feed.bozo:
                logger.warning(f"Feed parsing had issues: {feed.bozo_exception}")
            
            # Extract podcast metadata
            podcast_data = self._extract_podcast_metadata(feed)
            
            # Extract episodes
            episodes = self._extract_episodes(feed)
            
            podcast_data['episodes'] = episodes
            podcast_data['episode_count'] = len(episodes)
            
            logger.info(f"Successfully parsed feed with {len(episodes)} episodes")
            return podcast_data
            
        except Exception as e:
            logger.error(f"Failed to parse RSS feed {rss_url}: {str(e)}")
            return {'episodes': [], 'error': str(e)}
    
    def _extract_podcast_metadata(self, feed) -> Dict:
        """Extract podcast-level metadata from feed"""
        channel = feed.feed
        
        metadata = {
            'title': channel.get('title', ''),
            'description': channel.get('description', ''),
            'language': channel.get('language', 'en'),
            'author': channel.get('author', ''),
            'owner': channel.get('itunes_owner', {}).get('name', ''),
            'email': channel.get('itunes_owner', {}).get('email', ''),
            'image': self._get_image_url(channel),
            'categories': self._get_categories(channel),
            'explicit': self._is_explicit(channel.get('itunes_explicit', 'no')),
            'website': channel.get('link', ''),
            'generator': channel.get('generator', ''),
            'last_build_date': channel.get('updated', '')
        }
        
        # Check for Podcast 2.0 features
        if hasattr(channel, 'podcast_locked'):
            metadata['locked'] = channel.podcast_locked == 'yes'
        
        if hasattr(channel, 'podcast_funding'):
            metadata['funding_url'] = channel.podcast_funding.get('url', '')
        
        return metadata
    
    def _extract_episodes(self, feed) -> List[Dict]:
        """Extract episode information from feed entries"""
        episodes = []
        
        for entry in feed.entries:
            try:
                episode = self._parse_episode(entry)
                if episode:
                    episodes.append(episode)
            except Exception as e:
                logger.warning(f"Failed to parse episode: {str(e)}")
                continue
        
        return episodes
    
    def _parse_episode(self, entry) -> Optional[Dict]:
        """Parse a single episode entry"""
        
        # Get audio URL from enclosures
        audio_url = None
        audio_length = 0
        audio_type = None
        
        for enclosure in entry.get('enclosures', []):
            if 'audio' in enclosure.get('type', '').lower():
                audio_url = enclosure.get('href') or enclosure.get('url')
                audio_length = int(enclosure.get('length', 0))
                audio_type = enclosure.get('type')
                break
        
        # Skip if no audio URL
        if not audio_url:
            logger.debug(f"Skipping episode without audio: {entry.get('title', 'Unknown')}")
            return None
        
        # Extract basic metadata
        episode = {
            'guid': entry.get('id') or entry.get('guid') or audio_url,
            'title': entry.get('title', ''),
            'description': self._clean_description(entry.get('description', '')),
            'audio_url': audio_url,
            'audio_length': audio_length,
            'audio_type': audio_type,
            'published_date': self._parse_date(entry.get('published_parsed')),
            'duration_seconds': self._parse_duration(entry),
            'season': entry.get('itunes_season'),
            'episode': entry.get('itunes_episode'),
            'episode_type': entry.get('itunes_episodetype', 'full'),
            'explicit': self._is_explicit(entry.get('itunes_explicit', 'no'))
        }
        
        # Check for transcript information
        transcript_info = self._extract_transcript_info(entry)
        if transcript_info:
            episode['transcript_url'] = transcript_info.get('url')
            episode['transcript_type'] = transcript_info.get('type')
            episode['transcript_language'] = transcript_info.get('language')
        
        # Check for chapter information
        if hasattr(entry, 'podcast_chapters'):
            episode['chapters_url'] = entry.podcast_chapters.get('url')
            episode['chapters_type'] = entry.podcast_chapters.get('type')
        
        # Check for additional metadata
        if hasattr(entry, 'podcast_transcript'):
            # Podcast 2.0 transcript tag
            episode['has_podcast_transcript'] = True
            episode['podcast_transcript_url'] = entry.podcast_transcript.get('url')
            episode['podcast_transcript_type'] = entry.podcast_transcript.get('type')
        
        return episode
    
    def _extract_transcript_info(self, entry) -> Optional[Dict]:
        """
        Extract transcript information from episode entry
        Checks multiple possible transcript sources
        """
        transcript_info = None
        
        # Check for Podcast 2.0 transcript tag
        if hasattr(entry, 'podcast_transcript'):
            transcript_info = {
                'url': entry.podcast_transcript.get('url'),
                'type': entry.podcast_transcript.get('type', 'text/plain'),
                'language': entry.podcast_transcript.get('language', 'en'),
                'rel': entry.podcast_transcript.get('rel', '')
            }
            logger.debug(f"Found Podcast 2.0 transcript for episode: {entry.get('title')}")
        
        # Check for multiple transcript tags (some feeds have multiple formats)
        elif hasattr(entry, 'podcast_transcripts'):
            # Use the first available transcript
            for transcript in entry.podcast_transcripts:
                if transcript.get('url'):
                    transcript_info = {
                        'url': transcript.get('url'),
                        'type': transcript.get('type', 'text/plain'),
                        'language': transcript.get('language', 'en'),
                        'rel': transcript.get('rel', '')
                    }
                    break
        
        # Check for transcript in links
        for link in entry.get('links', []):
            rel_value = ''
            if isinstance(link, dict):
                rel_field = link.get('rel')
                if isinstance(rel_field, list):
                    rel_value = ' '.join(self._to_lower_str(v) for v in rel_field)
                else:
                    rel_value = self._to_lower_str(rel_field)
            if 'transcript' in rel_value:
                transcript_info = {
                    'url': link.get('href'),
                    'type': link.get('type', 'text/plain'),
                    'language': 'en'
                }
                logger.debug(f"Found transcript link for episode: {entry.get('title')}")
                break
        
        # Check for Apple Podcasts transcript indicators
        if not transcript_info and hasattr(entry, 'itunes_transcript'):
            transcript_url = entry.itunes_transcript
            if transcript_url:
                transcript_info = {
                    'url': transcript_url,
                    'type': 'text/vtt',  # Apple usually uses VTT
                    'language': 'en'
                }
        
        return transcript_info
    
    def fetch_transcript(self, transcript_url: str, transcript_type: str = 'text/plain') -> Optional[str]:
        """
        Fetch and parse transcript from URL
        
        Args:
            transcript_url: URL of the transcript
            transcript_type: MIME type of transcript
            
        Returns:
            Transcript text or None if failed
        """
        logger.info(f"Fetching transcript from: {transcript_url}")
        
        try:
            response = self.session.get(transcript_url, timeout=30)
            response.raise_for_status()
            
            content = response.text
            
            # Parse based on type
            transcript_type_lower = self._to_lower_str(transcript_type)
            if 'srt' in transcript_type_lower or transcript_url.endswith('.srt'):
                return self._parse_srt(content)
            elif 'vtt' in transcript_type_lower or transcript_url.endswith('.vtt'):
                return self._parse_vtt(content)
            elif 'html' in transcript_type_lower:
                return self._parse_html_transcript(content)
            elif 'json' in transcript_type_lower:
                return self._parse_json_transcript(content)
            else:
                # Assume plain text
                return content
                
        except Exception as e:
            logger.error(f"Failed to fetch transcript: {str(e)}")
            return None
    
    def _parse_srt(self, content: str) -> str:
        """Parse SRT format transcript"""
        lines = content.strip().split('\n')
        transcript_lines = []
        
        for i, line in enumerate(lines):
            # Skip index numbers and timestamps
            if not line.strip() or line.strip().isdigit():
                continue
            if '-->' in line:
                continue
            
            # Extract speaker and text if present
            if ':' in line:
                parts = line.split(':', 1)
                if len(parts[0].split()) <= 2:  # Likely a speaker name
                    transcript_lines.append(line)
                else:
                    transcript_lines.append(line)
            else:
                transcript_lines.append(line)
        
        return ' '.join(transcript_lines)
    
    def _parse_vtt(self, content: str) -> str:
        """Parse WebVTT format transcript"""
        lines = content.strip().split('\n')
        transcript_lines = []
        
        in_cue = False
        for line in lines:
            if line.startswith('WEBVTT'):
                continue
            if '-->' in line:
                in_cue = True
                continue
            if in_cue and line.strip():
                # Remove VTT tags
                clean_line = re.sub(r'<[^>]+>', '', line)
                transcript_lines.append(clean_line)
            elif not line.strip():
                in_cue = False
        
        return ' '.join(transcript_lines)
    
    def _parse_html_transcript(self, content: str) -> str:
        """Parse HTML format transcript"""
        soup = BeautifulSoup(content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text
        text = soup.get_text()
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text
    
    def _parse_json_transcript(self, content: str) -> str:
        """Parse JSON format transcript (Podcast 2.0 spec)"""
        import json
        
        try:
            data = json.loads(content)
            
            # Handle Podcast 2.0 JSON transcript format
            if 'segments' in data:
                segments = data['segments']
                transcript_parts = []
                
                for segment in segments:
                    speaker = segment.get('speaker', '')
                    text = segment.get('body', '')
                    
                    if speaker:
                        transcript_parts.append(f"{speaker}: {text}")
                    else:
                        transcript_parts.append(text)
                
                return ' '.join(transcript_parts)
            
            # Handle other JSON formats
            elif 'transcript' in data:
                return data['transcript']
            elif 'text' in data:
                return data['text']
            else:
                # Try to extract any text fields
                return str(data)
                
        except json.JSONDecodeError:
            logger.error("Failed to parse JSON transcript")
            return content
    
    def _clean_description(self, description: str) -> str:
        """Clean HTML from description text"""
        if not description:
            return ''
        
        # Remove HTML tags
        soup = BeautifulSoup(description, 'html.parser')
        text = soup.get_text()
        
        # Clean up whitespace
        text = ' '.join(text.split())
        
        # Truncate if too long
        if len(text) > 1000:
            text = text[:997] + '...'
        
        return text
    
    def _parse_date(self, date_tuple) -> Optional[str]:
        """Convert date tuple to ISO format string"""
        if not date_tuple:
            return None
        
        try:
            dt = datetime(*date_tuple[:6])
            return dt.isoformat()
        except:
            return None
    
    def _parse_duration(self, entry) -> Optional[int]:
        """Extract duration in seconds from various formats"""
        
        # Check iTunes duration
        if hasattr(entry, 'itunes_duration'):
            duration = entry.itunes_duration
            if duration:
                return self._duration_to_seconds(duration)
        
        # Check for duration in other fields
        for field in ['duration', 'podcast_duration']:
            if hasattr(entry, field):
                duration = getattr(entry, field)
                if duration:
                    return self._duration_to_seconds(duration)
        
        return None
    
    def _duration_to_seconds(self, duration) -> int:
        """Convert duration string to seconds"""
        if isinstance(duration, (int, float)):
            return int(duration)
        
        if isinstance(duration, str):
            # Handle HH:MM:SS format
            if ':' in duration:
                parts = duration.split(':')
                if len(parts) == 3:
                    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
                elif len(parts) == 2:
                    return int(parts[0]) * 60 + int(parts[1])
            else:
                # Try to parse as integer seconds
                try:
                    return int(duration)
                except:
                    pass
        
        return 0
    
    def _get_image_url(self, channel) -> Optional[str]:
        """Extract image URL from channel data"""
        # Try iTunes image first (usually higher quality)
        if hasattr(channel, 'itunes_image'):
            if isinstance(channel.itunes_image, dict):
                return channel.itunes_image.get('href')
            return channel.itunes_image
        
        # Try standard image
        if hasattr(channel, 'image'):
            if isinstance(channel.image, dict):
                return channel.image.get('url') or channel.image.get('href')
            return channel.image
        
        return None
    
    def _get_categories(self, channel) -> List[str]:
        """Extract categories from channel data"""
        categories = []
        
        # iTunes categories
        if hasattr(channel, 'itunes_category'):
            if isinstance(channel.itunes_category, list):
                for cat in channel.itunes_category:
                    if isinstance(cat, dict):
                        categories.append(cat.get('text', ''))
                    else:
                        categories.append(str(cat))
            elif isinstance(channel.itunes_category, dict):
                categories.append(channel.itunes_category.get('text', ''))
            else:
                categories.append(str(channel.itunes_category))
        
        # Standard categories
        if hasattr(channel, 'tags'):
            for tag in channel.tags:
                if tag.term == 'category':
                    categories.append(tag.label or tag.value)
        
        # Remove duplicates and empty strings
        categories = list(set(cat for cat in categories if cat))
        
        return categories

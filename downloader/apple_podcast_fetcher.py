"""
Apple RSS API Integration Module
Handles fetching top podcasts from Apple's RSS API
"""

import logging
import time
from typing import Dict, List, Optional
from datetime import datetime
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class ApplePodcastFetcher:
    """Apple RSS API client for fetching podcast metadata"""
    
    def __init__(self, config: Dict):
        """
        Initialize the Apple RSS API client
        
        Args:
            config: Configuration dictionary (not needed for Apple RSS API but kept for compatibility)
        """
        self.base_url = 'https://rss.marketingtools.apple.com/api/v2'
        
        # Setup session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        
        logger.info("Apple RSS API client initialized")
    
    def get_top_health_podcasts(self, limit: int = 100, country: str = "US") -> List[Dict]:
        """
        Fetch top health podcasts from Apple RSS API
        
        Args:
            limit: Number of podcasts to fetch (will fetch from top charts and filter)
            country: Country code for charts (default: US)
            
        Returns:
            List of podcast metadata dictionaries
        """
        logger.info(f"Fetching top podcasts from Apple RSS API for {country}")
        
        # Health-related genre IDs from Apple
        health_genres = {
            '1512', '1513', '1517', '1520', '1533',  # Health & Fitness, Alternative Health, Medicine, Mental Health, Science
            '1307', '1471', '1469', '1468', '1461'   # Self-Improvement, Personal Journals, How To, Health, Fitness
        }
        
        # Health-related keywords to look for in titles and descriptions
        health_keywords = [
            'health', 'fitness', 'wellness', 'medical', 'medicine', 'nutrition', 
            'diet', 'workout', 'mental health', 'therapy', 'psychology', 'mindfulness',
            'meditation', 'yoga', 'doctor', 'healthcare', 'healing', 'supplement',
            'sleep', 'anxiety', 'depression', 'stress', 'immune', 'longevity'
        ]
        
        podcasts = []
        
        try:
            # Fetch top 100 podcasts first
            top_url = f"{self.base_url}/{country.lower()}/podcasts/top/100/podcasts.json"
            
            logger.info(f"Fetching from: {top_url}")
            response = self.session.get(top_url, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if 'feed' not in data or 'results' not in data['feed']:
                logger.error("Invalid response format from Apple RSS API")
                raise Exception("Invalid response format from Apple RSS API")
            
            all_podcasts = data['feed']['results']
            logger.info(f"Retrieved {len(all_podcasts)} podcasts from Apple RSS API")
            
            # Filter for health-related podcasts
            health_podcasts_found = 0
            for podcast in all_podcasts:
                if self._is_health_related(podcast, health_genres, health_keywords):
                    health_podcasts_found += 1
                    logger.info(f"Processing health podcast {health_podcasts_found}: {podcast.get('name', 'Unknown')}")
                    
                    # Fetch additional details from iTunes API
                    apple_id = podcast.get('id')
                    itunes_details = None
                    if apple_id:
                        itunes_details = self.get_podcast_details(apple_id)
                        # Add small delay between API calls to avoid rate limiting
                        time.sleep(0.1)
                    
                    formatted = self._format_podcast_data(podcast, itunes_details)
                    podcasts.append(formatted)
                    
                    if len(podcasts) >= limit:
                        break
            
            # # If we don't have enough health podcasts, try to get more by checking different categories
            # if len(podcasts) < limit:
            #     logger.info(f"Only found {len(podcasts)} health podcasts, looking for more general wellness content...")
                
            #     # Add more general wellness-related podcasts with looser criteria
            #     for podcast in all_podcasts:
            #         if len(podcasts) >= limit:
            #             break
                        
            #         if not self._is_health_related(podcast, health_genres, health_keywords):
            #             # Check for looser health-related terms
            #             if self._has_loose_health_connection(podcast):
            #                 # Fetch additional details from iTunes API
            #                 apple_id = podcast.get('id')
            #                 itunes_details = None
            #                 if apple_id:
            #                     itunes_details = self.get_podcast_details(apple_id)
            #                     # Add small delay between API calls to avoid rate limiting
            #                     time.sleep(0.1)
                            
            #                 formatted = self._format_podcast_data(podcast, itunes_details)
            #                 podcasts.append(formatted)
            
            logger.info(f"Successfully filtered {len(podcasts)} health-related podcasts")
            return podcasts[:limit]
            
        except Exception as e:
            logger.error(f"Failed to fetch podcasts from Apple RSS API: {str(e)}")
            raise e
    
    def get_podcast_episodes(self, podcast_id: str, limit: int = 10) -> List[Dict]:
        """
        Fetch episodes for a specific podcast (not available via RSS API)
        This method is kept for compatibility but returns empty list
        
        Args:
            podcast_id: Apple podcast ID
            limit: Maximum number of episodes to fetch
            
        Returns:
            Empty list (episodes not available via RSS API)
        """
        logger.warning("Episode fetching not available via Apple RSS API")
        return []
    
    def get_podcast_details(self, apple_id: str, max_retries: int = 2) -> Optional[Dict]:
        """
        Fetch detailed podcast information from iTunes lookup API with retries
        
        Args:
            apple_id: Apple podcast ID
            max_retries: Maximum number of retry attempts
            
        Returns:
            Dictionary with detailed podcast information or None if failed
        """
        lookup_url = f"https://itunes.apple.com/lookup?id={apple_id}&entity=podcast"
        
        for attempt in range(max_retries + 1):
            try:
                logger.debug(f"Fetching iTunes details for podcast {apple_id} (attempt {attempt + 1})")
                response = self.session.get(lookup_url, timeout=15)
                
                # Handle rate limiting
                if response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', 5))
                    logger.warning(f"Rate limited by iTunes API, waiting {retry_after} seconds")
                    time.sleep(retry_after)
                    continue
                
                response.raise_for_status()
                data = response.json()
                
                if 'results' not in data or len(data['results']) == 0:
                    logger.warning(f"No iTunes data found for podcast {apple_id}")
                    return None
                
                logger.debug(f"Successfully fetched iTunes details for podcast {apple_id}")
                return data['results'][0]
                
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout fetching iTunes details for podcast {apple_id} (attempt {attempt + 1})")
                if attempt < max_retries:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                    
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request error fetching iTunes details for podcast {apple_id}: {str(e)} (attempt {attempt + 1})")
                if attempt < max_retries:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                    
            except Exception as e:
                logger.error(f"Unexpected error fetching iTunes details for podcast {apple_id}: {str(e)}")
                break
        
        logger.error(f"Failed to fetch iTunes details for podcast {apple_id} after {max_retries + 1} attempts")
        return None
    
    def _is_health_related(self, podcast: Dict, health_genres: set, health_keywords: List[str]) -> bool:
        """Check if a podcast is health-related based on genres and content"""
        
        # Check genres first
        genres = podcast.get('genres', [])
        for genre in genres:
            if str(genre.get('genreId', '')) in health_genres:
                return True
        
        # Check title and description for health keywords
        title = podcast.get('name', '').lower()
        artist = podcast.get('artistName', '').lower()
        
        # Combined text to search
        searchable_text = f"{title} {artist}"
        
        # Look for health keywords
        for keyword in health_keywords:
            if keyword in searchable_text:
                return True
        
        return False
    
    def _has_loose_health_connection(self, podcast: Dict) -> bool:
        """Check for loose health connections in podcast metadata"""
        
        # Looser keywords for general wellness
        loose_keywords = [
            'life', 'lifestyle', 'well-being', 'self-care', 'personal development',
            'motivation', 'inspiration', 'energy', 'vitality', 'balance'
        ]
        
        title = podcast.get('name', '').lower()
        artist = podcast.get('artistName', '').lower()
        searchable_text = f"{title} {artist}"
        
        for keyword in loose_keywords:
            if keyword in searchable_text:
                return True
        
        return False
    
    def _format_podcast_data(self, data: Dict, itunes_details: Optional[Dict] = None) -> Dict:
        """Format raw podcast data from Apple RSS API response, enhanced with iTunes lookup data"""
        
        # Extract genre names from RSS data
        categories = []
        if 'genres' in data:
            categories = [genre.get('name', '') for genre in data['genres']]
        
        # Extract Apple Podcast ID from URL or use the id field
        apple_id = data.get('id')
        
        # Default values from RSS API
        title = data.get('name')
        description = f"Popular podcast by {data.get('artistName', 'Unknown Artist')}"
        publisher = data.get('artistName')
        rss_url = None
        web_url = data.get('url')
        latest_episode_date = None
        artwork_url = data.get('artworkUrl100')
        explicit = data.get('contentAdvisoryRating') == 'Explict'  # Note: Apple API has typo "Explict"
        
        # Override with iTunes data if available
        if itunes_details:
            # Use iTunes data for better information
            title = itunes_details.get('collectionName') or title
            description = itunes_details.get('description') or description
            publisher = itunes_details.get('artistName') or publisher
            rss_url = itunes_details.get('feedUrl')  # This is the key field we need!
            web_url = itunes_details.get('collectionViewUrl') or web_url
            
            # Use higher resolution artwork from iTunes if available
            artwork_url = (itunes_details.get('artworkUrl600') or 
                          itunes_details.get('artworkUrl100') or 
                          artwork_url)
            
            # Use iTunes explicit rating if available
            if 'contentAdvisoryRating' in itunes_details:
                explicit = itunes_details.get('contentAdvisoryRating') == 'Explicit'
            
            # Parse release date if available
            if 'releaseDate' in itunes_details:
                try:
                    from datetime import datetime
                    release_date = itunes_details.get('releaseDate')
                    if release_date:
                        parsed_date = datetime.fromisoformat(release_date.replace('Z', '+00:00'))
                        latest_episode_date = parsed_date.isoformat()
                except Exception:
                    pass
            
            # Enhance categories with iTunes genre information
            if 'genres' in itunes_details:
                itunes_genres = itunes_details['genres']
                if isinstance(itunes_genres, list) and itunes_genres:
                    categories = itunes_genres
        
        return {
            'podchaser_id': f"apple_{apple_id}",  # Use Apple ID with prefix for uniqueness
            'title': title,
            'description': description,
            'publisher': publisher,
            'rss_url': rss_url,
            'apple_podcasts_id': apple_id,
            'spotify_id': None,  # Not available in either API
            'web_url': web_url,
            'latest_episode_date': latest_episode_date,
            'categories': categories,
            'artwork_url': artwork_url,
            'explicit': explicit,
            'source': 'apple_rss_itunes' if itunes_details else 'apple_rss'
        }

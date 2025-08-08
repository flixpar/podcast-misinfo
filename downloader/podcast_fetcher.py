"""
Podchaser API Integration Module
Handles authentication and fetching top health podcasts from Podchaser
"""

import logging
import time
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class PodcastFetcher:
    """Podchaser API client for fetching podcast metadata"""
    
    def __init__(self, config: Dict):
        """
        Initialize the Podchaser API client
        
        Args:
            config: Configuration dictionary with client_id and client_secret
        """
        self.api_url = config.get('api_url', 'https://api.podchaser.com/graphql')
        self.client_id = config.get('client_id')
        self.client_secret = config.get('client_secret')
        self.access_token = None
        self.token_expires_at = None
        
        # Setup session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        
        # Authenticate on initialization
        if self.client_id and self.client_secret:
            self.authenticate()
        else:
            logger.warning("No Podchaser credentials provided. API calls will fail.")
    
    def authenticate(self) -> bool:
        """
        Authenticate with Podchaser API and get access token
        
        Returns:
            True if authentication successful, False otherwise
        """
        logger.info("Authenticating with Podchaser API")
        
        mutation = """
            mutation {
                requestAccessToken(
                    input: {
                        grant_type: CLIENT_CREDENTIALS
                        client_id: "%s"
                        client_secret: "%s"
                    }
                ) {
                    access_token
                    expires_in
                }
            }
        """ % (self.client_id, self.client_secret)
        
        try:
            response = self.session.post(
                self.api_url,
                json={'query': mutation},
                headers={'Content-Type': 'application/json'}
            )
            response.raise_for_status()
            
            data = response.json()
            if 'errors' in data:
                logger.error(f"Authentication failed: {data['errors']}")
                return False
            
            token_data = data['data']['requestAccessToken']
            self.access_token = token_data['access_token']
            
            # Calculate token expiration (subtract 1 hour for safety)
            expires_in = token_data.get('expires_in', 31536000)  # Default 1 year
            self.token_expires_at = datetime.now() + timedelta(seconds=expires_in - 3600)
            
            logger.info("Successfully authenticated with Podchaser")
            return True
            
        except Exception as e:
            logger.error(f"Failed to authenticate: {str(e)}")
            return False
    
    def ensure_authenticated(self):
        """Ensure we have a valid access token"""
        if not self.access_token or datetime.now() >= self.token_expires_at:
            if not self.authenticate():
                raise Exception("Failed to authenticate with Podchaser API")
    
    def execute_query(self, query: str, variables: Optional[Dict] = None) -> Dict:
        """
        Execute a GraphQL query against the Podchaser API
        
        Args:
            query: GraphQL query string
            variables: Optional query variables
            
        Returns:
            Response data dictionary
        """
        self.ensure_authenticated()
        
        headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'application/json'
        }
        
        payload = {'query': query}
        if variables:
            payload['variables'] = variables
        
        try:
            response = self.session.post(
                self.api_url,
                json=payload,
                headers=headers,
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            if 'errors' in data:
                logger.error(f"GraphQL errors: {data['errors']}")
                
            return data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            raise
    
    def get_top_health_podcasts(self, limit: int = 100, country: str = "US") -> List[Dict]:
        """
        Fetch top health podcasts from Podchaser charts
        
        Args:
            limit: Number of podcasts to fetch
            country: Country code for charts (default: US)
            
        Returns:
            List of podcast metadata dictionaries
        """
        logger.info(f"Fetching top {limit} health podcasts from Podchaser")
        
        # GraphQL query for health category charts
        query = """
            query GetHealthPodcastCharts(
                $limit: Int!,
                $country: String!,
                $category: String!,
                $platform: ChartPlatform!,
                $day: Date!
            ) {
                charts(
                    platform: $platform,
                    country: $country,
                    category: $category,
                    day: $day,
                    first: $limit
                ) {
                    data {
                        podcast {
                            id
                            title
                            description
                            webUrl
                            rssUrl
                            applePodcastsId
                            spotifyId
                            latestEpisodeDate
                            categories {
                                title
                            }
                        }
                        position
                    }
                    paginatorInfo {
                        hasMorePages
                        currentPage
                    }
                }
            }
        """
        
        # Alternative query if charts doesn't work - search for health podcasts
        search_query = """
            query SearchHealthPodcasts($limit: Int!, $cursor: String) {
                podcasts(
                    searchTerm: "health",
                    filters: {
                        categories: ["Health & Fitness", "Medicine", "Mental Health", "Nutrition"]
                    },
                    sort: {
                        sortBy: FOLLOWER_COUNT,
                        direction: DESCENDING
                    },
                    first: $limit,
                    cursor: $cursor
                ) {
                    data {
                        id
                        title
                        description
                        webUrl
                        rssUrl
                        applePodcastsId
                        spotifyId
                        latestEpisodeDate
                        categories {
                            title
                        }
                    }
                    cursorInfo {
                        total
                        nextCursor
                    }
                }
            }
        """
        
        podcasts = []
        
        try:
            # Try charts endpoint first
            variables = {
                'limit': min(limit, 20),
                'country': country,
                'category': 'Health & Fitness',
                'platform': 'APPLE_PODCASTS',
                'day': (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
            }
            
            has_next = True
            cursor = None
            
            while has_next and len(podcasts) < limit:
                if cursor:
                    variables['after'] = cursor
                
                # Try charts first
                try:
                    response = self.execute_query(query, variables)
                    chart_data = response.get('data', {}).get('charts', {})
                    
                    if chart_data and 'data' in chart_data:
                        for item in chart_data['data']:
                            podcast_data = item['podcast']
                            podcasts.append(self._format_podcast_data(podcast_data))
                        
                        page_info = chart_data['paginatorInfo']
                        has_next = page_info['hasMorePages'] and len(podcasts) < limit
                        cursor = page_info.get('currentPage') + 1 if page_info['hasMorePages'] else None
                    else:
                        # Fallback to search if charts don't work
                        logger.info("Charts API not returning data, falling back to search")
                        break
                        
                except Exception as e:
                    logger.warning(f"Charts query failed: {str(e)}, trying search instead")
                    break
                
                # Rate limiting
                time.sleep(0.5)
            
            # If we didn't get enough from charts, use search
            if len(podcasts) < limit:
                logger.info(f"Using search to find additional health podcasts")
                
                variables = {'limit': min(limit - len(podcasts), 50)}
                cursor = None
                has_next = True
                
                while has_next and len(podcasts) < limit:
                    if cursor:
                        variables['cursor'] = cursor
                    
                    response = self.execute_query(search_query, variables)
                    search_data = response.get('data', {}).get('podcasts', {})
                    
                    if search_data and 'data' in search_data:
                        for podcast_data in search_data['data']:
                            formatted = self._format_podcast_data(podcast_data)
                            
                            # Avoid duplicates
                            if not any(p['podchaser_id'] == formatted['podchaser_id'] for p in podcasts):
                                podcasts.append(formatted)
                        
                        cursor_info = search_data['cursorInfo']
                        has_next = cursor_info.get('nextCursor') is not None and len(podcasts) < limit
                        cursor = cursor_info.get('nextCursor')
                    else:
                        break
                    
                    # Rate limiting
                    time.sleep(0.5)
            
            logger.info(f"Successfully fetched {len(podcasts)} health podcasts")
            return podcasts[:limit]
            
        except Exception as e:
            logger.error(f"Failed to fetch podcasts: {str(e)}")
            raise e
    
    def get_podcast_episodes(self, podcast_id: str, limit: int = 10) -> List[Dict]:
        """
        Fetch episodes for a specific podcast
        
        Args:
            podcast_id: Podchaser podcast ID
            limit: Maximum number of episodes to fetch
            
        Returns:
            List of episode metadata dictionaries
        """
        query = """
            query GetPodcastEpisodes($podcastId: ID!, $limit: Int!) {
                podcast(identifier: {id: $podcastId, type: PODCHASER}) {
                    episodes(first: $limit, sort: {sortBy: AIR_DATE, direction: DESC}) {
                        edges {
                            node {
                                id
                                guid
                                title
                                description
                                audioUrl
                                duration
                                airDate
                                season
                                episode
                            }
                        }
                    }
                }
            }
        """
        
        variables = {
            'podcastId': podcast_id,
            'limit': limit
        }
        
        try:
            response = self.execute_query(query, variables)
            episodes_data = response.get('data', {}).get('podcast', {}).get('episodes', {})
            
            episodes = []
            if episodes_data and 'edges' in episodes_data:
                for edge in episodes_data['edges']:
                    episode = edge['node']
                    episodes.append({
                        'id': episode.get('id'),
                        'guid': episode.get('guid'),
                        'title': episode.get('title'),
                        'description': episode.get('description'),
                        'audio_url': episode.get('audioUrl'),
                        'duration_seconds': episode.get('duration'),
                        'published_date': episode.get('airDate'),
                        'season': episode.get('season'),
                        'episode_number': episode.get('episode')
                    })
            
            return episodes
            
        except Exception as e:
            logger.error(f"Failed to fetch episodes for podcast {podcast_id}: {str(e)}")
            return []
    
    def _format_podcast_data(self, data: Dict) -> Dict:
        """Format raw podcast data from API response"""
        return {
            'podchaser_id': data.get('id'),
            'title': data.get('title'),
            'description': data.get('description'),
            'web_url': data.get('webUrl'),
            'rss_url': data.get('rssUrl'),
            'apple_podcasts_id': data.get('applePodcastsId'),
            'spotify_id': data.get('spotifyId'),
            'latest_episode_date': data.get('latestEpisodeDate'),
            'categories': [cat.get('title') for cat in data.get('categories', [])]
        }

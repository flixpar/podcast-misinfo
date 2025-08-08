"""
API Server Module
RESTful API for accessing and searching transcripts
"""

import os
import json
import sqlite3
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from functools import wraps
import hashlib
import secrets

from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_caching import Cache
import jwt
from werkzeug.security import generate_password_hash, check_password_hash

from transcript_processor import TranscriptProcessor

logger = logging.getLogger(__name__)


class TranscriptAPI:
    """RESTful API for transcript access"""
    
    def __init__(self, db_path: str, transcript_dir: str, config: Optional[Dict] = None):
        """
        Initialize the API server
        
        Args:
            db_path: Path to SQLite database
            transcript_dir: Path to transcript directory
            config: Optional configuration dictionary
        """
        self.db_path = db_path
        self.transcript_dir = Path(transcript_dir)
        self.config = config or {}
        
        # Initialize Flask app
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = self.config.get('secret_key', secrets.token_hex(32))
        
        # Enable CORS
        CORS(self.app, origins=self.config.get('cors_origins', '*'))
        
        # Initialize rate limiter
        self.limiter = Limiter(
            app=self.app,
            key_func=get_remote_address,
            default_limits=["1000 per hour", "100 per minute"]
        )
        
        # Initialize cache
        self.app.config['CACHE_TYPE'] = 'simple'
        self.cache = Cache(self.app)
        
        # Initialize transcript processor
        self.transcript_processor = TranscriptProcessor(
            transcript_dir=self.transcript_dir,
            db_conn=sqlite3.connect(self.db_path, check_same_thread=False)
        )
        
        # API keys storage (in production, use a proper database)
        self.api_keys = {}
        self._load_api_keys()
        
        # Setup routes
        self._setup_routes()
        
        # Request/response logging
        self._setup_logging()
    
    def _load_api_keys(self):
        """Load API keys from configuration or database"""
        # Default API key for testing
        self.api_keys['test_key_123'] = {
            'name': 'Test Application',
            'created_at': datetime.now().isoformat(),
            'rate_limit': '1000/hour',
            'permissions': ['read', 'search']
        }
        
        # Load from config if available
        if 'api_keys' in self.config:
            self.api_keys.update(self.config['api_keys'])
    
    def _setup_logging(self):
        """Setup request/response logging"""
        
        @self.app.before_request
        def log_request():
            """Log incoming requests"""
            logger.debug(f"Request: {request.method} {request.path} from {request.remote_addr}")
        
        @self.app.after_request
        def log_response(response):
            """Log outgoing responses"""
            logger.debug(f"Response: {response.status_code} for {request.path}")
            return response
    
    def require_api_key(f):
        """Decorator to require API key authentication"""
        @wraps(f)
        def decorated_function(*args, **kwargs):
            api_key = request.headers.get('X-API-Key')
            
            if not api_key:
                return jsonify({'error': 'API key required'}), 401
            
            if api_key not in kwargs['self'].api_keys:
                return jsonify({'error': 'Invalid API key'}), 401
            
            # Add API key info to request context
            request.api_key_info = kwargs['self'].api_keys[api_key]
            
            return f(*args, **kwargs)
        return decorated_function
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.route('/api/v1/health', methods=['GET'])
        @self.limiter.limit("10 per minute")
        def health_check():
            """Health check endpoint"""
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'version': '1.0.0'
            })
        
        @self.app.route('/api/v1/stats', methods=['GET'])
        @self.cache.cached(timeout=60)
        def get_statistics():
            """Get system statistics"""
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Get statistics
                cursor.execute("""
                    SELECT 
                        (SELECT COUNT(*) FROM podcasts) as total_podcasts,
                        (SELECT COUNT(*) FROM episodes) as total_episodes,
                        (SELECT COUNT(*) FROM transcripts) as total_transcripts,
                        (SELECT SUM(word_count) FROM transcripts) as total_words,
                        (SELECT AVG(confidence_score) FROM transcripts) as avg_confidence
                """)
                
                result = cursor.fetchone()
                stats = {
                    'total_podcasts': result[0] or 0,
                    'total_episodes': result[1] or 0,
                    'total_transcripts': result[2] or 0,
                    'total_words': result[3] or 0,
                    'avg_confidence': result[4] or 0
                }
                
                conn.close()
                return jsonify(stats)
                
            except Exception as e:
                logger.error(f"Failed to get statistics: {str(e)}")
                return jsonify({'error': 'Failed to retrieve statistics'}), 500
        
        @self.app.route('/api/v1/podcasts', methods=['GET'])
        @self.limiter.limit("100 per minute")
        @self.cache.cached(timeout=300, query_string=True)
        def list_podcasts():
            """List all podcasts with pagination"""
            page = request.args.get('page', 1, type=int)
            per_page = request.args.get('per_page', 20, type=int)
            search = request.args.get('search', '')
            
            # Limit per_page to prevent abuse
            per_page = min(per_page, 100)
            offset = (page - 1) * per_page
            
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Build query
                query = """
                    SELECT 
                        id, podchaser_id, title, description, publisher,
                        episode_count, status
                    FROM podcasts
                """
                params = []
                
                if search:
                    query += " WHERE title LIKE ? OR description LIKE ?"
                    params.extend([f'%{search}%', f'%{search}%'])
                
                query += " ORDER BY title LIMIT ? OFFSET ?"
                params.extend([per_page, offset])
                
                cursor.execute(query, params)
                
                podcasts = []
                for row in cursor.fetchall():
                    podcasts.append({
                        'id': row[0],
                        'podchaser_id': row[1],
                        'title': row[2],
                        'description': row[3],
                        'publisher': row[4],
                        'episode_count': row[5],
                        'status': row[6]
                    })
                
                # Get total count
                count_query = "SELECT COUNT(*) FROM podcasts"
                if search:
                    count_query += " WHERE title LIKE ? OR description LIKE ?"
                    cursor.execute(count_query, [f'%{search}%', f'%{search}%'])
                else:
                    cursor.execute(count_query)
                
                total = cursor.fetchone()[0]
                
                conn.close()
                
                return jsonify({
                    'podcasts': podcasts,
                    'pagination': {
                        'page': page,
                        'per_page': per_page,
                        'total': total,
                        'total_pages': (total + per_page - 1) // per_page
                    }
                })
                
            except Exception as e:
                logger.error(f"Failed to list podcasts: {str(e)}")
                return jsonify({'error': 'Failed to retrieve podcasts'}), 500
        
        @self.app.route('/api/v1/podcasts/<int:podcast_id>/episodes', methods=['GET'])
        @self.limiter.limit("100 per minute")
        def list_episodes(podcast_id):
            """List episodes for a podcast"""
            page = request.args.get('page', 1, type=int)
            per_page = request.args.get('per_page', 20, type=int)
            status = request.args.get('status', '')
            
            per_page = min(per_page, 100)
            offset = (page - 1) * per_page
            
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Build query
                query = """
                    SELECT 
                        id, episode_guid, title, description,
                        duration_seconds, published_date, status,
                        has_rss_transcript, transcript_file_path
                    FROM episodes
                    WHERE podcast_id = ?
                """
                params = [podcast_id]
                
                if status:
                    query += " AND status = ?"
                    params.append(status)
                
                query += " ORDER BY published_date DESC LIMIT ? OFFSET ?"
                params.extend([per_page, offset])
                
                cursor.execute(query, params)
                
                episodes = []
                for row in cursor.fetchall():
                    episodes.append({
                        'id': row[0],
                        'guid': row[1],
                        'title': row[2],
                        'description': row[3],
                        'duration_seconds': row[4],
                        'published_date': row[5],
                        'status': row[6],
                        'has_rss_transcript': bool(row[7]),
                        'has_generated_transcript': bool(row[8])
                    })
                
                conn.close()
                
                return jsonify({
                    'episodes': episodes,
                    'pagination': {
                        'page': page,
                        'per_page': per_page
                    }
                })
                
            except Exception as e:
                logger.error(f"Failed to list episodes: {str(e)}")
                return jsonify({'error': 'Failed to retrieve episodes'}), 500
        
        @self.app.route('/api/v1/transcripts/<int:episode_id>', methods=['GET'])
        @self.limiter.limit("100 per minute")
        def get_transcript(episode_id):
            """Get transcript for an episode"""
            format = request.args.get('format', 'json')
            include_timestamps = request.args.get('timestamps', 'true').lower() == 'true'
            
            try:
                # Get transcript file path from database
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT transcript_file_path, title
                    FROM episodes
                    WHERE id = ?
                """, (episode_id,))
                
                result = cursor.fetchone()
                conn.close()
                
                if not result or not result[0]:
                    return jsonify({'error': 'Transcript not found'}), 404
                
                file_path, title = result
                
                # Load transcript
                transcript = self.transcript_processor.load_transcript(file_path)
                
                if not transcript:
                    return jsonify({'error': 'Failed to load transcript'}), 500
                
                # Format based on request
                if format == 'txt':
                    text = transcript.get('text', '')
                    return Response(text, mimetype='text/plain')
                
                elif format == 'srt':
                    srt_content = self.transcript_processor._export_as_srt(transcript)
                    return Response(srt_content, mimetype='text/srt')
                
                elif format == 'vtt':
                    vtt_content = self.transcript_processor._export_as_vtt(transcript)
                    return Response(vtt_content, mimetype='text/vtt')
                
                else:  # JSON format
                    response_data = {
                        'episode_id': episode_id,
                        'title': title,
                        'text': transcript.get('text', ''),
                        'language': transcript.get('language', 'en'),
                        'metadata': transcript.get('metadata', {})
                    }
                    
                    if include_timestamps:
                        response_data['segments'] = transcript.get('segments', [])
                    
                    return jsonify(response_data)
                
            except Exception as e:
                logger.error(f"Failed to get transcript: {str(e)}")
                return jsonify({'error': 'Failed to retrieve transcript'}), 500
        
        @self.app.route('/api/v1/search', methods=['GET'])
        @self.limiter.limit("50 per minute")
        def search_transcripts():
            """Search across all transcripts"""
            query = request.args.get('q', '')
            limit = min(request.args.get('limit', 20, type=int), 100)
            
            if not query:
                return jsonify({'error': 'Search query required'}), 400
            
            if len(query) < 3:
                return jsonify({'error': 'Query must be at least 3 characters'}), 400
            
            try:
                results = self.transcript_processor.search_transcripts(query, limit)
                
                return jsonify({
                    'query': query,
                    'results': results,
                    'count': len(results)
                })
                
            except Exception as e:
                logger.error(f"Search failed: {str(e)}")
                return jsonify({'error': 'Search failed'}), 500
        
        @self.app.route('/api/v1/export', methods=['POST'])
        @self.limiter.limit("10 per minute")
        @require_api_key
        def export_transcripts(self):
            """Export multiple transcripts"""
            data = request.get_json()
            
            if not data or 'episode_ids' not in data:
                return jsonify({'error': 'episode_ids required'}), 400
            
            episode_ids = data['episode_ids']
            format = data.get('format', 'json')
            
            if not isinstance(episode_ids, list) or len(episode_ids) > 100:
                return jsonify({'error': 'Invalid episode_ids (max 100)'}), 400
            
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                exported_data = []
                
                for episode_id in episode_ids:
                    cursor.execute("""
                        SELECT e.title, p.title as podcast_title, e.transcript_file_path
                        FROM episodes e
                        JOIN podcasts p ON e.podcast_id = p.id
                        WHERE e.id = ?
                    """, (episode_id,))
                    
                    result = cursor.fetchone()
                    if result and result[2]:
                        transcript = self.transcript_processor.load_transcript(result[2])
                        if transcript:
                            exported_data.append({
                                'episode_id': episode_id,
                                'episode_title': result[0],
                                'podcast_title': result[1],
                                'transcript': transcript.get('text', ''),
                                'word_count': len(transcript.get('text', '').split())
                            })
                
                conn.close()
                
                if format == 'csv':
                    # Convert to CSV format
                    import csv
                    import io
                    
                    output = io.StringIO()
                    writer = csv.DictWriter(output, fieldnames=[
                        'episode_id', 'podcast_title', 'episode_title', 
                        'word_count', 'transcript'
                    ])
                    writer.writeheader()
                    writer.writerows(exported_data)
                    
                    return Response(
                        output.getvalue(),
                        mimetype='text/csv',
                        headers={'Content-Disposition': 'attachment; filename=transcripts.csv'}
                    )
                
                else:
                    return jsonify({
                        'exported': len(exported_data),
                        'data': exported_data
                    })
                
            except Exception as e:
                logger.error(f"Export failed: {str(e)}")
                return jsonify({'error': 'Export failed'}), 500
        
        @self.app.route('/api/v1/webhooks', methods=['POST'])
        @self.limiter.limit("5 per minute")
        @require_api_key
        def register_webhook(self):
            """Register a webhook for transcript completion notifications"""
            data = request.get_json()
            
            if not data or 'url' not in data:
                return jsonify({'error': 'Webhook URL required'}), 400
            
            webhook_url = data['url']
            events = data.get('events', ['transcript.completed'])
            
            # In production, save webhook to database
            webhook_id = hashlib.md5(webhook_url.encode()).hexdigest()[:8]
            
            # Store webhook (placeholder - implement proper storage)
            logger.info(f"Registered webhook {webhook_id} for {webhook_url}")
            
            return jsonify({
                'webhook_id': webhook_id,
                'url': webhook_url,
                'events': events,
                'status': 'active'
            })
        
        @self.app.route('/api/v1/analytics', methods=['GET'])
        @self.cache.cached(timeout=300)
        def get_analytics():
            """Get analytics data"""
            period = request.args.get('period', '7d')
            
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Parse period
                if period == '24h':
                    time_filter = "datetime('now', '-1 day')"
                elif period == '7d':
                    time_filter = "datetime('now', '-7 days')"
                elif period == '30d':
                    time_filter = "datetime('now', '-30 days')"
                else:
                    time_filter = "datetime('now', '-7 days')"
                
                # Get transcription activity
                cursor.execute(f"""
                    SELECT 
                        DATE(transcribed_at) as date,
                        COUNT(*) as count,
                        SUM(duration_seconds) as total_duration
                    FROM episodes
                    WHERE transcribed_at > {time_filter}
                    GROUP BY DATE(transcribed_at)
                    ORDER BY date
                """)
                
                activity = []
                for row in cursor.fetchall():
                    activity.append({
                        'date': row[0],
                        'episodes_transcribed': row[1],
                        'total_duration_hours': (row[2] or 0) / 3600
                    })
                
                # Get top podcasts by episode count
                cursor.execute("""
                    SELECT 
                        p.title,
                        COUNT(e.id) as episode_count,
                        SUM(CASE WHEN e.status = 'transcribed' THEN 1 ELSE 0 END) as transcribed_count
                    FROM podcasts p
                    JOIN episodes e ON p.id = e.podcast_id
                    GROUP BY p.id
                    ORDER BY episode_count DESC
                    LIMIT 10
                """)
                
                top_podcasts = []
                for row in cursor.fetchall():
                    top_podcasts.append({
                        'title': row[0],
                        'total_episodes': row[1],
                        'transcribed_episodes': row[2]
                    })
                
                conn.close()
                
                return jsonify({
                    'period': period,
                    'activity': activity,
                    'top_podcasts': top_podcasts
                })
                
            except Exception as e:
                logger.error(f"Failed to get analytics: {str(e)}")
                return jsonify({'error': 'Failed to retrieve analytics'}), 500
        
        @self.app.errorhandler(404)
        def not_found(error):
            """Handle 404 errors"""
            return jsonify({'error': 'Endpoint not found'}), 404
        
        @self.app.errorhandler(429)
        def rate_limit_exceeded(error):
            """Handle rate limit errors"""
            return jsonify({'error': 'Rate limit exceeded', 'message': str(error.description)}), 429
        
        @self.app.errorhandler(500)
        def internal_error(error):
            """Handle internal errors"""
            logger.error(f"Internal error: {str(error)}")
            return jsonify({'error': 'Internal server error'}), 500
    
    def run(self, host: str = '0.0.0.0', port: int = 8080, debug: bool = False):
        """
        Run the API server
        
        Args:
            host: Host to bind to
            port: Port to bind to
            debug: Enable debug mode
        """
        logger.info(f"Starting API server on http://{host}:{port}")
        self.app.run(host=host, port=port, debug=debug)


def create_api_documentation():
    """Generate API documentation in Markdown format"""
    
    documentation = """
# Podcast Transcription API Documentation

## Base URL
`http://localhost:8080/api/v1`

## Authentication
Most endpoints require an API key to be passed in the `X-API-Key` header.

## Endpoints

### Health Check
- **GET** `/health`
- Returns API health status

### Statistics
- **GET** `/stats`
- Returns system statistics

### List Podcasts
- **GET** `/podcasts`
- Query Parameters:
  - `page` (int): Page number (default: 1)
  - `per_page` (int): Items per page (default: 20, max: 100)
  - `search` (string): Search query

### List Episodes
- **GET** `/podcasts/{podcast_id}/episodes`
- Query Parameters:
  - `page` (int): Page number
  - `per_page` (int): Items per page
  - `status` (string): Filter by status

### Get Transcript
- **GET** `/transcripts/{episode_id}`
- Query Parameters:
  - `format` (string): Output format (json, txt, srt, vtt)
  - `timestamps` (bool): Include timestamps (default: true)

### Search Transcripts
- **GET** `/search`
- Query Parameters:
  - `q` (string): Search query (required, min 3 chars)
  - `limit` (int): Max results (default: 20, max: 100)

### Export Transcripts
- **POST** `/export`
- Requires API key
- Body:
  ```json
  {
    "episode_ids": [1, 2, 3],
    "format": "json"
  }
  ```

### Analytics
- **GET** `/analytics`
- Query Parameters:
  - `period` (string): Time period (24h, 7d, 30d)

## Rate Limits
- Default: 1000 requests per hour, 100 per minute
- Search: 50 requests per minute
- Export: 10 requests per minute

## Error Responses
```json
{
  "error": "Error message"
}
```

## Status Codes
- 200: Success
- 400: Bad Request
- 401: Unauthorized
- 404: Not Found
- 429: Rate Limit Exceeded
- 500: Internal Server Error
"""
    
    return documentation


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Podcast Transcription API Server')
    parser.add_argument('--db', default='data/podcast_metadata.db', help='Database path')
    parser.add_argument('--transcripts', default='data/transcripts', help='Transcripts directory')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8080, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--docs', action='store_true', help='Print API documentation')
    
    args = parser.parse_args()
    
    if args.docs:
        print(create_api_documentation())
    else:
        # Create config
        config = {
            'secret_key': os.getenv('API_SECRET_KEY', secrets.token_hex(32)),
            'cors_origins': os.getenv('CORS_ORIGINS', '*'),
        }
        
        # Initialize and run API
        api = TranscriptAPI(args.db, args.transcripts, config)
        api.run(args.host, args.port, args.debug)

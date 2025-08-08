"""
Test Suite for Podcast Transcription System
Comprehensive tests for all components
"""

import unittest
import tempfile
import shutil
import sqlite3
import json
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from podcast_fetcher import PodcastFetcher
from rss_parser import RSSParser
from audio_downloader import AudioDownloader
from transcript_processor import TranscriptProcessor


class TestPodcastFetcher(unittest.TestCase):
    """Test cases for PodcastFetcher"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            'client_id': 'test_client',
            'client_secret': 'test_secret',
            'api_url': 'https://api.podchaser.com/graphql'
        }
        
    @patch('podcast_fetcher.requests.Session')
    def test_authentication(self, mock_session):
        """Test Podchaser API authentication"""
        # Mock response
        mock_response = Mock()
        mock_response.json.return_value = {
            'data': {
                'requestAccessToken': {
                    'access_token': 'test_token_123',
                    'expires_in': 31536000
                }
            }
        }
        mock_response.raise_for_status = Mock()
        
        mock_session.return_value.post.return_value = mock_response
        
        # Test authentication
        fetcher = PodcastFetcher(self.config)
        
        self.assertIsNotNone(fetcher.access_token)
        self.assertEqual(fetcher.access_token, 'test_token_123')
    
    @patch('podcast_fetcher.PodcastFetcher.execute_query')
    def test_get_top_health_podcasts(self, mock_execute):
        """Test fetching top health podcasts"""
        # Mock GraphQL response
        mock_execute.return_value = {
            'data': {
                'podcasts': {
                    'edges': [
                        {
                            'node': {
                                'id': '123',
                                'title': 'Test Health Podcast',
                                'description': 'A test podcast about health',
                                'rssUrl': 'https://example.com/feed.xml'
                            }
                        }
                    ],
                    'pageInfo': {
                        'hasNextPage': False,
                        'endCursor': None
                    }
                }
            }
        }
        
        fetcher = PodcastFetcher(self.config)
        podcasts = fetcher.get_top_health_podcasts(limit=1)
        
        self.assertEqual(len(podcasts), 1)
        self.assertEqual(podcasts[0]['title'], 'Test Health Podcast')


class TestRSSParser(unittest.TestCase):
    """Test cases for RSSParser"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.parser = RSSParser()
        
        # Sample RSS feed content
        self.sample_rss = """<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0" xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd"
             xmlns:podcast="https://podcastindex.org/namespace/1.0">
            <channel>
                <title>Test Podcast</title>
                <description>Test Description</description>
                <item>
                    <title>Episode 1</title>
                    <description>Episode Description</description>
                    <enclosure url="https://example.com/episode1.mp3" type="audio/mpeg" length="1000000"/>
                    <guid>episode-1-guid</guid>
                    <pubDate>Mon, 01 Jan 2024 00:00:00 GMT</pubDate>
                    <itunes:duration>1800</itunes:duration>
                    <podcast:transcript url="https://example.com/transcript.srt" type="application/srt"/>
                </item>
            </channel>
        </rss>"""
    
    @patch('feedparser.parse')
    def test_parse_feed(self, mock_parse):
        """Test RSS feed parsing"""
        # Mock feedparser response
        mock_feed = Mock()
        mock_feed.bozo = False
        mock_feed.feed = Mock(
            title='Test Podcast',
            description='Test Description',
            language='en'
        )
        mock_feed.entries = [
            Mock(
                title='Episode 1',
                description='Episode Description',
                enclosures=[{
                    'href': 'https://example.com/episode1.mp3',
                    'type': 'audio/mpeg',
                    'length': 1000000
                }],
                id='episode-1-guid',
                published_parsed=time.struct_time((2024, 1, 1, 0, 0, 0, 0, 1, 0)),
                itunes_duration='30:00'
            )
        ]
        
        mock_parse.return_value = mock_feed
        
        # Test parsing
        result = self.parser.parse_feed('https://example.com/feed.xml')
        
        self.assertEqual(result['title'], 'Test Podcast')
        self.assertEqual(len(result['episodes']), 1)
        self.assertEqual(result['episodes'][0]['title'], 'Episode 1')
    
    def test_duration_parsing(self):
        """Test duration string parsing"""
        self.assertEqual(self.parser._duration_to_seconds('30:00'), 1800)
        self.assertEqual(self.parser._duration_to_seconds('1:30:00'), 5400)
        self.assertEqual(self.parser._duration_to_seconds('45'), 45)
        self.assertEqual(self.parser._duration_to_seconds(1800), 1800)


class TestAudioDownloader(unittest.TestCase):
    """Test cases for AudioDownloader"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.downloader = AudioDownloader(Path(self.temp_dir), max_workers=2)
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)
    
    def test_sanitize_filename(self):
        """Test filename sanitization"""
        self.assertEqual(
            self.downloader._sanitize_filename('Test: Episode <1>'),
            'Test_ Episode _1_'
        )
        self.assertEqual(
            self.downloader._sanitize_filename('A' * 200),
            'A' * 100
        )
    
    @patch('audio_downloader.AudioDownloader._download_with_resume')
    def test_download_episode(self, mock_download):
        """Test episode downloading"""
        mock_download.return_value = True
        
        result = self.downloader.download_episode(
            'https://example.com/episode.mp3',
            'Test Podcast',
            'Episode 1',
            'guid-123'
        )
        
        self.assertIsNotNone(result)
        self.assertTrue(Path(result).parent.exists())
    
    def test_verify_audio_file(self):
        """Test audio file verification"""
        # Create a fake MP3 file
        test_file = Path(self.temp_dir) / 'test.mp3'
        test_file.write_bytes(b'ID3' + b'\x00' * 1021)  # MP3 header
        
        self.assertTrue(self.downloader.verify_audio_file(test_file))
        
        # Test non-existent file
        self.assertFalse(self.downloader.verify_audio_file(Path('nonexistent.mp3')))
        
        # Test too small file
        small_file = Path(self.temp_dir) / 'small.mp3'
        small_file.write_bytes(b'123')
        self.assertFalse(self.downloader.verify_audio_file(small_file))


class TestTranscriptProcessor(unittest.TestCase):
    """Test cases for TranscriptProcessor"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / 'test.db'
        
        # Create test database
        self.conn = sqlite3.connect(str(self.db_path))
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE transcripts (
                id INTEGER PRIMARY KEY,
                episode_id INTEGER,
                word_count INTEGER,
                confidence_score REAL
            )
        """)
        self.conn.commit()
        
        self.processor = TranscriptProcessor(
            Path(self.temp_dir),
            self.conn
        )
    
    def tearDown(self):
        """Clean up test fixtures"""
        self.conn.close()
        shutil.rmtree(self.temp_dir)
    
    def test_save_and_load_transcript(self):
        """Test saving and loading transcripts"""
        # Test transcript
        transcript = {
            'text': 'This is a test transcript.',
            'language': 'en',
            'segments': [
                {
                    'start': 0.0,
                    'end': 2.0,
                    'text': 'This is a test',
                    'confidence': 0.95
                },
                {
                    'start': 2.0,
                    'end': 3.5,
                    'text': 'transcript.',
                    'confidence': 0.98
                }
            ]
        }
        
        metadata = {
            'duration': 3.5,
            'model': 'test_model'
        }
        
        # Save transcript
        file_path = self.processor.save_transcript(1, transcript, metadata)
        self.assertTrue(Path(file_path).exists())
        
        # Load transcript
        loaded = self.processor.load_transcript(file_path)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded['text'], transcript['text'])
        self.assertEqual(len(loaded['segments']), 2)
    
    def test_export_formats(self):
        """Test exporting transcripts in different formats"""
        transcript = {
            'text': 'Test transcript text.',
            'segments': [
                {'start': 0, 'end': 2, 'text': 'Test transcript'},
                {'start': 2, 'end': 3, 'text': 'text.'}
            ]
        }
        
        # Test text export
        text_export = self.processor._export_as_text(transcript)
        self.assertEqual(text_export, 'Test transcript text.')
        
        # Test SRT export
        srt_export = self.processor._export_as_srt(transcript)
        self.assertIn('-->', srt_export)
        self.assertIn('Test transcript', srt_export)
        
        # Test VTT export
        vtt_export = self.processor._export_as_vtt(transcript)
        self.assertIn('WEBVTT', vtt_export)
        self.assertIn('-->', vtt_export)
    
    def test_compression_ratio(self):
        """Test compression efficiency"""
        # Create a large transcript
        large_transcript = {
            'text': 'This is a test. ' * 1000,
            'segments': [
                {
                    'start': i,
                    'end': i + 1,
                    'text': 'This is a test.',
                    'confidence': 0.95
                }
                for i in range(100)
            ]
        }
        
        # Save and check compression
        file_path = self.processor.save_transcript(2, large_transcript)
        
        stats = self.processor.get_statistics()
        self.assertGreater(stats['compression_ratio'], 1.0)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.temp_dir) / 'data'
        self.data_dir.mkdir()
        
        # Create subdirectories
        (self.data_dir / 'audio').mkdir()
        (self.data_dir / 'transcripts').mkdir()
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir)
    
    @patch('podcast_fetcher.PodcastFetcher.get_top_health_podcasts')
    @patch('rss_parser.RSSParser.parse_feed')
    @patch('audio_downloader.AudioDownloader.download_episode')
    def test_pipeline_phase1_and_2(self, mock_download, mock_parse, mock_fetch):
        """Test metadata fetching and downloading phases"""
        
        # Mock Podchaser response
        mock_fetch.return_value = [
            {
                'podchaser_id': 'test_1',
                'title': 'Test Podcast 1',
                'rss_url': 'https://example.com/feed1.xml'
            }
        ]
        
        # Mock RSS response
        mock_parse.return_value = {
            'episodes': [
                {
                    'guid': 'ep1',
                    'title': 'Episode 1',
                    'audio_url': 'https://example.com/ep1.mp3',
                    'duration_seconds': 1800
                }
            ]
        }
        
        # Mock download response
        mock_download.return_value = self.data_dir / 'audio' / 'test.mp3'
        
        # Import main pipeline
        from podcast_transcription_main import PodcastPipeline
        
        # Create config
        config = {
            'podchaser': {
                'client_id': 'test',
                'client_secret': 'test'
            },
            'download': {
                'max_workers': 1,
                'max_episodes_per_podcast': 1
            },
            'transcription': {
                'batch_size': 1,
                'num_gpus': 1
            },
            'processing': {
                'max_parallel_podcasts': 1
            }
        }
        
        # Save config
        config_path = self.temp_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f)
        
        # Initialize pipeline with test config
        with patch('podcast_transcription_main.DATA_DIR', self.data_dir):
            with patch('podcast_transcription_main.DB_PATH', self.data_dir / 'test.db'):
                pipeline = PodcastPipeline(config_path)
                
                # Run phase 1
                podcasts = pipeline.phase1_fetch_metadata(limit=1)
                self.assertEqual(len(podcasts), 1)
                
                # Run phase 2
                stats = pipeline.phase2_download_audio(max_episodes_per_podcast=1)
                self.assertIn('total_episodes', stats)


class TestPerformance(unittest.TestCase):
    """Performance and stress tests"""
    
    def test_large_transcript_handling(self):
        """Test handling of large transcripts"""
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Create database
            conn = sqlite3.connect(':memory:')
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE transcripts (
                    id INTEGER PRIMARY KEY,
                    episode_id INTEGER,
                    word_count INTEGER,
                    confidence_score REAL
                )
            """)
            
            processor = TranscriptProcessor(Path(temp_dir), conn)
            
            # Create a very large transcript
            large_text = ' '.join(['word'] * 100000)  # 100k words
            large_transcript = {
                'text': large_text,
                'segments': [
                    {
                        'start': i * 10,
                        'end': (i + 1) * 10,
                        'text': ' '.join(['word'] * 100)
                    }
                    for i in range(1000)
                ]
            }
            
            # Measure save time
            start_time = time.time()
            file_path = processor.save_transcript(1, large_transcript)
            save_time = time.time() - start_time
            
            # Measure load time
            start_time = time.time()
            loaded = processor.load_transcript(file_path)
            load_time = time.time() - start_time
            
            # Check performance
            self.assertLess(save_time, 5.0)  # Should save in less than 5 seconds
            self.assertLess(load_time, 2.0)  # Should load in less than 2 seconds
            self.assertIsNotNone(loaded)
            
            conn.close()
            
        finally:
            shutil.rmtree(temp_dir)
    
    def test_concurrent_downloads(self):
        """Test concurrent download handling"""
        temp_dir = tempfile.mkdtemp()
        
        try:
            downloader = AudioDownloader(Path(temp_dir), max_workers=4)
            
            # Create batch of episodes
            episodes = [
                {
                    'audio_url': f'https://example.com/episode{i}.mp3',
                    'podcast_title': 'Test Podcast',
                    'episode_title': f'Episode {i}',
                    'episode_guid': f'guid-{i}'
                }
                for i in range(10)
            ]
            
            # Mock download function
            with patch.object(downloader, 'download_episode') as mock_download:
                mock_download.return_value = Path(temp_dir) / 'test.mp3'
                
                # Test batch download
                start_time = time.time()
                results = downloader.download_episodes_batch(episodes)
                batch_time = time.time() - start_time
                
                # Should process in parallel
                self.assertEqual(len(results), 10)
                self.assertEqual(mock_download.call_count, 10)
                
        finally:
            shutil.rmtree(temp_dir)


def run_tests():
    """Run all tests with coverage report"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestPodcastFetcher))
    suite.addTests(loader.loadTestsFromTestCase(TestRSSParser))
    suite.addTests(loader.loadTestsFromTestCase(TestAudioDownloader))
    suite.addTests(loader.loadTestsFromTestCase(TestTranscriptProcessor))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformance))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)

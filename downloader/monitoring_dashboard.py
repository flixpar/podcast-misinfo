"""
Monitoring Dashboard Module
Real-time monitoring and web dashboard for the transcription pipeline
"""

import os
import json
import sqlite3
import logging
import psutil
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from collections import deque
import time

# Web framework imports
from flask import Flask, render_template, jsonify, request, send_file
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import plotly.graph_objs as go
import plotly.utils

# GPU monitoring
try:
    import pynvml
    pynvml.nvmlInit()
    NVIDIA_SMI_AVAILABLE = True
except:
    NVIDIA_SMI_AVAILABLE = False

logger = logging.getLogger(__name__)


class SystemMonitor:
    """Monitor system resources and pipeline progress"""
    
    def __init__(self, db_path: str, data_dir: str):
        """
        Initialize system monitor
        
        Args:
            db_path: Path to SQLite database
            data_dir: Path to data directory
        """
        self.db_path = db_path
        self.data_dir = Path(data_dir)
        
        # Performance metrics buffer
        self.metrics_buffer = deque(maxlen=1000)
        self.gpu_metrics = {i: deque(maxlen=100) for i in range(4)}
        
        # Initialize monitoring
        self.start_time = time.time()
        self.last_update = time.time()
    
    def get_system_stats(self) -> Dict:
        """Get current system statistics"""
        stats = {
            'timestamp': datetime.now().isoformat(),
            'uptime_hours': (time.time() - self.start_time) / 3600
        }
        
        # CPU stats
        stats['cpu'] = {
            'percent': psutil.cpu_percent(interval=1),
            'cores': psutil.cpu_count(),
            'frequency': psutil.cpu_freq().current if psutil.cpu_freq() else 0
        }
        
        # Memory stats
        mem = psutil.virtual_memory()
        stats['memory'] = {
            'total_gb': mem.total / (1024**3),
            'used_gb': mem.used / (1024**3),
            'available_gb': mem.available / (1024**3),
            'percent': mem.percent
        }
        
        # Disk stats
        disk = psutil.disk_usage(str(self.data_dir))
        stats['disk'] = {
            'total_gb': disk.total / (1024**3),
            'used_gb': disk.used / (1024**3),
            'free_gb': disk.free / (1024**3),
            'percent': disk.percent
        }
        
        # GPU stats
        if NVIDIA_SMI_AVAILABLE:
            stats['gpus'] = self._get_gpu_stats()
        else:
            stats['gpus'] = []
        
        # Network stats
        net = psutil.net_io_counters()
        stats['network'] = {
            'bytes_sent_gb': net.bytes_sent / (1024**3),
            'bytes_recv_gb': net.bytes_recv / (1024**3),
            'packets_sent': net.packets_sent,
            'packets_recv': net.packets_recv
        }
        
        return stats
    
    def _get_gpu_stats(self) -> List[Dict]:
        """Get NVIDIA GPU statistics"""
        gpu_stats = []
        
        if not NVIDIA_SMI_AVAILABLE:
            return gpu_stats
        
        try:
            device_count = pynvml.nvmlDeviceGetCount()
            
            for i in range(min(device_count, 4)):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                
                # Get GPU information
                name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
                
                gpu_stat = {
                    'id': i,
                    'name': name,
                    'memory': {
                        'total_gb': mem_info.total / (1024**3),
                        'used_gb': mem_info.used / (1024**3),
                        'free_gb': mem_info.free / (1024**3),
                        'percent': (mem_info.used / mem_info.total) * 100
                    },
                    'utilization': util.gpu,
                    'temperature': temp,
                    'power_watts': power
                }
                
                gpu_stats.append(gpu_stat)
                
                # Store in buffer for history
                self.gpu_metrics[i].append({
                    'timestamp': time.time(),
                    'util': util.gpu,
                    'memory': (mem_info.used / mem_info.total) * 100,
                    'temp': temp,
                    'power': power
                })
                
        except Exception as e:
            logger.error(f"Failed to get GPU stats: {str(e)}")
        
        return gpu_stats
    
    def get_pipeline_stats(self) -> Dict:
        """Get pipeline processing statistics from database"""
        stats = {}
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Podcast statistics
            cursor.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) as pending,
                    SUM(CASE WHEN status = 'downloaded' THEN 1 ELSE 0 END) as downloaded,
                    SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) as errors
                FROM podcasts
            """)
            result = cursor.fetchone()
            stats['podcasts'] = {
                'total': result[0] or 0,
                'pending': result[1] or 0,
                'downloaded': result[2] or 0,
                'errors': result[3] or 0
            }
            
            # Episode statistics
            cursor.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) as pending,
                    SUM(CASE WHEN status = 'downloaded' THEN 1 ELSE 0 END) as downloaded,
                    SUM(CASE WHEN status = 'transcribed' THEN 1 ELSE 0 END) as transcribed,
                    SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) as errors,
                    SUM(CASE WHEN has_rss_transcript = 1 THEN 1 ELSE 0 END) as has_rss_transcript
                FROM episodes
            """)
            result = cursor.fetchone()
            stats['episodes'] = {
                'total': result[0] or 0,
                'pending': result[1] or 0,
                'downloaded': result[2] or 0,
                'transcribed': result[3] or 0,
                'errors': result[4] or 0,
                'has_rss_transcript': result[5] or 0
            }
            
            # Transcript statistics
            cursor.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(word_count) as total_words,
                    AVG(word_count) as avg_words,
                    AVG(confidence_score) as avg_confidence,
                    SUM(duration_seconds) as total_duration_seconds
                FROM transcripts
            """)
            result = cursor.fetchone()
            stats['transcripts'] = {
                'total': result[0] or 0,
                'total_words': result[1] or 0,
                'avg_words': result[2] or 0,
                'avg_confidence': result[3] or 0,
                'total_hours_transcribed': (result[4] or 0) / 3600
            }
            
            # Recent activity
            cursor.execute("""
                SELECT 
                    phase,
                    status,
                    COUNT(*) as count,
                    AVG(duration_seconds) as avg_duration
                FROM processing_logs
                WHERE started_at > datetime('now', '-1 hour')
                GROUP BY phase, status
            """)
            recent_activity = []
            for row in cursor.fetchall():
                recent_activity.append({
                    'phase': row[0],
                    'status': row[1],
                    'count': row[2],
                    'avg_duration': row[3]
                })
            stats['recent_activity'] = recent_activity
            
            # Processing rate
            cursor.execute("""
                SELECT 
                    COUNT(*) as episodes_last_hour
                FROM episodes
                WHERE transcribed_at > datetime('now', '-1 hour')
            """)
            episodes_per_hour = cursor.fetchone()[0] or 0
            stats['processing_rate'] = {
                'episodes_per_hour': episodes_per_hour,
                'estimated_completion_hours': (
                    (stats['episodes']['downloaded'] - stats['episodes']['transcribed']) / 
                    max(episodes_per_hour, 1)
                )
            }
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to get pipeline stats: {str(e)}")
            
        return stats
    
    def get_storage_stats(self) -> Dict:
        """Get storage usage statistics"""
        stats = {}
        
        # Audio directory
        audio_dir = self.data_dir / 'audio'
        if audio_dir.exists():
            audio_files = list(audio_dir.rglob('*'))
            audio_size = sum(f.stat().st_size for f in audio_files if f.is_file())
            stats['audio'] = {
                'file_count': len([f for f in audio_files if f.is_file()]),
                'total_gb': audio_size / (1024**3),
                'avg_mb': (audio_size / len(audio_files) / (1024**2)) if audio_files else 0
            }
        
        # Transcript directory
        transcript_dir = self.data_dir / 'transcripts'
        if transcript_dir.exists():
            transcript_files = list(transcript_dir.rglob('*.jsonl.zst'))
            transcript_size = sum(f.stat().st_size for f in transcript_files)
            stats['transcripts'] = {
                'file_count': len(transcript_files),
                'total_mb': transcript_size / (1024**2),
                'avg_kb': (transcript_size / len(transcript_files) / 1024) if transcript_files else 0
            }
        
        # Database size
        db_path = Path(self.db_path)
        if db_path.exists():
            stats['database'] = {
                'size_mb': db_path.stat().st_size / (1024**2)
            }
        
        return stats
    
    def get_performance_metrics(self) -> Dict:
        """Get performance metrics and predictions"""
        metrics = {
            'current': self.get_system_stats(),
            'pipeline': self.get_pipeline_stats(),
            'storage': self.get_storage_stats()
        }
        
        # Calculate additional metrics
        if metrics['pipeline']['episodes']['total'] > 0:
            metrics['progress'] = {
                'overall_percent': (
                    metrics['pipeline']['episodes']['transcribed'] / 
                    metrics['pipeline']['episodes']['total'] * 100
                ),
                'download_percent': (
                    metrics['pipeline']['episodes']['downloaded'] / 
                    metrics['pipeline']['episodes']['total'] * 100
                )
            }
        
        # GPU efficiency
        if NVIDIA_SMI_AVAILABLE and self.gpu_metrics[0]:
            gpu_efficiency = []
            for gpu_id, buffer in self.gpu_metrics.items():
                if buffer:
                    recent = list(buffer)[-20:]  # Last 20 measurements
                    avg_util = sum(m['util'] for m in recent) / len(recent)
                    avg_memory = sum(m['memory'] for m in recent) / len(recent)
                    gpu_efficiency.append({
                        'gpu_id': gpu_id,
                        'avg_utilization': avg_util,
                        'avg_memory': avg_memory
                    })
            metrics['gpu_efficiency'] = gpu_efficiency
        
        return metrics


class DashboardApp:
    """Web dashboard application"""
    
    def __init__(self, monitor: SystemMonitor, host: str = '0.0.0.0', port: int = 5000):
        """
        Initialize dashboard app
        
        Args:
            monitor: SystemMonitor instance
            host: Host to bind to
            port: Port to bind to
        """
        self.monitor = monitor
        self.host = host
        self.port = port
        
        # Initialize Flask app
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'podcast-transcription-secret'
        CORS(self.app)
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # Setup routes
        self._setup_routes()
        
        # Background task for real-time updates
        self.update_thread = None
    
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def index():
            """Main dashboard page"""
            return render_template('dashboard.html')
        
        @self.app.route('/api/stats')
        def get_stats():
            """Get current statistics"""
            return jsonify(self.monitor.get_performance_metrics())
        
        @self.app.route('/api/system')
        def get_system():
            """Get system statistics"""
            return jsonify(self.monitor.get_system_stats())
        
        @self.app.route('/api/pipeline')
        def get_pipeline():
            """Get pipeline statistics"""
            return jsonify(self.monitor.get_pipeline_stats())
        
        @self.app.route('/api/gpu/<int:gpu_id>')
        def get_gpu_history(gpu_id):
            """Get GPU history data"""
            if gpu_id in self.monitor.gpu_metrics:
                data = list(self.monitor.gpu_metrics[gpu_id])
                return jsonify(data)
            return jsonify([])
        
        @self.app.route('/api/episodes/recent')
        def get_recent_episodes():
            """Get recently processed episodes"""
            try:
                conn = sqlite3.connect(self.monitor.db_path)
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT 
                        e.id,
                        e.title,
                        p.title as podcast_title,
                        e.status,
                        e.duration_seconds,
                        e.transcribed_at,
                        t.word_count,
                        t.confidence_score
                    FROM episodes e
                    JOIN podcasts p ON e.podcast_id = p.id
                    LEFT JOIN transcripts t ON t.episode_id = e.id
                    ORDER BY e.transcribed_at DESC
                    LIMIT 20
                """)
                
                episodes = []
                for row in cursor.fetchall():
                    episodes.append({
                        'id': row[0],
                        'title': row[1],
                        'podcast': row[2],
                        'status': row[3],
                        'duration': row[4],
                        'transcribed_at': row[5],
                        'word_count': row[6],
                        'confidence': row[7]
                    })
                
                conn.close()
                return jsonify(episodes)
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/export/<format>')
        def export_data(format):
            """Export transcripts in various formats"""
            if format not in ['csv', 'json', 'txt']:
                return jsonify({'error': 'Invalid format'}), 400
            
            # Implementation would export data based on format
            # This is a placeholder
            return jsonify({'message': f'Export to {format} not yet implemented'}), 501
        
        @self.socketio.on('connect')
        def handle_connect():
            """Handle client connection"""
            logger.info('Client connected to dashboard')
            emit('connected', {'data': 'Connected to monitoring dashboard'})
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection"""
            logger.info('Client disconnected from dashboard')
    
    def start_background_updates(self):
        """Start background thread for real-time updates"""
        def update_loop():
            while True:
                try:
                    # Get current metrics
                    metrics = self.monitor.get_performance_metrics()
                    
                    # Emit to all connected clients
                    self.socketio.emit('update', metrics)
                    
                    # Sleep for update interval
                    self.socketio.sleep(5)
                    
                except Exception as e:
                    logger.error(f"Error in update loop: {str(e)}")
                    self.socketio.sleep(5)
        
        self.update_thread = self.socketio.start_background_task(update_loop)
    
    def run(self):
        """Run the dashboard application"""
        logger.info(f"Starting dashboard on {self.host}:{self.port}")
        
        # Start background updates
        self.start_background_updates()
        
        # Run the app
        self.socketio.run(
            self.app,
            host=self.host,
            port=self.port,
            debug=False,
            use_reloader=False
        )


# HTML template for dashboard (would normally be in templates/dashboard.html)
DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Podcast Transcription Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js"></script>
    <script src="https://cdn.socket.io/4.5.0/socket.io.min.js"></script>
    <style>
        .metric-card {
            padding: 20px;
            margin: 10px 0;
            border-radius: 10px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        .gpu-card {
            padding: 15px;
            margin: 10px 0;
            border-radius: 8px;
            background: #f8f9fa;
        }
        .progress-ring {
            transform: rotate(-90deg);
        }
        .chart-container {
            position: relative;
            height: 300px;
            margin: 20px 0;
        }
    </style>
</head>
<body>
    <nav class="navbar navbar-dark bg-dark">
        <div class="container-fluid">
            <span class="navbar-brand mb-0 h1">🎙️ Podcast Transcription Dashboard</span>
            <span class="navbar-text" id="connection-status">
                <span class="badge bg-success">Connected</span>
            </span>
        </div>
    </nav>
    
    <div class="container-fluid mt-3">
        <!-- Overview Cards -->
        <div class="row">
            <div class="col-md-3">
                <div class="metric-card">
                    <h5>Total Podcasts</h5>
                    <h2 id="total-podcasts">-</h2>
                    <small>Processing Status</small>
                    <div class="progress mt-2">
                        <div id="podcast-progress" class="progress-bar" style="width: 0%"></div>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="metric-card">
                    <h5>Episodes</h5>
                    <h2 id="total-episodes">-</h2>
                    <small id="episodes-status">-</small>
                    <div class="progress mt-2">
                        <div id="episode-progress" class="progress-bar" style="width: 0%"></div>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="metric-card">
                    <h5>Transcriptions</h5>
                    <h2 id="total-transcripts">-</h2>
                    <small id="transcript-rate">- per hour</small>
                </div>
            </div>
            <div class="col-md-3">
                <div class="metric-card">
                    <h5>Storage Used</h5>
                    <h2 id="storage-used">-</h2>
                    <small id="storage-details">-</small>
                </div>
            </div>
        </div>
        
        <!-- System Metrics -->
        <div class="row mt-4">
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">System Resources</div>
                    <div class="card-body">
                        <div class="chart-container">
                            <canvas id="system-chart"></canvas>
                        </div>
                    </div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">Processing Rate</div>
                    <div class="card-body">
                        <div class="chart-container">
                            <canvas id="rate-chart"></canvas>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- GPU Status -->
        <div class="row mt-4">
            <div class="col-12">
                <div class="card">
                    <div class="card-header">GPU Status</div>
                    <div class="card-body">
                        <div class="row" id="gpu-container">
                            <!-- GPU cards will be inserted here -->
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Recent Episodes -->
        <div class="row mt-4">
            <div class="col-12">
                <div class="card">
                    <div class="card-header">Recent Transcriptions</div>
                    <div class="card-body">
                        <table class="table table-striped">
                            <thead>
                                <tr>
                                    <th>Podcast</th>
                                    <th>Episode</th>
                                    <th>Status</th>
                                    <th>Duration</th>
                                    <th>Words</th>
                                    <th>Confidence</th>
                                </tr>
                            </thead>
                            <tbody id="recent-episodes">
                                <!-- Episodes will be inserted here -->
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Initialize Socket.IO connection
        const socket = io();
        
        // Charts
        let systemChart, rateChart;
        
        // Initialize charts
        function initCharts() {
            // System resources chart
            const systemCtx = document.getElementById('system-chart').getContext('2d');
            systemChart = new Chart(systemCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'CPU %',
                        data: [],
                        borderColor: 'rgb(255, 99, 132)',
                        tension: 0.1
                    }, {
                        label: 'Memory %',
                        data: [],
                        borderColor: 'rgb(54, 162, 235)',
                        tension: 0.1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 100
                        }
                    }
                }
            });
            
            // Processing rate chart
            const rateCtx = document.getElementById('rate-chart').getContext('2d');
            rateChart = new Chart(rateCtx, {
                type: 'bar',
                data: {
                    labels: ['Downloaded', 'Transcribed', 'Errors'],
                    datasets: [{
                        label: 'Episodes',
                        data: [0, 0, 0],
                        backgroundColor: [
                            'rgba(75, 192, 192, 0.6)',
                            'rgba(54, 162, 235, 0.6)',
                            'rgba(255, 99, 132, 0.6)'
                        ]
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false
                }
            });
        }
        
        // Update dashboard with new data
        function updateDashboard(data) {
            // Update overview cards
            if (data.pipeline) {
                document.getElementById('total-podcasts').textContent = data.pipeline.podcasts.total;
                document.getElementById('total-episodes').textContent = data.pipeline.episodes.total;
                document.getElementById('total-transcripts').textContent = data.pipeline.transcripts.total;
                
                // Update progress bars
                const podcastProgress = (data.pipeline.podcasts.downloaded / data.pipeline.podcasts.total) * 100;
                document.getElementById('podcast-progress').style.width = podcastProgress + '%';
                
                const episodeProgress = (data.pipeline.episodes.transcribed / data.pipeline.episodes.total) * 100;
                document.getElementById('episode-progress').style.width = episodeProgress + '%';
                
                // Update rates
                document.getElementById('transcript-rate').textContent = 
                    data.pipeline.processing_rate.episodes_per_hour + ' per hour';
                
                // Update chart data
                rateChart.data.datasets[0].data = [
                    data.pipeline.episodes.downloaded,
                    data.pipeline.episodes.transcribed,
                    data.pipeline.episodes.errors
                ];
                rateChart.update();
            }
            
            // Update storage
            if (data.storage) {
                const totalStorage = (data.storage.audio?.total_gb || 0) + 
                                    (data.storage.transcripts?.total_mb || 0) / 1024;
                document.getElementById('storage-used').textContent = totalStorage.toFixed(2) + ' GB';
                document.getElementById('storage-details').textContent = 
                    `${data.storage.audio?.file_count || 0} audio, ${data.storage.transcripts?.file_count || 0} transcripts`;
            }
            
            // Update system charts
            if (data.current) {
                const time = new Date().toLocaleTimeString();
                systemChart.data.labels.push(time);
                systemChart.data.datasets[0].data.push(data.current.cpu.percent);
                systemChart.data.datasets[1].data.push(data.current.memory.percent);
                
                // Keep only last 20 points
                if (systemChart.data.labels.length > 20) {
                    systemChart.data.labels.shift();
                    systemChart.data.datasets[0].data.shift();
                    systemChart.data.datasets[1].data.shift();
                }
                systemChart.update();
                
                // Update GPU cards
                updateGPUCards(data.current.gpus);
            }
        }
        
        // Update GPU cards
        function updateGPUCards(gpus) {
            const container = document.getElementById('gpu-container');
            container.innerHTML = '';
            
            gpus.forEach(gpu => {
                const card = document.createElement('div');
                card.className = 'col-md-3';
                card.innerHTML = `
                    <div class="gpu-card">
                        <h6>GPU ${gpu.id}: ${gpu.name}</h6>
                        <div class="progress mb-2">
                            <div class="progress-bar bg-success" style="width: ${gpu.utilization}%">
                                ${gpu.utilization}% Util
                            </div>
                        </div>
                        <div class="progress mb-2">
                            <div class="progress-bar bg-info" style="width: ${gpu.memory.percent}%">
                                ${gpu.memory.percent.toFixed(1)}% Memory
                            </div>
                        </div>
                        <small>
                            Temp: ${gpu.temperature}°C | 
                            Power: ${gpu.power_watts.toFixed(1)}W
                        </small>
                    </div>
                `;
                container.appendChild(card);
            });
        }
        
        // Socket.IO event handlers
        socket.on('connect', function() {
            console.log('Connected to dashboard');
            document.getElementById('connection-status').innerHTML = 
                '<span class="badge bg-success">Connected</span>';
        });
        
        socket.on('disconnect', function() {
            console.log('Disconnected from dashboard');
            document.getElementById('connection-status').innerHTML = 
                '<span class="badge bg-danger">Disconnected</span>';
        });
        
        socket.on('update', function(data) {
            updateDashboard(data);
        });
        
        // Load recent episodes
        async function loadRecentEpisodes() {
            try {
                const response = await fetch('/api/episodes/recent');
                const episodes = await response.json();
                
                const tbody = document.getElementById('recent-episodes');
                tbody.innerHTML = '';
                
                episodes.forEach(episode => {
                    const row = tbody.insertRow();
                    row.innerHTML = `
                        <td>${episode.podcast}</td>
                        <td>${episode.title}</td>
                        <td><span class="badge bg-${getStatusColor(episode.status)}">${episode.status}</span></td>
                        <td>${formatDuration(episode.duration)}</td>
                        <td>${episode.word_count || '-'}</td>
                        <td>${episode.confidence ? (episode.confidence * 100).toFixed(1) + '%' : '-'}</td>
                    `;
                });
            } catch (error) {
                console.error('Failed to load recent episodes:', error);
            }
        }
        
        function getStatusColor(status) {
            switch(status) {
                case 'transcribed': return 'success';
                case 'downloaded': return 'info';
                case 'pending': return 'warning';
                case 'error': return 'danger';
                default: return 'secondary';
            }
        }
        
        function formatDuration(seconds) {
            if (!seconds) return '-';
            const hours = Math.floor(seconds / 3600);
            const minutes = Math.floor((seconds % 3600) / 60);
            return hours > 0 ? `${hours}h ${minutes}m` : `${minutes}m`;
        }
        
        // Initialize on load
        window.onload = function() {
            initCharts();
            loadRecentEpisodes();
            
            // Refresh recent episodes every 30 seconds
            setInterval(loadRecentEpisodes, 30000);
        };
    </script>
</body>
</html>
"""


def run_dashboard(db_path: str, data_dir: str, host: str = '0.0.0.0', port: int = 5000):
    """
    Run the monitoring dashboard
    
    Args:
        db_path: Path to SQLite database
        data_dir: Path to data directory
        host: Host to bind to
        port: Port to bind to
    """
    # Create monitor
    monitor = SystemMonitor(db_path, data_dir)
    
    # Create and run dashboard
    dashboard = DashboardApp(monitor, host, port)
    
    # Create templates directory and save HTML
    templates_dir = Path('templates')
    templates_dir.mkdir(exist_ok=True)
    
    with open(templates_dir / 'dashboard.html', 'w') as f:
        f.write(DASHBOARD_HTML)
    
    logger.info(f"Starting dashboard on http://{host}:{port}")
    dashboard.run()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Podcast Transcription Monitoring Dashboard')
    parser.add_argument('--db', default='data/podcast_metadata.db', help='Database path')
    parser.add_argument('--data-dir', default='data', help='Data directory path')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    
    args = parser.parse_args()
    
    run_dashboard(args.db, args.data_dir, args.host, args.port)

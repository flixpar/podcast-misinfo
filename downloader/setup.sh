#!/bin/bash

# Podcast Transcription System Setup Script

echo "Setting up Podcast Transcription System..."

# Check for uv
if ! command -v uv &> /dev/null; then
    echo "uv could not be found. Please install uv."
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
uv venv

# Activate virtual environment
source .venv/bin/activate

# Install pytorch
echo "Installing pytorch..."
uv pip install torch torchaudio --torch-backend=auto

# Install requirements
echo "Installing requirements..."
uv pip install -r requirements.txt

# Install NeMo
uv pip install nemo_toolkit[asr]

# Create directory structure
echo "Creating directory structure..."
mkdir -p data/audio
mkdir -p data/transcripts
mkdir -p logs
mkdir -p temp

# Download Parakeet model (optional - will be downloaded on first use)
echo "Model will be downloaded on first use from Hugging Face"

# Check CUDA availability
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA devices: {torch.cuda.device_count()}')"

# Create config file if not exists
if [ ! -f config.json ]; then
    echo "Creating config.json template..."
    cat > config.json << 'EOF'
{
    "podchaser": {
        "client_id": "YOUR_PODCHASER_CLIENT_ID",
        "client_secret": "YOUR_PODCHASER_CLIENT_SECRET",
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
        "max_duration_seconds": 1440,
        "enable_timestamps": true,
        "language": "en"
    },
    "processing": {
        "max_parallel_podcasts": 4,
        "checkpoint_interval": 10
    }
}
EOF
    echo "Please edit config.json with your API credentials"
fi

echo "Setup complete!"
echo ""
echo "To activate the environment, run: source venv/bin/activate"
echo "To run the pipeline: python main.py --help"
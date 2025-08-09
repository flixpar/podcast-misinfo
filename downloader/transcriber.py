"""
Transcriber Module
NVIDIA Parakeet TDT 0.6B V2 integration with multi-GPU support
"""

import os
import logging
import time
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
from queue import Queue
import json

# NeMo imports
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.parts.utils.manifest_utils import write_manifest

# Audio processing
import librosa
import soundfile as sf
from pydub import AudioSegment

logger = logging.getLogger(__name__)


class ParakeetTranscriber:
    """
    NVIDIA Parakeet TDT transcriber with multi-GPU support
    Optimized for 4x NVIDIA H100 GPUs
    """
    
    def __init__(self, model_name: str = "nvidia/parakeet-tdt-0.6b-v2",
                 batch_size: int = 8,
                 gpu_ids: Optional[List[int]] = None,
                 chunk_duration_seconds: int = 1200,
                 overlap_seconds: int = 30):
        """
        Initialize the Parakeet transcriber
        
        Args:
            model_name: Name or path of the Parakeet model
            batch_size: Batch size for inference
            gpu_ids: List of GPU IDs to use (default: [0,1,2,3])
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.gpu_ids = gpu_ids or [0, 1, 2, 3]
        
        # Model will be loaded on each GPU in separate processes
        self.models = {}
        self.model_loaded = False
        
        # Audio processing parameters
        self.sample_rate = 16000  # Parakeet expects 16kHz
        self.max_duration = int(chunk_duration_seconds)
        self.chunk_duration = int(chunk_duration_seconds)
        self.overlap_duration = int(overlap_seconds)
        
        # Initialize models on GPUs
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize models on specified GPUs"""
        logger.info(f"Initializing Parakeet models on GPUs: {self.gpu_ids}")
        
        for gpu_id in self.gpu_ids:
            try:
                # Set CUDA device
                torch.cuda.set_device(gpu_id)
                
                # Load model
                logger.info(f"Loading model on GPU {gpu_id}")
                model = ASRModel.from_pretrained(
                    model_name=self.model_name,
                    map_location=f'cuda:{gpu_id}'
                )
                
                # Set to evaluation mode
                model.eval()
                
                # Store model
                self.models[gpu_id] = model
                
                logger.info(f"Successfully loaded model on GPU {gpu_id}")
                
            except Exception as e:
                logger.exception(f"Failed to load model on GPU {gpu_id}: {str(e)}")
                raise
        
        self.model_loaded = True
    
    def transcribe_batch(self, batch_data: List[Dict]) -> List[Dict]:
        """
        Transcribe a batch of audio files using multiple GPUs
        
        Args:
            batch_data: List of dictionaries with audio file information
                Each dict should have: id, audio_path, title, podcast_title
                
        Returns:
            List of transcription results
        """
        if not self.model_loaded:
            raise RuntimeError("Models not initialized")
        
        logger.info(f"Transcribing batch of {len(batch_data)} files")
        
        # Prepare audio files
        prepared_batch = self._prepare_batch(batch_data)
        
        # Distribute work across GPUs
        results = self._parallel_transcribe(prepared_batch)
        
        return results
    
    def _prepare_batch(self, batch_data: List[Dict]) -> List[Dict]:
        """Prepare audio files for transcription"""
        prepared = []
        
        for item in batch_data:
            audio_path = Path(item['audio_path'])
            
            if not audio_path.exists():
                logger.warning(f"Audio file not found: {audio_path}")
                continue
            
            try:
                # Load and preprocess audio
                audio_info = self._preprocess_audio(audio_path)
                
                prepared.append({
                    'episode_id': item['id'],
                    'original_path': str(audio_path),
                    'processed_path': audio_info['processed_path'],
                    'duration': audio_info['duration'],
                    'sample_rate': audio_info['sample_rate'],
                    'title': item['title'],
                    'podcast_title': item['podcast_title']
                })
                
            except Exception as e:
                logger.exception(f"Failed to prepare audio {audio_path}: {str(e)}")
                continue
        
        return prepared
    
    def _preprocess_audio(self, audio_path: Path) -> Dict:
        """
        Preprocess audio file for transcription
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with processed audio information
        """
        # Create temporary processed audio directory
        temp_dir = Path('/tmp/processed_audio')
        temp_dir.mkdir(exist_ok=True)
        
        # Output path for processed audio
        processed_path = temp_dir / f"{audio_path.stem}_16khz.wav"
        
        # Check if already processed
        if processed_path.exists():
            # Load to get duration
            audio, sr = librosa.load(str(processed_path), sr=None)
            duration = len(audio) / sr
            
            return {
                'processed_path': str(processed_path),
                'duration': duration,
                'sample_rate': sr
            }
        
        logger.debug(f"Preprocessing audio: {audio_path}")
        
        try:
            # Load audio with librosa (handles various formats)
            audio, original_sr = librosa.load(str(audio_path), sr=None)
            
            # Resample to 16kHz if needed
            if original_sr != self.sample_rate:
                logger.debug(f"Resampling from {original_sr}Hz to {self.sample_rate}Hz")
                audio = librosa.resample(audio, orig_sr=original_sr, target_sr=self.sample_rate)
                sr = self.sample_rate
            else:
                sr = original_sr
            
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            # Normalize audio
            audio = audio / np.max(np.abs(audio))
            
            # Save processed audio
            sf.write(processed_path, audio, sr)
            
            duration = len(audio) / sr
            
            return {
                'processed_path': str(processed_path),
                'duration': duration,
                'sample_rate': sr
            }
            
        except Exception as e:
            logger.exception(f"Audio preprocessing failed: {str(e)}")
            
            # Try alternative method with pydub
            try:
                audio_segment = AudioSegment.from_file(str(audio_path))
                audio_segment = audio_segment.set_frame_rate(self.sample_rate)
                audio_segment = audio_segment.set_channels(1)  # Convert to mono
                
                audio_segment.export(processed_path, format='wav')
                
                duration = len(audio_segment) / 1000.0  # Convert ms to seconds
                
                return {
                    'processed_path': str(processed_path),
                    'duration': duration,
                    'sample_rate': self.sample_rate
                }
                
            except Exception as e2:
                logger.exception(f"Alternative preprocessing also failed: {str(e2)}")
                raise
    
    def _parallel_transcribe(self, prepared_batch: List[Dict]) -> List[Dict]:
        """Distribute transcription across multiple GPUs"""
        
        # Split batch across GPUs
        gpu_batches = self._distribute_batch(prepared_batch)
        
        results = []
        
        # Use ThreadPoolExecutor for GPU parallelization
        with ThreadPoolExecutor(max_workers=len(self.gpu_ids)) as executor:
            futures = []
            
            for gpu_id, gpu_batch in gpu_batches.items():
                if gpu_batch:
                    future = executor.submit(
                        self._transcribe_on_gpu,
                        gpu_id, gpu_batch
                    )
                    futures.append(future)
            
            # Collect results
            for future in futures:
                try:
                    gpu_results = future.result(timeout=3600)  # 1 hour timeout
                    results.extend(gpu_results)
                except Exception as e:
                    logger.exception(f"GPU transcription failed: {str(e)}")
        
        return results
    
    def _distribute_batch(self, batch: List[Dict]) -> Dict[int, List[Dict]]:
        """Distribute batch items across GPUs for load balancing"""
        gpu_batches = {gpu_id: [] for gpu_id in self.gpu_ids}
        
        # Sort by duration for better load balancing
        sorted_batch = sorted(batch, key=lambda x: x.get('duration', 0), reverse=True)
        
        # Distribute using round-robin with duration awareness
        gpu_durations = {gpu_id: 0 for gpu_id in self.gpu_ids}
        
        for item in sorted_batch:
            # Find GPU with minimum total duration
            min_gpu = min(gpu_durations.keys(), key=lambda k: gpu_durations[k])
            gpu_batches[min_gpu].append(item)
            gpu_durations[min_gpu] += item.get('duration', 0)
        
        # Log distribution
        for gpu_id, items in gpu_batches.items():
            total_duration = sum(item.get('duration', 0) for item in items)
            logger.info(f"GPU {gpu_id}: {len(items)} files, {total_duration:.1f}s total duration")
        
        return gpu_batches
    
    def _transcribe_on_gpu(self, gpu_id: int, batch: List[Dict]) -> List[Dict]:
        """
        Transcribe batch on specific GPU
        
        Args:
            gpu_id: GPU ID to use
            batch: List of items to transcribe
            
        Returns:
            List of transcription results
        """
        logger.info(f"Starting transcription on GPU {gpu_id} for {len(batch)} files")
        
        # Set CUDA device
        torch.cuda.set_device(gpu_id)
        model = self.models[gpu_id]
        
        results = []
        
        for item in batch:
            try:
                start_time = time.time()
                
                # Handle long audio by chunking if necessary
                if item['duration'] > self.max_duration:
                    result = self._transcribe_long_audio(model, item)
                else:
                    result = self._transcribe_single(model, item)
                
                # Add timing information (guard against zero/None duration)
                result['transcription_time'] = time.time() - start_time
                duration_seconds = item.get('duration') or 0
                result['rtf'] = (
                    result['transcription_time'] / duration_seconds
                    if duration_seconds > 0 else None
                )
                
                results.append(result)
                
                rtf_value = result.get('rtf')
                rtf_str = f"{rtf_value:.2f}" if isinstance(rtf_value, (int, float)) and rtf_value is not None else "n/a"
                logger.info(f"Transcribed '{item['title']}' on GPU {gpu_id} (RTF: {rtf_str})")
                
            except Exception as e:
                logger.exception(f"Failed to transcribe {item['title']}: {str(e)}")
                results.append({
                    'episode_id': item['episode_id'],
                    'success': False,
                    'error': str(e)
                })
        
        return results
    
    def _transcribe_single(self, model, item: Dict) -> Dict:
        """Transcribe a single audio file"""
        
        try:
            # Transcribe with timestamps
            result = model.transcribe(
                [item['processed_path']],
                batch_size=1,
                return_hypotheses=True,
                num_workers=0  # Avoid multiprocessing issues
            )
            
            # Normalize output to text + metadata
            transcript_text, metadata = self._normalize_transcription_output(result)
            
            # Get word-level timestamps if available
            timestamps = self._extract_timestamps(model, item['processed_path'])
            
            # Format transcript
            transcript = {
                'text': transcript_text,
                'segments': self._format_segments(transcript_text, timestamps),
                'language': 'en'
            }
            
            # Calculate metrics
            word_count = len(transcript_text.split())
            confidence = metadata.get('confidence', 0.95)  # Default high confidence
            
            return {
                'episode_id': item['episode_id'],
                'success': True,
                'transcript': transcript,
                'metadata': {
                    'duration': item['duration'],
                    'word_count': word_count,
                    'confidence': confidence,
                    'model': self.model_name,
                    'sample_rate': item['sample_rate']
                }
            }
            
        except Exception as e:
            logger.exception(f"Transcription failed: {str(e)}")
            raise

    def _normalize_transcription_output(self, result: Any) -> Tuple[str, Dict[str, Any]]:
        """
        Normalize various possible NeMo transcription outputs into (text, metadata).

        Handles:
        - list[str]
        - list[Hypothesis] (objects with .text/.words/.confidence)
        - list[tuple]
        - plain str
        - dict with 'text'
        - anything else -> str(result)
        """
        text: str = ""
        metadata: Dict[str, Any] = {}

        try:
            # Common case: list with a single item
            if isinstance(result, list) and len(result) > 0:
                first = result[0]

                # Tuple format: (text, metadata)
                if isinstance(first, tuple) and len(first) > 0:
                    text = first[0] if isinstance(first[0], str) else str(first[0])
                    if len(first) > 1 and isinstance(first[1], dict):
                        metadata = first[1]
                    return text, metadata

                # Hypothesis-like object: has .text attribute
                if hasattr(first, 'text'):
                    text = getattr(first, 'text', '') or ''
                    # Optionally capture confidence/words if present
                    conf = getattr(first, 'confidence', None)
                    if conf is not None:
                        metadata['confidence'] = conf
                    words = getattr(first, 'words', None)
                    if words is not None:
                        metadata['words'] = words
                    return text, metadata

                # Dict with 'text' key
                if isinstance(first, dict) and 'text' in first:
                    text = first.get('text') or ''
                    # Merge any remaining fields as metadata
                    md = {k: v for k, v in first.items() if k != 'text'}
                    if md:
                        metadata.update(md)
                    return text, metadata

                # Plain string
                if isinstance(first, str):
                    return first, metadata

                # Fallback
                return str(first), metadata

            # Not a list, handle direct values
            if isinstance(result, str):
                return result, metadata
            if isinstance(result, dict) and 'text' in result:
                text = result.get('text') or ''
                md = {k: v for k, v in result.items() if k != 'text'}
                if md:
                    metadata.update(md)
                return text, metadata
            # Non-list Hypothesis-like object
            if hasattr(result, 'text'):
                text = getattr(result, 'text', '') or ''
                conf = getattr(result, 'confidence', None)
                if conf is not None:
                    metadata['confidence'] = conf
                words = getattr(result, 'words', None)
                if words is not None:
                    metadata['words'] = words
                return text, metadata

            # Fallback to string conversion
            return str(result), metadata
        except Exception:
            # Last-resort fallback
            return str(result), metadata
    
    def _transcribe_long_audio(self, model, item: Dict) -> Dict:
        """
        Transcribe long audio by chunking
        
        Args:
            model: ASR model
            item: Audio item dictionary
            
        Returns:
            Transcription result
        """
        logger.info(f"Transcribing long audio ({item['duration']:.1f}s) in chunks")
        
        # Load audio
        audio, sr = librosa.load(item['processed_path'], sr=None)
        
        # Calculate chunk size based on configured values
        chunk_samples = int(self.chunk_duration * sr)
        overlap_samples = int(self.overlap_duration * sr)
        
        # Create chunks
        chunks = []
        start = 0
        chunk_paths = []
        
        while start < len(audio):
            end = min(start + chunk_samples, len(audio))
            chunk = audio[start:end]
            
            # Save chunk temporarily
            chunk_path = Path(f"/tmp/chunk_{item['episode_id']}_{len(chunks)}.wav")
            sf.write(chunk_path, chunk, sr)
            chunk_paths.append(str(chunk_path))
            
            chunks.append({
                'start_time': start / sr,
                'end_time': end / sr,
                'path': str(chunk_path)
            })
            
            start = end - overlap_samples if end < len(audio) else end
        
        # Transcribe chunks
        all_segments = []
        full_text = []
        
        for i, chunk in enumerate(chunks):
            logger.debug(f"Transcribing chunk {i+1}/{len(chunks)}")
            
            result = model.transcribe(
                [chunk['path']],
                batch_size=1,
                return_hypotheses=False,
                num_workers=0
            )
            
            if result:
                chunk_text, _ = self._normalize_transcription_output(result)
                text = chunk_text if isinstance(chunk_text, str) else str(chunk_text)
                full_text.append(text)
                
                # Adjust timestamps for chunk offset
                segments = self._format_segments(text, None)
                for segment in segments:
                    # Normalize and offset start time
                    if segment.get('start') is not None:
                        try:
                            segment['start'] = float(segment['start']) + float(chunk['start_time'])
                        except Exception:
                            segment['start'] = float(chunk['start_time'])
                    else:
                        segment['start'] = float(chunk['start_time'])

                    # Normalize and offset end time
                    if segment.get('end') is not None:
                        try:
                            segment['end'] = float(segment['end']) + float(chunk['start_time'])
                        except Exception:
                            segment['end'] = float(chunk['end_time'])
                    else:
                        segment['end'] = float(chunk['end_time'])
                
                all_segments.extend(segments)
        
        # Clean up chunk files
        for path in chunk_paths:
            try:
                Path(path).unlink()
            except:
                pass
        
        # Combine results
        transcript = {
            'text': ' '.join(full_text),
            'segments': all_segments,
            'language': 'en'
        }
        
        return {
            'episode_id': item['episode_id'],
            'success': True,
            'transcript': transcript,
            'metadata': {
                'duration': item['duration'],
                'word_count': len(' '.join(full_text).split()),
                'confidence': 0.95,
                'model': self.model_name,
                'sample_rate': item['sample_rate'],
                'chunks_processed': len(chunks)
            }
        }
    
    def _extract_timestamps(self, model, audio_path: str) -> Optional[List[Dict]]:
        """
        Extract word-level timestamps from model if available
        
        Args:
            model: ASR model
            audio_path: Path to audio file
            
        Returns:
            List of timestamp dictionaries or None
        """
        try:
            # Try to get timestamps using model's built-in functionality
            # This depends on the specific model configuration
            
            # For Parakeet TDT, timestamps are included in the output
            # when using the appropriate decoding configuration
            
            if hasattr(model, 'get_transcription_with_alignment'):
                result = model.get_transcription_with_alignment([audio_path])
                if result and 'words' in result[0]:
                    return result[0]['words']
            
            return None
            
        except Exception as e:
            logger.debug(f"Could not extract timestamps: {str(e)}")
            return None
    
    def _format_segments(self, text: str, timestamps: Optional[List[Dict]] = None) -> List[Dict]:
        """
        Format transcript into segments with timestamps
        
        Args:
            text: Full transcript text
            timestamps: Optional word-level timestamps
            
        Returns:
            List of segment dictionaries
        """
        # Ensure text is a string
        if not isinstance(text, str):
            text = getattr(text, 'text', str(text))

        segments = []
        
        if timestamps:
            # Use provided timestamps to create segments
            current_segment = {
                'start': 0,
                'end': 0,
                'text': '',
                'words': []
            }
            
            for word_info in timestamps:
                current_segment['words'].append(word_info)
                current_segment['text'] += word_info['word'] + ' '
                current_segment['end'] = word_info.get('end_time', 0)
                
                # Create new segment after sentence end
                if word_info['word'].endswith(('.', '!', '?')):
                    current_segment['text'] = current_segment['text'].strip()
                    segments.append(current_segment)
                    current_segment = {
                        'start': current_segment['end'],
                        'end': 0,
                        'text': '',
                        'words': []
                    }
            
            # Add remaining segment
            if current_segment['text']:
                current_segment['text'] = current_segment['text'].strip()
                segments.append(current_segment)
        
        else:
            # Create segments based on sentences without timestamps
            sentences = text.replace('!', '.').replace('?', '.').split('.')
            
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence:
                    segments.append({
                        'start': None,
                        'end': None,
                        'text': sentence,
                        'words': None
                    })
        
        return segments
    
    def cleanup(self):
        """Clean up resources and temporary files"""
        logger.info("Cleaning up transcriber resources")
        
        # Clear CUDA cache
        torch.cuda.empty_cache()
        
        # Clean up temporary files
        temp_dir = Path('/tmp/processed_audio')
        if temp_dir.exists():
            for file in temp_dir.glob('*'):
                try:
                    file.unlink()
                except:
                    pass
        
        # Clean up chunk files
        for file in Path('/tmp').glob('chunk_*.wav'):
            try:
                file.unlink()
            except:
                pass

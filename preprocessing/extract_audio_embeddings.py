"""
Extract audio embeddings using wav2vec 2.0.
Processes TTS MP3 files → 128D embeddings for contrastive learning.
"""

import json
import numpy as np
import torch
import librosa
from pathlib import Path
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from tqdm import tqdm
import argparse


def load_audio(audio_path: str, target_sr: int = 16000) -> np.ndarray:
    """
    Load audio file and resample to target sample rate.
    
    Args:
        audio_path: Path to audio file
        target_sr: Target sample rate (wav2vec expects 16kHz)
        
    Returns:
        Audio waveform as numpy array
    """
    waveform, sr = librosa.load(audio_path, sr=target_sr)
    return waveform


class AudioEmbeddingExtractor:
    """Extract embeddings using wav2vec 2.0."""
    
    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base",
        device: str = None
    ):
        """
        Initialize wav2vec 2.0 model.
        
        Args:
            model_name: HuggingFace model name
            device: 'cpu' or 'cuda'
        """
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"Loading wav2vec 2.0 model: {model_name}")
        print(f"Device: {self.device}")
        
        # Load model and processor
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print("✓ Model loaded")
    
    def extract_embedding(self, waveform: np.ndarray) -> np.ndarray:
        """
        Extract embedding from audio waveform.
        
        Args:
            waveform: Audio waveform (16kHz)
            
        Returns:
            Embedding vector (768D from wav2vec, will be projected to 128D later)
        """
        # Process audio
        inputs = self.processor(
            waveform,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Extract features
        with torch.no_grad():
            outputs = self.model(**inputs)
            
            # Get last hidden state
            hidden_states = outputs.last_hidden_state  # (batch, time, 768)
            
            # Global average pooling over time dimension
            embedding = hidden_states.mean(dim=1)  # (batch, 768)
        
        # Convert to numpy
        embedding = embedding.cpu().numpy()[0]
        
        return embedding


def main():
    """Extract audio embeddings for all TTS files."""
    parser = argparse.ArgumentParser(description='Extract wav2vec 2.0 audio embeddings')
    parser.add_argument('--audio-dir', default='data/audio',
                       help='Directory with audio files')
    parser.add_argument('--metadata', default='data/audio/metadata.json',
                       help='Metadata JSON file')
    parser.add_argument('--output', default='data/features/audio_embeddings.npy',
                       help='Output numpy file')
    parser.add_argument('--model', default='facebook/wav2vec2-base',
                       help='wav2vec model name')
    parser.add_argument('--device', default=None,
                       choices=['cpu', 'cuda'],
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Extracting Audio Embeddings (wav2vec 2.0)")
    print("="*60)
    print(f"Audio directory: {args.audio_dir}")
    print(f"Metadata: {args.metadata}")
    print(f"Output: {args.output}")
    print(f"Model: {args.model}")
    print()
    
    # Check if metadata exists
    metadata_path = Path(args.metadata)
    if not metadata_path.exists():
        print(f"Error: {args.metadata} not found!")
        print("Please generate TTS audio first: python data_collection/generate_tts.py")
        return
    
    # Load metadata
    print("Loading metadata...")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    print(f"✓ Found {len(metadata)} audio files")
    
    # Initialize extractor
    extractor = AudioEmbeddingExtractor(
        model_name=args.model,
        device=args.device
    )
    
    # Process all audio files
    embeddings = []
    texts = []
    failed = []
    
    audio_dir = Path(args.audio_dir)
    
    print("\nExtracting embeddings...")
    for item in tqdm(metadata, desc="Processing audio"):
        audio_path = audio_dir / item['filename']
        
        if not audio_path.exists():
            failed.append(item['filename'])
            continue
        
        try:
            # Load audio
            waveform = load_audio(str(audio_path))
            
            # Extract embedding
            embedding = extractor.extract_embedding(waveform)
            
            embeddings.append(embedding)
            texts.append(item['text'])
        
        except Exception as e:
            print(f"\nError processing {item['filename']}: {e}")
            failed.append(item['filename'])
    
    # Convert to numpy array
    embeddings = np.array(embeddings)
    
    # Save embeddings
    np.save(args.output, embeddings)
    print(f"\n✓ Saved embeddings: {embeddings.shape}")
    
    # Save texts for reference
    texts_output = output_path.parent / 'audio_texts.txt'
    with open(texts_output, 'w', encoding='utf-8') as f:
        for text in texts:
            f.write(text + '\n')
    print(f"✓ Saved texts to {texts_output}")
    
    # Save mapping
    mapping = {
        'embeddings_shape': embeddings.shape,
        'texts': texts,
        'failed': failed
    }
    mapping_output = output_path.parent / 'audio_embedding_mapping.json'
    with open(mapping_output, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, indent=2)
    print(f"✓ Saved mapping to {mapping_output}")
    
    # Statistics
    print(f"\nStatistics:")
    print(f"  Successful: {len(embeddings)}")
    print(f"  Failed: {len(failed)}")
    print(f"  Embedding dimension: {embeddings.shape[1]}")
    print(f"  Mean norm: {np.linalg.norm(embeddings, axis=1).mean():.2f}")
    print(f"  Std norm: {np.linalg.norm(embeddings, axis=1).std():.2f}")
    
    if failed:
        print(f"\nFailed files: {failed[:5]}..." if len(failed) > 5 else f"\nFailed files: {failed}")
    
    print("\n✓ Audio embedding extraction complete!")
    print(f"\nNext step: python preprocessing/align_triplets.py")


if __name__ == "__main__":
    main()
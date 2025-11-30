import json
import numpy as np
import pickle
from pathlib import Path
from scipy.interpolate import interp1d
import argparse


def load_finger_data(npy_path: str) -> tuple:
    """
    Load finger trajectory data.
    
    Returns:
        (timestamps, positions) where positions is (N, 2) with [x, y]
    """
    data = np.load(npy_path)
    timestamps = data[:, 0]
    positions = data[:, 1:3]
    return timestamps, positions


def load_imu_data(npy_path: str) -> tuple:
    """
    Load IMU accelerometer data.
    
    Returns:
        (timestamps, accelerations) where accelerations is (N, 3) with [ax, ay, az]
    """
    data = np.load(npy_path)
    timestamps = data[:, 0]
    accelerations = data[:, 1:4]
    return timestamps, accelerations


def align_sensor_modalities(
    finger_t: np.ndarray,
    finger_pos: np.ndarray,
    imu_t: np.ndarray,
    imu_acc: np.ndarray
) -> tuple:
    """
    Align finger and IMU data to common timebase.
    
    Returns:
        (common_timestamps, fused_5d_signal) where signal is [x, y, ax, ay, az]
    """
    # Find common time range
    t_start = max(finger_t[0], imu_t[0])
    t_end = min(finger_t[-1], imu_t[-1])
    
    # Create common timebase (use finger FPS as reference, typically 30Hz)
    duration = t_end - t_start
    num_samples = int(duration * 30)  # 30 FPS
    common_t = np.linspace(t_start, t_end, num_samples)
    
    # Interpolate finger data
    finger_interp = interp1d(finger_t, finger_pos, axis=0, kind='linear', 
                            bounds_error=False, fill_value='extrapolate')
    finger_aligned = finger_interp(common_t)
    
    # Interpolate IMU data
    imu_interp = interp1d(imu_t, imu_acc, axis=0, kind='linear',
                         bounds_error=False, fill_value='extrapolate')
    imu_aligned = imu_interp(common_t)
    
    # Fuse into 5D signal
    fused_signal = np.column_stack([finger_aligned, imu_aligned])  # (N, 5)
    
    return common_t, fused_signal


def create_training_pairs(
    sensor_t: np.ndarray,
    sensor_signal: np.ndarray,
    audio_embeddings: np.ndarray,
    texts: list,
    window_size: int = 30,  # 1 second at 30 FPS
    stride: int = 15  # 0.5 second overlap
) -> list:
    """
    Create (sensor_window, audio_embedding) training pairs.
    
    Args:
        sensor_t: Sensor timestamps (N,)
        sensor_signal: Sensor data (N, 5)
        audio_embeddings: Audio embeddings (M, 768)
        texts: List of texts corresponding to embeddings
        window_size: Sensor window size in frames
        stride: Stride between windows
        
    Returns:
        List of (sensor_window, audio_embedding, text_idx) tuples
    """
    pairs = []
    
    num_embeddings = len(audio_embeddings)
    num_sensor_samples = len(sensor_signal)
    
    print(f"\nCreating training pairs...")
    print(f"  Sensor samples: {num_sensor_samples}")
    print(f"  Audio embeddings: {num_embeddings}")
    print(f"  Window size: {window_size}")
    print(f"  Stride: {stride}")
    
    # Strategy: Evenly distribute sensor data across audio embeddings
    # Each audio embedding gets approximately equal sensor samples
    samples_per_embedding = max(1, num_sensor_samples // num_embeddings)
    
    print(f"  Samples per embedding: {samples_per_embedding}")
    
    # Only use embeddings that we have enough sensor data for
    usable_embeddings = min(num_embeddings, num_sensor_samples // window_size)
    
    if usable_embeddings == 0:
        print(f"\n⚠️  Warning: Sensor recording too short for window size {window_size}")
        print(f"     Need at least {window_size} samples, have {num_sensor_samples}")
        return pairs
    
    for idx in range(usable_embeddings):
        # Calculate sensor segment for this audio embedding
        seg_start = idx * samples_per_embedding
        seg_end = min((idx + 1) * samples_per_embedding, num_sensor_samples)
        
        # Ensure segment is large enough for window
        if seg_end - seg_start < window_size:
            continue
        
        segment = sensor_signal[seg_start:seg_end]
        audio_emb = audio_embeddings[idx]
        
        # Create sliding windows over this segment
        num_windows = 0
        for i in range(0, len(segment) - window_size + 1, stride):
            window = segment[i:i + window_size]
            
            pairs.append({
                'sensor': window,  # (window_size, 5)
                'audio': audio_emb,  # (768,)
                'text_idx': idx,
                'text': texts[idx] if idx < len(texts) else ''
            })
            num_windows += 1
        
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{usable_embeddings} embeddings, {len(pairs)} pairs so far...")
    
    print(f"  ✓ Created {len(pairs)} training pairs from {usable_embeddings} embeddings")
    
    return pairs


def main():
    """Align all modalities and create training pairs."""
    parser = argparse.ArgumentParser(description='Align sensor and audio data')
    parser.add_argument('--finger', default='data/features/finger_xy.npy',
                       help='Finger trajectory file')
    parser.add_argument('--imu', default='data/features/imu_acc.npy',
                       help='IMU accelerometer file')
    parser.add_argument('--audio', default='data/features/audio_embeddings.npy',
                       help='Audio embeddings file')
    parser.add_argument('--texts', default='data/features/audio_texts.txt',
                       help='Audio texts file')
    parser.add_argument('--output', default='data/features/aligned_pairs.pkl',
                       help='Output pickle file')
    parser.add_argument('--window-size', type=int, default=30,
                       help='Sensor window size (frames)')
    parser.add_argument('--stride', type=int, default=15,
                       help='Window stride (frames)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Aligning Sensor and Audio Data")
    print("="*60)
    print(f"Finger data: {args.finger}")
    print(f"IMU data: {args.imu}")
    print(f"Audio embeddings: {args.audio}")
    print(f"Window size: {args.window_size} frames")
    print(f"Stride: {args.stride} frames")
    print()
    
    # Check if all inputs exist
    required_files = [args.finger, args.imu, args.audio, args.texts]
    for file_path in required_files:
        if not Path(file_path).exists():
            print(f"Error: {file_path} not found!")
            print("Please run all preprocessing steps first.")
            return
    
    # Load data
    print("Loading data...")
    finger_t, finger_pos = load_finger_data(args.finger)
    print(f"✓ Loaded finger data: {finger_pos.shape}")
    
    imu_t, imu_acc = load_imu_data(args.imu)
    print(f"✓ Loaded IMU data: {imu_acc.shape}")
    
    audio_embeddings = np.load(args.audio)
    print(f"✓ Loaded audio embeddings: {audio_embeddings.shape}")
    
    with open(args.texts, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f]
    print(f"✓ Loaded {len(texts)} texts")
    
    # Align sensor modalities
    print("\nAligning finger and IMU data...")
    sensor_t, fused_signal = align_sensor_modalities(
        finger_t, finger_pos,
        imu_t, imu_acc
    )
    print(f"✓ Fused 5D sensor signal: {fused_signal.shape}")
    print(f"  Duration: {sensor_t[-1] - sensor_t[0]:.2f} seconds")
    print(f"  Sample rate: {len(sensor_t) / (sensor_t[-1] - sensor_t[0]):.1f} Hz")
    
    # Create training pairs
    print("\nCreating training pairs...")
    pairs = create_training_pairs(
        sensor_t,
        fused_signal,
        audio_embeddings,
        texts,
        window_size=args.window_size,
        stride=args.stride
    )
    print(f"✓ Created {len(pairs)} training pairs")
    
    # Save to pickle
    print("\nSaving aligned pairs...")
    with open(args.output, 'wb') as f:
        pickle.dump(pairs, f)
    print(f"✓ Saved to {args.output}")
    
    # Check if we got any pairs
    if len(pairs) == 0:
        print("\n❌ ERROR: No training pairs created!")
        print("\nPossible causes:")
        print("  1. Sensor recording too short")
        print("  2. Not enough audio embeddings")
        print("  3. Mismatch between recording duration and audio count")
        print("\nTroubleshooting:")
        print(f"  - Sensor duration: {sensor_t[-1] - sensor_t[0]:.1f} seconds")
        print(f"  - Audio embeddings: {len(audio_embeddings)}")
        print(f"  - Samples per embedding: {len(fused_signal) // len(audio_embeddings)}")
        print(f"  - Window size needed: {args.window_size} frames")
        print("\nSolution: Record for longer, or reduce --window-size")
        return
    
    # Statistics
    print(f"\nStatistics:")
    print(f"  Total pairs: {len(pairs)}")
    print(f"  Sensor window shape: {pairs[0]['sensor'].shape}")
    print(f"  Audio embedding shape: {pairs[0]['audio'].shape}")
    print(f"  Unique texts: {len(set(p['text_idx'] for p in pairs))}")
    
    # Sample pairs
    print(f"\nSample pairs:")
    for i in range(min(3, len(pairs))):
        print(f"  Pair {i}:")
        print(f"    Text: '{pairs[i]['text'][:60]}...'")
        print(f"    Sensor range: [{pairs[i]['sensor'].min():.2f}, {pairs[i]['sensor'].max():.2f}]")
    
    print("\n✓ Data alignment complete!")
    print(f"\nNext step: Upload to Google Colab and train!")
    print(f"  1. Open training/train_colab.ipynb in Colab")
    print(f"  2. Upload {args.output}")
    print(f"  3. Run all cells")


if __name__ == "__main__":
    main()
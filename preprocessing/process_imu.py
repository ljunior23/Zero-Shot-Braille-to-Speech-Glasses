"""
Process phone IMU data (accelerometer + gyroscope).
Resamples from ~200Hz to 30Hz to match webcam frame rate.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import resample, butter, filtfilt
import argparse


def load_imu_data(json_path: str) -> dict:
    """Load recorded IMU data."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def extract_imu_signals(data: list) -> tuple:
    """
    Extract accelerometer and gyroscope signals.
    
    Args:
        data: List of IMU samples
        
    Returns:
        (timestamps, accelerometer, gyroscope) arrays
    """
    timestamps = []
    acc_data = []
    gyr_data = []
    
    for sample in data:
        timestamps.append(sample['timestamp'])
        
        acc = sample['acceleration']
        acc_data.append([acc['x'], acc['y'], acc['z']])
        
        rot = sample['rotation']
        gyr_data.append([rot['alpha'], rot['beta'], rot['gamma']])
    
    timestamps = np.array(timestamps)
    accelerometer = np.array(acc_data)
    gyroscope = np.array(gyr_data)
    
    return timestamps, accelerometer, gyroscope


def butter_lowpass_filter(data: np.ndarray, cutoff: float, fs: float, order: int = 4) -> np.ndarray:
    """
    Apply Butterworth low-pass filter.
    
    Args:
        data: Input signal
        cutoff: Cutoff frequency (Hz)
        fs: Sampling frequency (Hz)
        order: Filter order
        
    Returns:
        Filtered signal
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    
    filtered = np.zeros_like(data)
    for i in range(data.shape[1]):
        filtered[:, i] = filtfilt(b, a, data[:, i])
    
    return filtered


def resample_imu(timestamps: np.ndarray, signal: np.ndarray, target_fps: int = 30) -> tuple:
    """
    Resample IMU signal to target frame rate.
    
    Args:
        timestamps: Original timestamps
        signal: Original signal (N, 3)
        target_fps: Target frames per second
        
    Returns:
        (resampled_timestamps, resampled_signal)
    """
    # Calculate original sampling rate
    duration = timestamps[-1] - timestamps[0]
    original_fps = len(timestamps) / duration
    
    # Create uniform time grid
    num_samples = int(duration * target_fps)
    t_uniform = np.linspace(timestamps[0], timestamps[-1], num_samples)
    
    # Resample each axis
    resampled_signal = np.zeros((num_samples, signal.shape[1]))
    for i in range(signal.shape[1]):
        resampled_signal[:, i] = np.interp(t_uniform, timestamps, signal[:, i])
    
    return t_uniform, resampled_signal


def normalize_imu(signal: np.ndarray) -> np.ndarray:
    """
    Normalize IMU signal (z-score normalization).
    
    Args:
        signal: Input signal (N, 3)
        
    Returns:
        Normalized signal
    """
    normalized = np.zeros_like(signal)
    
    for i in range(signal.shape[1]):
        mean = signal[:, i].mean()
        std = signal[:, i].std()
        if std > 0:
            normalized[:, i] = (signal[:, i] - mean) / std
        else:
            normalized[:, i] = signal[:, i] - mean
    
    return normalized


def visualize_imu(timestamps: np.ndarray, acc: np.ndarray, gyr: np.ndarray, 
                  title: str = "IMU Data", save_path: str = None):
    """Visualize IMU data."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot accelerometer
    axes[0].plot(timestamps, acc[:, 0], label='Accel X', linewidth=1.5)
    axes[0].plot(timestamps, acc[:, 1], label='Accel Y', linewidth=1.5)
    axes[0].plot(timestamps, acc[:, 2], label='Accel Z', linewidth=1.5)
    axes[0].set_xlabel('Time (seconds)')
    axes[0].set_ylabel('Acceleration (normalized)')
    axes[0].set_title(f'{title} - Accelerometer')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot gyroscope
    axes[1].plot(timestamps, gyr[:, 0], label='Gyro X', linewidth=1.5)
    axes[1].plot(timestamps, gyr[:, 1], label='Gyro Y', linewidth=1.5)
    axes[1].plot(timestamps, gyr[:, 2], label='Gyro Z', linewidth=1.5)
    axes[1].set_xlabel('Time (seconds)')
    axes[1].set_ylabel('Angular velocity (normalized)')
    axes[1].set_title(f'{title} - Gyroscope')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    """Process phone IMU data."""
    parser = argparse.ArgumentParser(description='Process phone IMU data')
    parser.add_argument('--input', default='data/recordings/imu_data.json',
                       help='Input JSON file')
    parser.add_argument('--output', default='data/features/imu_acc.npy',
                       help='Output numpy file')
    parser.add_argument('--target-fps', type=int, default=30,
                       help='Target frame rate')
    parser.add_argument('--filter-cutoff', type=float, default=10.0,
                       help='Low-pass filter cutoff frequency (Hz)')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualization')
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Processing Phone IMU Data")
    print("="*60)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Target FPS: {args.target_fps}")
    print(f"Filter cutoff: {args.filter_cutoff} Hz")
    print()
    
    # Check if input exists
    if not Path(args.input).exists():
        print(f"Error: {args.input} not found!")
        print("Please record IMU data first using record_imu.html on phone")
        return
    
    # Load data
    print("Loading IMU data...")
    data = load_imu_data(args.input)
    print(f"✓ Loaded {len(data)} samples")
    
    # Extract signals
    print("\nExtracting IMU signals...")
    timestamps, accelerometer, gyroscope = extract_imu_signals(data)
    
    duration = timestamps[-1] - timestamps[0]
    original_fps = len(timestamps) / duration
    
    print(f"✓ Extracted signals:")
    print(f"  Duration: {duration:.2f} seconds")
    print(f"  Original sampling rate: {original_fps:.1f} Hz")
    print(f"  Accelerometer shape: {accelerometer.shape}")
    print(f"  Gyroscope shape: {gyroscope.shape}")
    
    # Apply low-pass filter
    print(f"\nApplying low-pass filter (cutoff={args.filter_cutoff} Hz)...")
    acc_filtered = butter_lowpass_filter(accelerometer, args.filter_cutoff, original_fps)
    gyr_filtered = butter_lowpass_filter(gyroscope, args.filter_cutoff, original_fps)
    print(f"✓ Filtered signals")
    
    # Resample to target FPS
    print(f"\nResampling to {args.target_fps} FPS...")
    t_acc, acc_resampled = resample_imu(timestamps, acc_filtered, target_fps=args.target_fps)
    t_gyr, gyr_resampled = resample_imu(timestamps, gyr_filtered, target_fps=args.target_fps)
    
    print(f"✓ Resampled:")
    print(f"  Accelerometer: {acc_resampled.shape}")
    print(f"  Gyroscope: {gyr_resampled.shape}")
    print(f"  Final FPS: {len(t_acc) / (t_acc[-1] - t_acc[0]):.1f}")
    
    # Normalize
    print("\nNormalizing signals...")
    acc_normalized = normalize_imu(acc_resampled)
    gyr_normalized = normalize_imu(gyr_resampled)
    print(f"✓ Applied z-score normalization")
    
    # Save accelerometer (primary signal for finger taps)
    np.save(args.output, np.column_stack([t_acc, acc_normalized]))
    print(f"\n✓ Saved accelerometer to {args.output}")
    
    # Save gyroscope separately
    gyr_output = output_path.parent / 'imu_gyr.npy'
    np.save(gyr_output, np.column_stack([t_gyr, gyr_normalized]))
    print(f"✓ Saved gyroscope to {gyr_output}")
    
    # Statistics
    print(f"\nStatistics:")
    print(f"  Total samples: {len(acc_normalized)}")
    print(f"  Duration: {t_acc[-1] - t_acc[0]:.2f} seconds")
    print(f"  Accel X range: [{acc_normalized[:, 0].min():.2f}, {acc_normalized[:, 0].max():.2f}]")
    print(f"  Accel Y range: [{acc_normalized[:, 1].min():.2f}, {acc_normalized[:, 1].max():.2f}]")
    print(f"  Accel Z range: [{acc_normalized[:, 2].min():.2f}, {acc_normalized[:, 2].max():.2f}]")
    
    # Visualize
    if args.visualize:
        print("\nCreating visualization...")
        viz_path = output_path.parent / 'imu_visualization.png'
        visualize_imu(t_acc, acc_normalized, gyr_normalized, save_path=str(viz_path))
    
    print("\n✓ IMU processing complete!")
    print(f"\nNext step: python preprocessing/extract_audio_embeddings.py")


if __name__ == "__main__":
    main()
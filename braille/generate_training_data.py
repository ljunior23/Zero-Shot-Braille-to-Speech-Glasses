"""
Generate synthetic Braille training dataset.
Creates images of Braille characters with variations.
"""

import numpy as np
import cv2
from pathlib import Path
import json
from tqdm import tqdm


# Braille dot positions in standard 2x3 cell
DOT_POSITIONS = {
    1: (0.25, 0.2),
    2: (0.25, 0.5),
    3: (0.25, 0.8),
    4: (0.75, 0.2),
    5: (0.75, 0.5),
    6: (0.75, 0.8),
}

# Braille patterns for letters a-z
BRAILLE_PATTERNS = {
    'a': [1],
    'b': [1, 2],
    'c': [1, 4],
    'd': [1, 4, 5],
    'e': [1, 5],
    'f': [1, 2, 4],
    'g': [1, 2, 4, 5],
    'h': [1, 2, 5],
    'i': [2, 4],
    'j': [2, 4, 5],
    'k': [1, 3],
    'l': [1, 2, 3],
    'm': [1, 3, 4],
    'n': [1, 3, 4, 5],
    'o': [1, 3, 5],
    'p': [1, 2, 3, 4],
    'q': [1, 2, 3, 4, 5],
    'r': [1, 2, 3, 5],
    's': [2, 3, 4],
    't': [2, 3, 4, 5],
    'u': [1, 3, 6],
    'v': [1, 2, 3, 6],
    'w': [2, 4, 5, 6],
    'x': [1, 3, 4, 6],
    'y': [1, 3, 4, 5, 6],
    'z': [1, 3, 5, 6],
}


def generate_braille_image(
    pattern: list,
    cell_width: int = 40,
    cell_height: int = 60,
    dot_radius: int = 6,
    add_noise: bool = True,
    noise_level: float = 0.1
) -> np.ndarray:
    """
    Generate single Braille character image.
    
    Args:
        pattern: List of dot positions (1-6)
        cell_width: Width of cell in pixels
        cell_height: Height of cell in pixels
        dot_radius: Radius of dots
        add_noise: Add random variations
        noise_level: Amount of noise (0-1)
        
    Returns:
        Grayscale image of Braille character
    """
    # Create white background
    img = np.ones((cell_height, cell_width), dtype=np.uint8) * 255
    
    # Add slight noise to background if requested
    if add_noise:
        noise = np.random.randn(cell_height, cell_width) * (noise_level * 30)
        img = np.clip(img.astype(float) + noise, 0, 255).astype(np.uint8)
    
    # Draw dots
    for dot_num in pattern:
        # Get dot position (relative to cell size)
        dx_rel, dy_rel = DOT_POSITIONS[dot_num]
        
        # Add random jitter if requested
        if add_noise:
            dx_jitter = np.random.randn() * noise_level * 3
            dy_jitter = np.random.randn() * noise_level * 3
        else:
            dx_jitter = 0
            dy_jitter = 0
        
        # Calculate absolute position
        x = int(cell_width * dx_rel + dx_jitter)
        y = int(cell_height * dy_rel + dy_jitter)
        
        # Vary dot size slightly if requested
        if add_noise:
            radius = int(dot_radius + np.random.randn() * noise_level * 2)
            radius = max(3, min(radius, dot_radius + 3))
        else:
            radius = dot_radius
        
        # Draw filled circle (black dot)
        cv2.circle(img, (x, y), radius, 0, -1)
    
    return img


def generate_dataset(
    output_dir: str = 'data/braille_dataset',
    samples_per_char: int = 1000,
    img_size: tuple = (60, 40),
    train_split: float = 0.8
):
    """
    Generate complete Braille training dataset.
    
    Args:
        output_dir: Where to save dataset
        samples_per_char: Number of samples per character
        img_size: Image size (height, width)
        train_split: Fraction for training (rest is validation)
    """
    output_path = Path(output_dir)
    
    # Create directories
    train_dir = output_path / 'train'
    val_dir = output_path / 'val'
    
    for char in BRAILLE_PATTERNS.keys():
        (train_dir / char).mkdir(parents=True, exist_ok=True)
        (val_dir / char).mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Generating Braille Training Dataset")
    print("="*60)
    print(f"Output directory: {output_dir}")
    print(f"Characters: {len(BRAILLE_PATTERNS)}")
    print(f"Samples per character: {samples_per_char}")
    print(f"Train/Val split: {train_split:.0%}/{1-train_split:.0%}")
    print(f"Image size: {img_size}")
    print()
    
    metadata = {
        'characters': list(BRAILLE_PATTERNS.keys()),
        'samples_per_char': samples_per_char,
        'img_size': img_size,
        'train_split': train_split,
        'total_samples': len(BRAILLE_PATTERNS) * samples_per_char
    }
    
    # Generate samples
    for char, pattern in tqdm(BRAILLE_PATTERNS.items(), desc="Generating characters"):
        for i in range(samples_per_char):
            # Generate image with variations
            img = generate_braille_image(
                pattern,
                cell_width=img_size[1],
                cell_height=img_size[0],
                dot_radius=6,
                add_noise=True,
                noise_level=0.1 + np.random.rand() * 0.1  # 10-20% noise
            )
            
            # Randomly rotate slightly
            if np.random.rand() > 0.5:
                angle = np.random.randn() * 5  # ±5 degrees
                M = cv2.getRotationMatrix2D((img_size[1]/2, img_size[0]/2), angle, 1.0)
                img = cv2.warpAffine(img, M, (img_size[1], img_size[0]), 
                                    borderMode=cv2.BORDER_CONSTANT, 
                                    borderValue=255)
            
            # Decide train or val
            if i < int(samples_per_char * train_split):
                save_dir = train_dir / char
            else:
                save_dir = val_dir / char
            
            # Save image
            filename = save_dir / f"{char}_{i:04d}.png"
            cv2.imwrite(str(filename), img)
    
    # Save metadata
    with open(output_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Dataset generated successfully!")
    print(f"  Training samples: {int(len(BRAILLE_PATTERNS) * samples_per_char * train_split)}")
    print(f"  Validation samples: {int(len(BRAILLE_PATTERNS) * samples_per_char * (1 - train_split))}")
    print(f"  Total: {len(BRAILLE_PATTERNS) * samples_per_char}")
    print(f"\nSaved to: {output_dir}")


def visualize_samples(output_dir: str = 'data/braille_dataset', num_chars: int = 5):
    """Visualize sample images from dataset."""
    import matplotlib.pyplot as plt
    
    train_dir = Path(output_dir) / 'train'
    
    chars = list(BRAILLE_PATTERNS.keys())[:num_chars]
    
    fig, axes = plt.subplots(1, num_chars, figsize=(15, 3))
    
    for i, char in enumerate(chars):
        # Load first image for this character
        img_path = list((train_dir / char).glob('*.png'))[0]
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f"'{char}' - {BRAILLE_PATTERNS[char]}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'sample_visualization.png', dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved to {output_dir}/sample_visualization.png")
    plt.show()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate Braille training dataset')
    parser.add_argument('--output', default='data/braille_dataset',
                       help='Output directory')
    parser.add_argument('--samples', type=int, default=1000,
                       help='Samples per character')
    parser.add_argument('--width', type=int, default=40,
                       help='Image width')
    parser.add_argument('--height', type=int, default=60,
                       help='Image height')
    parser.add_argument('--train-split', type=float, default=0.8,
                       help='Training split fraction')
    parser.add_argument('--visualize', action='store_true',
                       help='Show sample visualizations')
    
    args = parser.parse_args()
    
    # Generate dataset
    generate_dataset(
        output_dir=args.output,
        samples_per_char=args.samples,
        img_size=(args.height, args.width),
        train_split=args.train_split
    )
    
    # Visualize if requested
    if args.visualize:
        visualize_samples(args.output)


if __name__ == "__main__":
    main()
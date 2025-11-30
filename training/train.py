import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse



import sys
sys.path.append('.')
from model import ContrastiveModel


class SensorAudioDataset(Dataset):
    """Dataset for (sensor, audio) pairs."""
    
    def __init__(self, pairs_path: str):
        """
        Load aligned pairs from pickle file.
        
        Args:
            pairs_path: Path to aligned_pairs.pkl
        """
        with open(pairs_path, 'rb') as f:
            self.pairs = pickle.load(f)
        
        print(f"Loaded {len(self.pairs)} training pairs")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        
        sensor = torch.tensor(pair['sensor'], dtype=torch.float32)  # (time, 5)
        audio = torch.tensor(pair['audio'], dtype=torch.float32)  # (768,)
        
        return sensor, audio


def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for sensor, audio in pbar:
        sensor = sensor.to(device)
        audio = audio.to(device)
        
        # Forward pass
        sensor_emb, audio_emb = model(sensor, audio)
        
        # Compute loss
        loss = model.contrastive_loss(sensor_emb, audio_emb)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Calculate accuracy (nearest neighbor in batch)
        with torch.no_grad():
            similarity = torch.matmul(sensor_emb, audio_emb.T)
            predictions = similarity.argmax(dim=1)
            targets = torch.arange(len(sensor), device=device)
            correct += (predictions == targets).sum().item()
            total += len(sensor)
        
        total_loss += loss.item()
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100 * correct / total:.1f}%'
        })
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def validate(model, dataloader, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for sensor, audio in tqdm(dataloader, desc='Validating'):
            sensor = sensor.to(device)
            audio = audio.to(device)
            
            # Forward pass
            sensor_emb, audio_emb = model(sensor, audio)
            
            # Compute loss
            loss = model.contrastive_loss(sensor_emb, audio_emb)
            total_loss += loss.item()
            
            # Calculate accuracy
            similarity = torch.matmul(sensor_emb, audio_emb.T)
            predictions = similarity.argmax(dim=1)
            targets = torch.arange(len(sensor), device=device)
            correct += (predictions == targets).sum().item()
            total += len(sensor)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_path='training_curves.png'):
    """Plot training curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    ax1.plot(train_losses, label='Train Loss', linewidth=2)
    ax1.plot(val_losses, label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Contrastive Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy
    ax2.plot([a * 100 for a in train_accs], label='Train Accuracy', linewidth=2)
    ax2.plot([a * 100 for a in val_accs], label='Val Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Contrastive Accuracy (Batch-Level)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved training curves to {save_path}")
    plt.close()


def main():
    """Main training loop."""
    parser = argparse.ArgumentParser(description='Train contrastive model')
    parser.add_argument('--data', default='data/features/aligned_pairs.pkl',
                       help='Path to aligned pairs')
    parser.add_argument('--batch-size', type=int, default=256,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--output-dir', default='models',
                       help='Output directory')
    parser.add_argument('--device', default=None,
                       help='Device (cpu/cuda/mps)')
    
    args = parser.parse_args()
    
    # Device
    if args.device:
        device = args.device
    else:
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    
    print("="*60)
    print("Training Zero-Shot Finger Reading Model")
    print("="*60)
    print(f"Data: {args.data}")
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print()
    
    # Check data
    if not Path(args.data).exists():
        print(f"Error: {args.data} not found!")
        print("Please run preprocessing first.")
        return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print("Loading dataset...")
    full_dataset = SensorAudioDataset(args.data)
    
    # Split into train/val (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    print(f"✓ Train samples: {len(train_dataset)}")
    print(f"✓ Val samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if device == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device == 'cuda' else False
    )
    
    # Initialize model
    print("\nInitializing model...")
    model = ContrastiveModel(
        sensor_input_dim=5,
        audio_input_dim=768,
        embedding_dim=128,
        temperature=0.07
    )
    model = model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Model parameters: {num_params:,}")
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )
    
    # Training loop
    print("\nStarting training...")
    print("="*60)
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 40)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Update learning rate
        scheduler.step()
        
        # Print metrics
        print(f"\nResults:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
            }, output_dir / 'best_model.pt')
            print("  ✓ Saved best model")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, output_dir / f'checkpoint_epoch_{epoch+1}.pt')
    
    # Plot training curves
    print("\n" + "="*60)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final validation accuracy: {val_accs[-1]*100:.2f}%")
    
    plot_training_curves(
        train_losses, val_losses,
        train_accs, val_accs,
        save_path=str(output_dir / 'training_curves.png')
    )
    
    print(f"\n✓ Model saved to {output_dir}/best_model.pt")
    print(f"✓ Ready for inference!")


if __name__ == "__main__":
    main()
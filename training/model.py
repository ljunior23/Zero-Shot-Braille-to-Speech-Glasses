import torch
import torch.nn as nn
import torch.nn.functional as F


class SensorEncoder(nn.Module):
    """
    Encode 5D sensor signal [x, y, ax, ay, az] to 128D embedding.
    Architecture: 1D CNN + temporal pooling.
    """
    
    def __init__(
        self,
        input_dim: int = 5,
        hidden_dim: int = 64,
        output_dim: int = 128,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 1D Convolutional layers
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
        )
        
        # Global average pooling (adaptive)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, time, 5) sensor signal
            
        Returns:
            (batch, 128) embedding
        """
        # Transpose for Conv1d: (batch, 5, time)
        x = x.transpose(1, 2)
        
        # Convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        # Global pooling
        x = self.global_pool(x)  # (batch, hidden_dim, 1)
        x = x.squeeze(-1)  # (batch, hidden_dim)
        
        # Projection
        x = self.projection(x)  # (batch, 128)
        
        # L2 normalize for contrastive learning
        x = F.normalize(x, p=2, dim=1)
        
        return x


class AudioProjector(nn.Module):
    """
    Project wav2vec embeddings (768D) to 128D for contrastive learning.
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 256,
        output_dim: int = 128,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, 768) wav2vec embedding
            
        Returns:
            (batch, 128) projected embedding
        """
        x = self.projection(x)
        
        # L2 normalize
        x = F.normalize(x, p=2, dim=1)
        
        return x


class ContrastiveModel(nn.Module):
    """
    Full contrastive learning model.
    Aligns sensor embeddings with audio embeddings.
    """
    
    def __init__(
        self,
        sensor_input_dim: int = 5,
        audio_input_dim: int = 768,
        embedding_dim: int = 128,
        temperature: float = 0.07
    ):
        super().__init__()
        
        self.sensor_encoder = SensorEncoder(
            input_dim=sensor_input_dim,
            output_dim=embedding_dim
        )
        
        self.audio_projector = AudioProjector(
            input_dim=audio_input_dim,
            output_dim=embedding_dim
        )
        
        self.temperature = temperature
    
    def forward(self, sensor, audio):
        """
        Args:
            sensor: (batch, time, 5) sensor signal
            audio: (batch, 768) audio embedding
            
        Returns:
            (sensor_emb, audio_emb) both (batch, 128)
        """
        sensor_emb = self.sensor_encoder(sensor)
        audio_emb = self.audio_projector(audio)
        
        return sensor_emb, audio_emb
    
    def contrastive_loss(self, sensor_emb, audio_emb):
        """
        NT-Xent (Normalized Temperature-scaled Cross Entropy) loss.
        Same as SimCLR.
        
        Args:
            sensor_emb: (batch, 128)
            audio_emb: (batch, 128)
            
        Returns:
            Scalar loss
        """
        batch_size = sensor_emb.shape[0]
        
        # Compute similarity matrix
        similarity = torch.matmul(sensor_emb, audio_emb.T) / self.temperature
        
        # Labels: diagonal elements are positive pairs
        labels = torch.arange(batch_size, device=sensor_emb.device)
        
        # Cross-entropy loss in both directions
        loss_s2a = F.cross_entropy(similarity, labels)
        loss_a2s = F.cross_entropy(similarity.T, labels)
        
        # Average
        loss = (loss_s2a + loss_a2s) / 2
        
        return loss


def test_model():
    """Test model architecture."""
    print("Testing Contrastive Model...")
    
    # Create dummy inputs
    batch_size = 16
    time_steps = 30  # 1 second at 30 FPS
    
    sensor = torch.randn(batch_size, time_steps, 5)
    audio = torch.randn(batch_size, 768)
    
    # Initialize model
    model = ContrastiveModel()
    
    # Forward pass
    sensor_emb, audio_emb = model(sensor, audio)
    
    print(f"✓ Sensor embedding: {sensor_emb.shape}")
    print(f"✓ Audio embedding: {audio_emb.shape}")
    
    # Compute loss
    loss = model.contrastive_loss(sensor_emb, audio_emb)
    print(f"✓ Contrastive loss: {loss.item():.4f}")
    
    # Check if embeddings are normalized
    sensor_norms = torch.norm(sensor_emb, p=2, dim=1)
    audio_norms = torch.norm(audio_emb, p=2, dim=1)
    print(f"✓ Sensor embedding norms: {sensor_norms.mean():.4f} (should be ~1.0)")
    print(f"✓ Audio embedding norms: {audio_norms.mean():.4f} (should be ~1.0)")
    
    print("\n✓ Model test passed!")


if __name__ == "__main__":
    test_model()
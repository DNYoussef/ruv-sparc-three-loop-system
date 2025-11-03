#!/usr/bin/env python3
"""
Complete Model Training Example
Demonstrates end-to-end model training workflow with:
- Data loading and preprocessing
- Model architecture setup
- Training with monitoring
- Checkpointing and evaluation
- Hyperparameter optimization
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from resources.scripts.model_trainer import ModelTrainer


# Custom Dataset Example
class CustomImageDataset(Dataset):
    """
    Custom dataset for image classification
    Demonstrates how to create a PyTorch dataset
    """

    def __init__(self, data_path, transform=None):
        """
        Args:
            data_path: Path to dataset
            transform: Optional transform to apply
        """
        self.data_path = Path(data_path)
        self.transform = transform
        self.samples = self._load_samples()

    def _load_samples(self):
        """Load dataset samples"""
        # This is a simplified example
        # In practice, you'd load actual image paths and labels
        samples = []
        # ... load your data here
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """Get a single sample"""
        image_path, label = self.samples[idx]

        # Load image
        # image = load_image(image_path)

        # Apply transforms
        if self.transform:
            # image = self.transform(image)
            pass

        # Return dummy data for demonstration
        image = torch.randn(3, 224, 224)
        return image, label


# Custom Model Architecture
class CustomCNN(nn.Module):
    """
    Custom CNN architecture for image classification
    Demonstrates creating a neural network from scratch
    """

    def __init__(self, num_classes=10, dropout=0.5):
        """
        Args:
            num_classes: Number of output classes
            dropout: Dropout probability
        """
        super(CustomCNN, self).__init__()

        # Convolutional layers
        self.conv_layers = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        """Forward pass"""
        # Convolutional layers
        x = self.conv_layers(x)

        # Global average pooling
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.fc_layers(x)

        return x


# Training Example 1: Basic Training
def example_basic_training():
    """
    Example 1: Basic model training
    Shows how to use the ModelTrainer for simple training
    """
    print("=" * 80)
    print("EXAMPLE 1: Basic Model Training")
    print("=" * 80)

    # Initialize trainer with configuration
    trainer = ModelTrainer(
        config_path='resources/templates/training-config.yaml',
        distributed=False,
        debug=True
    )

    # Create data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Create dummy datasets for demonstration
    # In practice, replace with actual datasets
    train_data = torch.utils.data.TensorDataset(
        torch.randn(100, 3, 224, 224),
        torch.randint(0, 10, (100,))
    )
    val_data = torch.utils.data.TensorDataset(
        torch.randn(20, 3, 224, 224),
        torch.randint(0, 10, (20,))
    )

    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=16)

    # Build and train model
    model = trainer.build_model()
    trained_model = trainer.train(train_loader, val_loader, model)

    # Save model
    trainer.save(trained_model, 'models/basic_model.pth')

    print("\nBasic training completed!")


# Training Example 2: Custom Model Training
def example_custom_model_training():
    """
    Example 2: Training with custom model architecture
    Shows how to train a custom neural network
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Custom Model Training")
    print("=" * 80)

    # Initialize custom model
    model = CustomCNN(num_classes=10, dropout=0.5)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Setup optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    # Create dummy data
    train_loader = DataLoader(
        torch.utils.data.TensorDataset(
            torch.randn(100, 3, 224, 224),
            torch.randint(0, 10, (100,))
        ),
        batch_size=16,
        shuffle=True
    )

    # Training loop
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            # Forward pass
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 2 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}')

        # Update learning rate
        scheduler.step()

        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}')

    # Save custom model
    torch.save(model.state_dict(), 'models/custom_model.pth')
    print("\nCustom model training completed!")


# Training Example 3: Advanced Training with Callbacks
def example_advanced_training():
    """
    Example 3: Advanced training with callbacks and monitoring
    Shows how to implement custom callbacks and monitoring
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Advanced Training with Callbacks")
    print("=" * 80)

    # Custom callback for monitoring
    class TrainingCallback:
        """Custom callback for training monitoring"""

        def __init__(self):
            self.epoch_losses = []
            self.epoch_accuracies = []

        def on_epoch_end(self, epoch, loss, accuracy):
            """Called at end of each epoch"""
            self.epoch_losses.append(loss)
            self.epoch_accuracies.append(accuracy)
            print(f'Callback: Epoch {epoch} - Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%')

            # Check for improvement
            if len(self.epoch_accuracies) > 1:
                if accuracy > max(self.epoch_accuracies[:-1]):
                    print(f'  New best accuracy: {accuracy:.2f}%!')

    # Initialize model and callback
    model = CustomCNN(num_classes=10)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    callback = TrainingCallback()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Create data loaders
    train_loader = DataLoader(
        torch.utils.data.TensorDataset(
            torch.randn(100, 3, 224, 224),
            torch.randint(0, 10, (100,))
        ),
        batch_size=16,
        shuffle=True
    )
    val_loader = DataLoader(
        torch.utils.data.TensorDataset(
            torch.randn(20, 3, 224, 224),
            torch.randint(0, 10, (20,))
        ),
        batch_size=16
    )

    # Training with callbacks
    num_epochs = 5
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation phase
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        avg_loss = train_loss / len(train_loader)
        accuracy = 100. * correct / total

        # Trigger callback
        callback.on_epoch_end(epoch + 1, avg_loss, accuracy)

    print("\nAdvanced training with callbacks completed!")

    # Plot training history
    print("\nTraining History:")
    print(f"Losses: {callback.epoch_losses}")
    print(f"Accuracies: {callback.epoch_accuracies}")


# Main function
def main():
    """
    Run all training examples
    Demonstrates different training approaches
    """
    print("\n" + "#" * 80)
    print("# MODEL TRAINING EXAMPLES")
    print("#" * 80)

    # Create output directory
    Path('models').mkdir(exist_ok=True)

    try:
        # Example 1: Basic training
        example_basic_training()

        # Example 2: Custom model training
        example_custom_model_training()

        # Example 3: Advanced training with callbacks
        example_advanced_training()

        print("\n" + "=" * 80)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 80)

    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

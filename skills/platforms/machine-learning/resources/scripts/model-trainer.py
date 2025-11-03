#!/usr/bin/env python3
"""
Model Training Pipeline
Comprehensive training framework with distributed training, mixed precision, and experiment tracking
"""

import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging
from datetime import datetime
import json

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


class ModelTrainer:
    """
    Complete model training pipeline with:
    - Distributed training (DDP)
    - Mixed precision training
    - Experiment tracking (TensorBoard, W&B)
    - Checkpointing and early stopping
    - Learning rate scheduling
    """

    def __init__(
        self,
        config_path: str,
        distributed: bool = False,
        debug: bool = False
    ):
        """
        Initialize trainer with configuration

        Args:
            config_path: Path to YAML configuration file
            distributed: Enable distributed training
            debug: Enable debug mode with verbose logging
        """
        self.config = self._load_config(config_path)
        self.distributed = distributed
        self.debug = debug

        # Setup logging
        self._setup_logging()

        # Setup distributed training
        if distributed:
            self._setup_distributed()

        # Device configuration
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.logger.info(f"Using device: {self.device}")

        # Mixed precision training
        self.use_amp = self.config.get('training', {}).get('mixed_precision', True)
        self.scaler = GradScaler() if self.use_amp else None

        # Experiment tracking
        self._setup_tracking()

        # Training state
        self.current_epoch = 0
        self.best_metric = float('-inf')
        self.patience_counter = 0

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _setup_logging(self):
        """Setup logging configuration"""
        log_level = logging.DEBUG if self.debug else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def _setup_distributed(self):
        """Setup distributed training"""
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl')
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.logger.info(f"Distributed training: rank {self.rank}/{self.world_size}")

    def _setup_tracking(self):
        """Setup experiment tracking"""
        self.writers = {}

        # TensorBoard
        if TENSORBOARD_AVAILABLE and self.config.get('logging', {}).get('tensorboard', True):
            log_dir = Path('runs') / datetime.now().strftime('%Y%m%d_%H%M%S')
            self.writers['tensorboard'] = SummaryWriter(log_dir)
            self.logger.info(f"TensorBoard logging to: {log_dir}")

        # Weights & Biases
        if WANDB_AVAILABLE and self.config.get('logging', {}).get('wandb', False):
            wandb.init(
                project=self.config.get('logging', {}).get('wandb_project', 'ml-training'),
                config=self.config
            )
            self.writers['wandb'] = wandb
            self.logger.info("Weights & Biases logging enabled")

    def build_model(self) -> nn.Module:
        """Build model from configuration"""
        model_config = self.config['model']
        architecture = model_config['architecture']

        # Import and build model
        if architecture == 'resnet50':
            from torchvision.models import resnet50
            model = resnet50(pretrained=model_config.get('pretrained', False))
            # Replace classifier for custom number of classes
            num_classes = model_config.get('num_classes', 1000)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif architecture == 'efficientnet':
            from torchvision.models import efficientnet_b0
            model = efficientnet_b0(pretrained=model_config.get('pretrained', False))
            num_classes = model_config.get('num_classes', 1000)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

        model = model.to(self.device)

        # Wrap with DDP if distributed
        if self.distributed:
            model = DDP(model, device_ids=[self.rank])

        self.logger.info(f"Built model: {architecture}")
        return model

    def build_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """Build optimizer from configuration"""
        train_config = self.config['training']
        optimizer_name = train_config.get('optimizer', 'adam').lower()
        lr = train_config.get('learning_rate', 0.001)

        if optimizer_name == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=lr)
        elif optimizer_name == 'sgd':
            optimizer = optim.SGD(
                model.parameters(),
                lr=lr,
                momentum=train_config.get('momentum', 0.9)
            )
        elif optimizer_name == 'adamw':
            optimizer = optim.AdamW(
                model.parameters(),
                lr=lr,
                weight_decay=train_config.get('weight_decay', 0.01)
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        self.logger.info(f"Built optimizer: {optimizer_name}")
        return optimizer

    def build_scheduler(self, optimizer: optim.Optimizer) -> Optional[Any]:
        """Build learning rate scheduler"""
        train_config = self.config['training']
        scheduler_name = train_config.get('scheduler', None)

        if scheduler_name is None:
            return None

        if scheduler_name == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=train_config['epochs']
            )
        elif scheduler_name == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=train_config.get('step_size', 30),
                gamma=train_config.get('gamma', 0.1)
            )
        elif scheduler_name == 'reduce_on_plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='max',
                patience=train_config.get('scheduler_patience', 10)
            )
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")

        self.logger.info(f"Built scheduler: {scheduler_name}")
        return scheduler

    def train_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        epoch: int
    ) -> Dict[str, float]:
        """Train for one epoch"""
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)

            optimizer.zero_grad()

            # Mixed precision training
            if self.use_amp:
                with autocast():
                    output = model(data)
                    loss = criterion(output, target)

                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

            # Track metrics
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            # Log progress
            if batch_idx % 100 == 0:
                self.logger.info(
                    f'Epoch: {epoch} [{batch_idx}/{len(train_loader)}] '
                    f'Loss: {loss.item():.4f}'
                )

        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total

        return {'loss': avg_loss, 'accuracy': accuracy}

    def validate(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        criterion: nn.Module
    ) -> Dict[str, float]:
        """Validate model"""
        model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                loss = criterion(output, target)

                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total

        return {'loss': avg_loss, 'accuracy': accuracy}

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        model: Optional[nn.Module] = None
    ) -> nn.Module:
        """
        Complete training loop

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            model: Optional pre-built model

        Returns:
            Trained model
        """
        # Build components
        if model is None:
            model = self.build_model()
        optimizer = self.build_optimizer(model)
        scheduler = self.build_scheduler(optimizer)
        criterion = nn.CrossEntropyLoss()

        # Training configuration
        train_config = self.config['training']
        epochs = train_config['epochs']
        early_stopping = train_config.get('early_stopping', False)
        patience = train_config.get('patience', 10)

        # Training loop
        for epoch in range(epochs):
            self.current_epoch = epoch

            # Train
            train_metrics = self.train_epoch(
                model, train_loader, criterion, optimizer, epoch
            )

            # Validate
            val_metrics = self.validate(model, val_loader, criterion)

            # Log metrics
            self._log_metrics(epoch, train_metrics, val_metrics)

            # Learning rate scheduling
            if scheduler is not None:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_metrics['accuracy'])
                else:
                    scheduler.step()

            # Checkpointing
            if val_metrics['accuracy'] > self.best_metric:
                self.best_metric = val_metrics['accuracy']
                self._save_checkpoint(model, optimizer, epoch, val_metrics)
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            # Early stopping
            if early_stopping and self.patience_counter >= patience:
                self.logger.info(f"Early stopping at epoch {epoch}")
                break

        return model

    def _log_metrics(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float]
    ):
        """Log metrics to tracking services"""
        self.logger.info(
            f"Epoch {epoch}: "
            f"Train Loss: {train_metrics['loss']:.4f}, "
            f"Train Acc: {train_metrics['accuracy']:.2f}%, "
            f"Val Loss: {val_metrics['loss']:.4f}, "
            f"Val Acc: {val_metrics['accuracy']:.2f}%"
        )

        # TensorBoard
        if 'tensorboard' in self.writers:
            writer = self.writers['tensorboard']
            writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
            writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
            writer.add_scalar('Accuracy/train', train_metrics['accuracy'], epoch)
            writer.add_scalar('Accuracy/val', val_metrics['accuracy'], epoch)

        # Weights & Biases
        if 'wandb' in self.writers:
            wandb.log({
                'train/loss': train_metrics['loss'],
                'train/accuracy': train_metrics['accuracy'],
                'val/loss': val_metrics['loss'],
                'val/accuracy': val_metrics['accuracy'],
                'epoch': epoch
            })

    def _save_checkpoint(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float]
    ):
        """Save model checkpoint"""
        checkpoint_dir = Path('checkpoints')
        checkpoint_dir.mkdir(exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }

        checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Saved checkpoint: {checkpoint_path}")

        # Save best model
        best_path = checkpoint_dir / 'best_model.pth'
        torch.save(checkpoint, best_path)

    def evaluate(
        self,
        model: nn.Module,
        test_loader: DataLoader
    ) -> Dict[str, float]:
        """Evaluate model on test set"""
        criterion = nn.CrossEntropyLoss()
        return self.validate(model, test_loader, criterion)

    def save(self, model: nn.Module, path: str):
        """Save trained model"""
        torch.save(model.state_dict(), path)
        self.logger.info(f"Saved model to: {path}")


def main():
    """Example usage"""
    # Load configuration
    trainer = ModelTrainer('resources/templates/training-config.yaml')

    # Build model
    model = trainer.build_model()

    # Create dummy data loaders (replace with actual data)
    from torch.utils.data import TensorDataset, DataLoader
    X_train = torch.randn(1000, 3, 224, 224)
    y_train = torch.randint(0, 10, (1000,))
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    X_val = torch.randn(200, 3, 224, 224)
    y_val = torch.randint(0, 10, (200,))
    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # Train
    model = trainer.train(train_loader, val_loader)

    # Save
    trainer.save(model, 'models/trained_model.pth')


if __name__ == '__main__':
    main()

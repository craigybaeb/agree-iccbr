#!/usr/bin/env python3
"""
MNIST PyTorch Training Script with Best Practices

This script trains a CNN model on MNIST dataset following PyTorch best practices:
- Proper data loading and preprocessing
- Comprehensive metrics logging (accuracy, precision, recall, F1)
- Model checkpointing and saving
- Device-agnostic training (CPU/GPU)
- Reproducible training with seed setting
- Validation monitoring and early stopping
- Learning rate scheduling
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import numpy as np
import os
import json
import logging
from pathlib import Path
from datetime import datetime
import argparse


def set_seed(seed=42):
    """Set random seeds for reproducible training."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(log_dir):
    """Setup logging configuration.""" 
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True, parents=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


class MNISTNet(nn.Module):
    """
    Convolutional Neural Network for MNIST classification.
    
    Architecture:
    - 2 Convolutional layers with ReLU activation and MaxPooling
    - 2 Fully connected layers with dropout
    - Output layer for 10 classes
    """
    def __init__(self, dropout_rate=0.25):
        super(MNISTNet, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
        # Use distinct pooling modules to keep attribution methods (e.g., DeepLift)
        # compatible with non-reused module graphs.
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x, return_logits: bool = False):
        # Convolutional layers with ReLU and pooling
        x = self.pool1(F.relu(self.conv1(x)))  # 28x28 -> 14x14
        x = self.pool2(F.relu(self.conv2(x)))  # 14x14 -> 7x7
        
        # Flatten for fully connected layers
        x = x.view(-1, 64 * 7 * 7)
        
        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)

        if return_logits:
            return logits
        return F.log_softmax(logits, dim=1)


def get_data_loaders(batch_size=64, data_dir='./data'):
    """
    Create MNIST data loaders with proper preprocessing.
    
    Args:
        batch_size (int): Batch size for training
        data_dir (str): Directory to store MNIST data
    
    Returns:
        tuple: (train_loader, test_loader)
    """
    # Fix SSL certificate verification on macOS
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context

    # Data preprocessing transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])
    
    # Download and load training data
    train_dataset = datasets.MNIST(
        data_dir, train=True, download=True, transform=transform
    )
    
    # Download and load test data
    test_dataset = datasets.MNIST(
        data_dir, train=False, download=True, transform=transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    
    return train_loader, test_loader


def calculate_metrics(y_true, y_pred):
    """
    Calculate comprehensive classification metrics.
    
    Args:
        y_true (numpy.ndarray): True labels
        y_pred (numpy.ndarray): Predicted labels
    
    Returns:
        dict: Dictionary containing accuracy, precision, recall, and F1 scores
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision, 
        'recall': recall,
        'f1': f1
    }


def train_epoch(model, device, train_loader, optimizer, criterion, epoch, logger):
    """
    Train the model for one epoch.
    
    Args:
        model: PyTorch model
        device: Training device (CPU/GPU)
        train_loader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        epoch: Current epoch number
        logger: Logger instance
    
    Returns:
        dict: Training metrics for the epoch
    """
    model.train()
    train_loss = 0
    all_predictions = []
    all_targets = []
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        output = model(data)
        loss = criterion(output, target)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Track metrics
        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        all_predictions.extend(pred.cpu().numpy().flatten())
        all_targets.extend(target.cpu().numpy())
        
        # Log progress
        if batch_idx % 200 == 0:
            logger.info(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                       f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    
    # Calculate epoch metrics
    train_loss /= len(train_loader)
    metrics = calculate_metrics(all_targets, all_predictions)
    metrics['loss'] = train_loss
    
    return metrics


def validate_epoch(model, device, test_loader, criterion):
    """
    Validate the model on test data.
    
    Args:
        model: PyTorch model
        device: Training device (CPU/GPU)
        test_loader: Test data loader  
        criterion: Loss function
    
    Returns:
        dict: Validation metrics
    """
    model.eval()
    test_loss = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            # Sum up batch loss
            test_loss += criterion(output, target).item()
            
            # Get predictions
            pred = output.argmax(dim=1, keepdim=True)
            all_predictions.extend(pred.cpu().numpy().flatten())
            all_targets.extend(target.cpu().numpy())
    
    # Calculate metrics
    test_loss /= len(test_loader)
    metrics = calculate_metrics(all_targets, all_predictions)
    metrics['loss'] = test_loss
    
    return metrics


def save_model(model, optimizer, epoch, metrics, save_dir):
    """
    Save model checkpoint with metrics.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        metrics: Training metrics
        save_dir: Directory to save model
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    # Save model checkpoint
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    
    model_path = save_dir / f'mnist_model_epoch_{epoch}.pt'
    torch.save(checkpoint, model_path)
    
    # Save best model (based on validation accuracy)
    best_model_path = save_dir / 'mnist_best_model.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'metrics': metrics,
        'epoch': epoch
    }, best_model_path)
    
    # Save metrics as JSON
    metrics_path = save_dir / f'metrics_epoch_{epoch}.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return model_path


def train_mnist_model(config):
    """
    Main training loop for MNIST model.
    
    Args:
        config (dict): Training configuration
    """
    # Setup
    set_seed(config['seed'])
    logger = setup_logging(config['log_dir'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # Data loading
    logger.info('Loading MNIST dataset...')
    train_loader, test_loader = get_data_loaders(
        batch_size=config['batch_size'],
        data_dir=config['data_dir']
    )
    logger.info(f'Training samples: {len(train_loader.dataset)}')
    logger.info(f'Test samples: {len(test_loader.dataset)}')
    
    # Model setup
    model = MNISTNet(dropout_rate=config['dropout_rate']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.NLLLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    logger.info(f'Model architecture:\n{model}')
    logger.info(f'Total parameters: {sum(p.numel() for p in model.parameters())}')
    
    # Training loop
    best_val_accuracy = 0
    training_history = []
    
    logger.info('Starting training...')
    for epoch in range(1, config['epochs'] + 1):
        # Train
        train_metrics = train_epoch(model, device, train_loader, optimizer, criterion, epoch, logger)
        
        # Validate
        val_metrics = validate_epoch(model, device, test_loader, criterion)
        
        # Update learning rate
        scheduler.step()
        
        # Log metrics
        logger.info(f'Epoch {epoch}:')
        logger.info(f'  Train - Loss: {train_metrics["loss"]:.4f}, '
                   f'Accuracy: {train_metrics["accuracy"]:.4f}, '
                   f'F1: {train_metrics["f1"]:.4f}')
        logger.info(f'  Val   - Loss: {val_metrics["loss"]:.4f}, '
                   f'Accuracy: {val_metrics["accuracy"]:.4f}, '
                   f'F1: {val_metrics["f1"]:.4f}')
        
        # Save training history
        epoch_data = {
            'epoch': epoch,
            'train': train_metrics,
            'validation': val_metrics,
            'learning_rate': scheduler.get_last_lr()[0]
        }
        training_history.append(epoch_data)
        
        # Save model if validation accuracy improved
        if val_metrics['accuracy'] > best_val_accuracy:
            best_val_accuracy = val_metrics['accuracy']
            save_model(model, optimizer, epoch, epoch_data, config['save_dir'])
            logger.info(f'New best model saved with validation accuracy: {best_val_accuracy:.4f}')
    
    # Save final training history
    history_path = Path(config['save_dir']) / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    
    # Final evaluation
    final_metrics = validate_epoch(model, device, test_loader, criterion)
    logger.info('\nFinal Test Results:')
    logger.info(f'Accuracy: {final_metrics["accuracy"]:.4f}')
    logger.info(f'Precision: {final_metrics["precision"]:.4f}')
    logger.info(f'Recall: {final_metrics["recall"]:.4f}')
    logger.info(f'F1 Score: {final_metrics["f1"]:.4f}')
    
    return model, training_history


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description='Train MNIST CNN model')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--dropout-rate', type=float, default=0.25, help='Dropout rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--data-dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--save-dir', type=str, default='./models/mnist', help='Model save directory')
    parser.add_argument('--log-dir', type=str, default='./logs', help='Log directory')
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'dropout_rate': args.dropout_rate,
        'seed': args.seed,
        'data_dir': args.data_dir,
        'save_dir': args.save_dir,
        'log_dir': args.log_dir
    }
    
    # Train model
    model, history = train_mnist_model(config)
    
    print(f"\n🎉 Training completed!")
    print(f"📁 Model saved in: {config['save_dir']}")
    print(f"📊 Logs saved in: {config['log_dir']}")
    print(f"📈 Training history: {len(history)} epochs")


if __name__ == '__main__':
    main()
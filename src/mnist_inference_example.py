#!/usr/bin/env python3
"""
MNIST Model Inference Example

This script demonstrates how to load and use the trained MNIST model for inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import json
from pathlib import Path


class MNISTNet(nn.Module):
    """Same model architecture as training script."""
    def __init__(self, dropout_rate=0.25):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def load_trained_model(model_path):
    """
    Load a trained MNIST model.
    
    Args:
        model_path (str): Path to the saved model
        
    Returns:
        tuple: (model, metrics)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create and load model
    model = MNISTNet()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, checkpoint.get('metrics', {}), device


def preprocess_image(image_path):
    """
    Preprocess an image for MNIST prediction.
    
    Args:
        image_path (str): Path to image file
        
    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    image = Image.open(image_path)
    return transform(image).unsqueeze(0)  # Add batch dimension


def predict_digit(model, image_tensor, device):
    """
    Predict digit from image tensor.
    
    Args:
        model: Trained model
        image_tensor: Preprocessed image tensor
        device: Device to run inference on
        
    Returns:
        tuple: (predicted_digit, confidence_scores)
    """
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.exp(output)  # Convert log probabilities to probabilities
        predicted_digit = output.argmax(dim=1).item()
        confidence = probabilities.max().item()
        
    return predicted_digit, probabilities.squeeze().cpu().numpy()


def main():
    """Example usage of the trained MNIST model."""
    
    # Model path (adjust as needed)
    model_path = './models/mnist/mnist_best_model.pt'
    
    if not Path(model_path).exists():
        print(f"❌ Model not found at {model_path}")
        print("Please run the training script first: python train_mnist_model.py")
        return
    
    # Load model
    print("Loading trained MNIST model...")
    model, metrics, device = load_trained_model(model_path)
    
    print(f"✅ Model loaded successfully!")
    print(f"🎯 Device: {device}")
    
    if metrics:
        print(f"📊 Model performance:")
        if 'validation' in metrics:
            val_metrics = metrics['validation']
            print(f"   Accuracy: {val_metrics.get('accuracy', 'N/A'):.4f}")
            print(f"   F1 Score: {val_metrics.get('f1', 'N/A'):.4f}")
            print(f"   Precision: {val_metrics.get('precision', 'N/A'):.4f}")
            print(f"   Recall: {val_metrics.get('recall', 'N/A'):.4f}")
    
    # Example with test data (if available)
    try:
        from torchvision import datasets
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
        
        # Test on a few random samples
        print(f"\n🔍 Testing on random MNIST samples:")
        for i in range(5):
            sample_idx = np.random.randint(len(test_dataset))
            image, true_label = test_dataset[sample_idx]
            
            predicted_digit, confidence_scores = predict_digit(
                model, image.unsqueeze(0), device
            )
            
            confidence = confidence_scores[predicted_digit]
            
            print(f"Sample {i+1}: True={true_label}, Predicted={predicted_digit}, "
                  f"Confidence={confidence:.4f} {'✅' if predicted_digit == true_label else '❌'}")
    
    except Exception as e:
        print(f"Note: Could not test with MNIST data: {e}")
    
    print(f"\n💡 Usage example:")
    print(f"   model, metrics, device = load_trained_model('{model_path}')")
    print(f"   image_tensor = preprocess_image('my_digit.png')")
    print(f"   digit, confidence = predict_digit(model, image_tensor, device)")
    print(f"   print(f'Predicted digit: {{digit}} (confidence: {{confidence[digit]:.4f}})')")


if __name__ == '__main__':
    main()
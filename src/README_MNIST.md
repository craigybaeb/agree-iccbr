# MNIST PyTorch Training

This directory contains a comprehensive MNIST digit classification training pipeline built with PyTorch best practices.

## 🎯 Features

- **Complete CNN architecture** optimized for MNIST
- **Comprehensive metrics logging**: Accuracy, Precision, Recall, F1 Score
- **Model checkpointing** with automatic best model saving
- **Reproducible training** with seed setting
- **Device-agnostic** (CPU/GPU automatic detection)
- **Learning rate scheduling** and early stopping
- **Detailed logging** with training history export
- **Inference example** for using trained models

## 📁 Files

- `train_mnist_model.py` - Main training script with full pipeline
- `mnist_inference_example.py` - Example for using trained models
- `mnist_requirements.txt` - Python dependencies
- `README_MNIST.md` - This documentation

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r mnist_requirements.txt
```

### 2. Train the Model

```bash
# Basic training
python train_mnist_model.py

# Custom configuration
python train_mnist_model.py --epochs 15 --batch-size 128 --learning-rate 0.001
```

### 3. Use the Trained Model

```bash
python mnist_inference_example.py
```

## ⚙️ Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 10 | Number of training epochs |
| `--batch-size` | 64 | Training batch size |
| `--learning-rate` | 0.001 | Learning rate for Adam optimizer |
| `--dropout-rate` | 0.25 | Dropout rate for regularization |
| `--seed` | 42 | Random seed for reproducibility |
| `--data-dir` | `./data` | Directory for MNIST dataset |
| `--save-dir` | `./models/mnist` | Directory to save trained models |
| `--log-dir` | `./logs` | Directory for training logs |

## 📊 Metrics Tracked

The training script automatically calculates and logs:

- **Accuracy**: Overall classification accuracy
- **Precision**: Macro-averaged precision across all classes
- **Recall**: Macro-averaged recall across all classes  
- **F1 Score**: Macro-averaged F1 score across all classes
- **Training Loss**: Cross-entropy loss during training
- **Validation Loss**: Cross-entropy loss on test set

## 🏗️ Model Architecture

```
MNISTNet(
  (conv1): Conv2d(1, 32, kernel_size=3, padding=1)
  (conv2): Conv2d(32, 64, kernel_size=3, padding=1)  
  (pool): MaxPool2d(kernel_size=2, stride=2)
  (fc1): Linear(3136, 128)
  (fc2): Linear(128, 10)
  (dropout): Dropout(p=0.25)
)
```

**Total Parameters**: ~590K parameters

## 📈 Expected Performance

With default settings, you can expect:

- **Training Accuracy**: >99%
- **Test Accuracy**: >98%
- **Training Time**: ~2-3 minutes on CPU, <1 minute on GPU
- **Model Size**: ~2.3MB

## 💾 Output Files

After training, you'll find:

```
models/mnist/
├── mnist_best_model.pt          # Best model (highest validation accuracy)
├── mnist_model_epoch_X.pt       # Model from epoch X
├── metrics_epoch_X.json         # Metrics for epoch X
└── training_history.json        # Complete training history

logs/
└── training.log                 # Detailed training logs
```

## 🔍 Usage Example

```python
# Load trained model
from mnist_inference_example import load_trained_model, predict_digit
model, metrics, device = load_trained_model('models/mnist/mnist_best_model.pt')

# Predict on new image
image_tensor = preprocess_image('my_digit.png')
predicted_digit, confidence_scores = predict_digit(model, image_tensor, device)

print(f"Predicted: {predicted_digit} (confidence: {confidence_scores[predicted_digit]:.4f})")
```

## 🏆 Best Practices Implemented

- ✅ **Reproducible**: Fixed random seeds
- ✅ **Monitored**: Comprehensive metrics logging
- ✅ **Robust**: Proper validation and test splits
- ✅ **Efficient**: GPU acceleration when available
- ✅ **Maintainable**: Clean, documented code structure
- ✅ **Extensible**: Easy to modify for other datasets
- ✅ **Production-ready**: Model checkpointing and loading

## 🔧 Customization

To adapt for other datasets:

1. Modify the `get_data_loaders()` function for your dataset
2. Adjust the model architecture in `MNISTNet` class
3. Update input dimensions and number of classes as needed
4. Customize preprocessing transforms

## 🐛 Troubleshooting

**CUDA out of memory**: Reduce `--batch-size`  
**Slow training**: Ensure PyTorch with CUDA is installed  
**Poor accuracy**: Increase `--epochs` or adjust `--learning-rate`  
**Overfitting**: Increase `--dropout-rate` or add data augmentation

---

📝 **Note**: This script follows PyTorch best practices and is suitable for both educational and production use cases.
# Emotion Model Fine-Tuning

A comprehensive guide to fine-tuning a transformer-based model for emotion classification using the GoEmotions dataset.

## Overview

This project fine-tunes **DistilRoBERTa-base** on the [GoEmotions dataset](https://www.kaggle.com/datasets/debarshichanda/goemotions) to classify text into one of 28 emotion categories. The model learns to recognize subtle emotional nuances in human language and can be used for sentiment analysis, content moderation, and emotion-aware applications.

## Dataset Source

- **Dataset**: GoEmotions
- **Source**: [Kaggle - debarshichanda/goemotions](https://www.kaggle.com/datasets/debarshichanda/goemotions)
- **Size**: 
  - Train: 43,410 samples
  - Validation: 5,426 samples
  - Test: 5,427 samples
- **Format**: TSV (Tab-Separated Values) with text, label, and ID columns
- **Emotion Categories**: 28 distinct emotions including joy, disappointment, hope, fear, relief, gratitude, anger, sadness, nervousness, amusement, love, curiosity, confusion, surprise, disgust, and more

## Fine-Tuning Process

### 1. **Dependencies Installation**
```bash
pip install transformers datasets evaluate accelerate kagglehub
```

### 2. **Dataset Download & Loading**
- Download from Kaggle using `kagglehub.dataset_download()`
- Load three splits: train, dev (validation), test
- Parse TSV files with text, label, and ID columns
- Map numeric labels to emotion names using `emotions.txt`

### 3. **Data Preprocessing**
- **Label Simplification**: Extract primary emotion from multi-label format
- **Dataset Creation**: Convert pandas DataFrames to Hugging Face `Dataset` objects
- **Tokenization**: 
  - Tokenizer: DistilRoBERTa's tokenizer
  - Max length: 128 tokens
  - Padding: `max_length`
  - Truncation: Enabled

### 4. **Model Configuration**
- **Base Model**: `distilroberta-base`
  - Lightweight alternative to RoBERTa (25% smaller, 60% faster)
  - Pre-trained on large corpus of text
- **Task**: Sequence Classification
- **Num Labels**: 28 (one per emotion)
- **Config**: Custom `id2label` and `label2id` mappings

### 5. **Training Setup**
- **Optimizer**: Adam (default)
- **Learning Rate**: 2e-5
- **Batch Size**: 16 (train & eval)
- **Epochs**: 3
- **Evaluation Strategy**: Per epoch
- **Save Strategy**: Per epoch (keeps best model by F1 score)
- **Weight Decay**: 0.01 (L2 regularization)
- **Metrics**:
  - Accuracy: Standard classification accuracy
  - F1 Score: Weighted average (handles class imbalance)

### 6. **Training & Evaluation**
- Train using Hugging Face `Trainer` API
- Automatically evaluates on validation set each epoch
- Saves best model based on F1 score
- Evaluates final performance on test set

### 7. **Model Preservation**
- Save fine-tuned model weights (`model.safetensors`)
- Save tokenizer configuration and vocabulary
- Output directory: `./emotion_model/`

### 8. **Inference**
- Create text classification pipeline
- Run predictions on custom text
- Get scores for all 28 emotion categories
- Support for batch processing

## File Structure

```
emotion_model/
├── config.json                  # Model configuration
├── model.safetensors           # Fine-tuned model weights
├── tokenizer.json              # Tokenizer vocabulary
├── tokenizer_config.json       # Tokenizer settings
├── special_tokens_map.json     # Special token mappings
├── vocab.json                  # RoBERTa vocabulary
└── merges.txt                  # BPE merge file
fineTuning.ipynb               # Complete notebook with all steps
```

## Notebook Structure

| Cell | Purpose |
|------|---------|
| 1-4 | Install dependencies |
| 5-8 | Download and explore GoEmotions dataset |
| 9 | Load data into pandas DataFrames |
| 10-12 | Parse emotions and create label mappings |
| 13-15 | Simplify multi-label format to single labels |
| 16 | Create Hugging Face datasets and tokenize |
| 17 | Initialize model for fine-tuning |
| 18 | Setup trainer with metrics and training arguments |
| 19 | Execute fine-tuning and save model |
| 20 | Evaluate on test set |
| 21-22 | Perform inference with examples |
| 23 | Test on complex multi-emotion texts |

## Key Features

✅ **Multi-emotion Recognition**: Classifies text into 28 distinct emotion categories
✅ **Efficient Model**: DistilRoBERTa is 60% faster than full RoBERTa
✅ **Weighted F1**: Handles class imbalance with weighted metrics
✅ **Batch Processing**: Supports efficient inference on multiple texts
✅ **Production Ready**: Saves tokenizer and model for deployment

## Example Usage

```python
from transformers import pipeline

# Load the fine-tuned model
emotion_pipeline = pipeline(
    "text-classification",
    model="./emotion_model",
    return_all_scores=True
)

# Classify emotion
text = "I'm so excited about this opportunity!"
predictions = emotion_pipeline(text)
print(predictions)
```

## Emotion Categories

The model can classify into these 28 emotions:
admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity, desire, disappointment, disapproval, disgust, fear, gratitude, grief, hope, joy, love, nervousness, neutral, optimism, pride, realization, relief, remorse, sadness, surprise, neutral

## Performance Metrics

The model is evaluated using:
- **Accuracy**: Overall correctness across all samples
- **F1 Score (Weighted)**: Harmonic mean of precision and recall, weighted by class support

Results are saved after each epoch, with the best model selected based on validation F1 score.

## Technical Stack

- **Transformers**: Hugging Face transformers library for model architecture
- **Datasets**: Hugging Face datasets library for efficient data handling
- **PyTorch**: Deep learning framework (backend)
- **Kagglehub**: Download datasets from Kaggle
- **Evaluate**: Hugging Face evaluate library for metrics

## Notes

- **GPU Recommended**: Fine-tuning runs faster with CUDA-enabled GPU
- **Memory**: DistilRoBERTa requires ~500MB GPU memory for batch size 16
- **Training Time**: ~15-30 minutes on GPU, depends on hardware
- **Max Sequence Length**: 128 tokens (suitable for typical text snippets)
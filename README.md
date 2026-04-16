# Speaker Identification using MFCC & CNN

End‑to‑end speaker identification system using MFCC + delta + delta‑delta features on the AudioMNIST dataset. Built with TensorFlow/Keras.

## Results

- **Model:** 6‑layer CNN with BatchNorm, Dropout, Global Pooling
- **Evaluation:** Accuracy, weighted F1‑score, per‑speaker accuracy, confusion matrix
- **Best validation accuracy:** *0.9840*

## Key Features

- MFCC, delta, and delta‑delta feature extraction (Librosa)
- Stratified train/val/test splits
- Early stopping + learning rate scheduling + model checkpointing
- Per‑speaker accuracy analysis and confusion matrix visualization

# License Plate State Classifier

A deep learning project that predicts which US state a license plate belongs to based on its format and characters.

## Overview

This project uses a **Bidirectional LSTM with Attention** mechanism to classify license plates into 51 US states/territories. The model learns character-level patterns from synthetic license plate data generated based on real state-specific formats.

## Model Architecture

- **Embedding Layer**: Character-level encoding (vocab size: 39)
- **BiLSTM**: 2-layer bidirectional LSTM (hidden dim: 128)
- **Attention**: Learns to focus on important characters
- **Output**: 51-class classification (US states)

## Dataset

- **510,000** synthetic license plates (10,000 per state)
- Generated based on real US state license plate format rules
- Includes state-specific features:
  - Texas/Tennessee: No vowels (A, E, I, O, U)
  - Delaware/Rhode Island/New Hampshire: Pure numeric
  - Wyoming/Montana: County code prefixes
  - Hawaii: Island-specific first letters
  - California: Unique 1ABC234 format

## Results

| Metric | Value |
|--------|-------|
| Test Accuracy | ~50% |
| Baseline (format only) | ~43% |


## Files

```
├── plate_data.ipynb        # Dataset generator
├── Train_model.ipynb       # Model training
├── data_analysis.ipynb     # Format pattern analysis
└── dataset.csv             # Generated dataset
```

## Usage

1. Generate dataset:
```python
# Run plate_data.ipynb to generate dataset.csv
```

2. Train model:
```python
# Run Train_model.ipynb
# Requires: PyTorch, pandas, sklearn
```

## Analysis
### Why ~50% accuracy?

Analysis shows that **31 out of 51 states share similar format patterns** (e.g., `ABC1234`), making them hard to distinguish. Only 20 states have unique formats that the model can reliably identify.the most common pattern LLDDD (~153k samples) is used by 17 states; LL-DDDD is shared by 11, and LLLDDD by 10. Thus, even perfect identification of the format still maps to multiple states, making the label fundamentally hard to determine from the string alone.
<img width="699" height="316" alt="截屏2026-01-08 22 56 00" src="https://github.com/user-attachments/assets/02cf2215-18ee-4148-ba0a-6bc4d642b734" />
<img width="702" height="284" alt="截屏2026-01-08 22 56 23" src="https://github.com/user-attachments/assets/a860c193-03eb-41d1-bf83-6d1373fd2801" />
<img width="704" height="289" alt="截屏2026-01-08 22 56 44" src="https://github.com/user-attachments/assets/e34721cd-97a4-46c0-a2b7-8a4adc699ade" />

### Summary:
States with UNIQUE format: 20
States with SHARED format: 31

Model can easily distinguish: 20 states
Model struggles with: 31 states (same format as others)

This is why accuracy plateaus around 50%

## Future Improvements

- Add artificial distinguishing features for states with shared formats
- Consider computer vision
- Include real license plate images (CNN + LSTM)

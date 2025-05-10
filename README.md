# Code Redundancy Detection Using Deep Learning

This project presents a deep learning-based approach to detect redundancy between two code snippets. Unlike traditional plagiarism detectors that rely solely on syntax, this system identifies **semantic similarity**, making it effective even when the structure, formatting, or variable names differ. The model is particularly useful in educational platforms, code review systems, and automated refactoring tools.

The dataset consists of code files collected from various open-source coding platforms and repositories. These were grouped into 50 problem folders, each containing 500 solution files. A custom script was used to generate 50,000 labeled pairs — balanced between redundant (same problem) and non-redundant (different problems).

The model is built using a **Bidirectional LSTM Siamese architecture** that processes two code snippets in parallel, extracts semantic features, and compares them using feature operations like absolute difference and element-wise product. The system is trained on tokenized sequences and outputs a **redundancy score** (0–1) with a classification decision based on a threshold (default 0.6).

## 🔑 Key Features

- Deep learning model using **BiLSTM Siamese network**
- Supports **batch and single pair prediction**
- Includes **training, evaluation, and CLI-based inference**
- Evaluation metrics: **90% accuracy**, **0.967 AUC**
- Output visualizations: 
  - ROC curve  
  - Precision-recall curve  
  - Confusion matrix  
  - Redundancy score distribution

## ⚙️ Usage

### Training the Model
```bash
python LSTM_Model.py
```

### Running Predictions
```bash
python code_redundancy_predictor.py
```

The system works via command line and supports code input as text or file paths.

## 🚀 Future Enhancements

- Support for multiple languages (Python, Java, etc.)
- Integration of transformer-based models (e.g., CodeBERT)
- Web or GUI interface
- IDE plugin compatibility

---

This project showcases the potential of AI in understanding source code at a deeper level and offers a scalable foundation for intelligent code analysis tools.

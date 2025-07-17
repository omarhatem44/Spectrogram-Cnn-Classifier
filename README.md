# 🎧 Spectrogram CNN Classifier

A deep learning pipeline for binary classification of spectrogram images derived from physiological audio signals, such as heart sounds.  
The model utilizes a pre-trained **ResNet-18** convolutional neural network, fine-tuned on a dataset based on **PhysioNet**.  
It processes spectrogram images of segmented audio clips and achieves a test accuracy of **95.04%**, demonstrating strong performance in distinguishing between normal and abnormal patterns.

---

## 🧠 Project Overview

This project presents a deep learning approach for classifying spectrogram images extracted from medical audio data (e.g., phonocardiograms).  
The model effectively learns frequency-time representations to distinguish between normal and abnormal heart sounds.

---

## 🔍 Key Features

-  Preprocessing of physiological audio into spectrogram images  
-  Transfer learning using **ResNet-18**  
-  Weighted loss function to handle class imbalance  
-  📢 Accuracy: **95.04%** on the test set  
-  Outputs predictions in CSV format for easy post-processing  
-  Includes training, validation, and testing pipeline  

---

## 📁 Project Structure

```
spectrogram-cnn-classifier/
│
├── data/                      # Contains CSV labels and sample spectrograms
│   └── physionet_labels_multi_segment.csv
│
├── model/                     # Trained model and test predictions
│   ├── spectrogram_model.pth
│   └── test_predictions.csv
│
├── Train_Model.py               # Main training and evaluation script
├── README.md                  # Project documentation
├── requirements.txt           # Python dependencies
               
```

---

## ⚙️ Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

Main Python packages used:

- `torch`  
- `torchvision`  
- `pandas`  
- `tqdm`  
- `Pillow`  

---

## 🚀 How to Run

1. Make sure your data is structured correctly:
   - `data/physionet_labels_multi_segment.csv` contains filenames and labels
   - Spectrogram images are saved as `.png` files with names matching the CSV

2. Run the script:

```bash
python Cnn_Model.py
```

- If a saved model is found, it will be loaded automatically.
- If not, the model will be trained from scratch and saved for future use.

---

## 🧪 Output

- ✅ **Training and testing accuracy** printed to the terminal  
- ✅ **Predictions CSV** saved to:  `model/test_predictions.csv`  
- ✅ **Trained model weights** saved to:  `model/spectrogram_model.pth`

---

## 📊 Example Test Output

```text
Test Accuracy: 0.9504
Predictions saved to: model/test_predictions.csv
```

---

## 📌 Notes

- The full dataset is not included in the repository due to size limitations.
- You can easily adapt this pipeline for:
  - Multi-class classification
  - Other types of medical or physiological audio signals
  - Integration with clinical decision support tools

---

## 📬 Contact

**Author**: Omar Hatem Mohamed 

**Linkedin**: www.linkedin.com/in/omar-ellaban-355ba4369

**Email**: omarhatemmoahemd@gmail.com

---

## 📝 License

This project is licensed under the **MIT License**.  
Feel free to use, modify, and distribute with attribution.

---

# 🎧 Speech Emotion Recognizer

An end-to-end deep learning project that classifies **emotions from speech audio** using a custom-trained **CNN + BiLSTM + Attention** model. The application is deployed via **Streamlit** and supports real-time inference on uploaded `.wav` files.

Built using **PyTorch**, **Librosa**, and **Streamlit**, with a stylish UI for real-world deployment.

---

##  Project Description

This project focuses on recognizing emotions conveyed in speech using deep learning. It leverages **audio signal processing** to extract meaningful features and classifies speech into one of the following emotions:

> **Angry, Calm, Disgust, Fearful, Happy, Sad, Surprised**

☑️ **Note**: The *Neutral* class was intentionally **excluded** from training due to its persistent low per-class accuracy (below 75%), which negatively impacted overall model performance.

✅ Overall Accuracy: 82.86290322580645

✅ Weighted F1 Score: 82.7966227377242

✅ Per Class Accuracy:

   0: 90.79%
   
   1: 94.74%
   
   2: 79.49%
   
   3: 76.32%
   
   4: 81.58%
   
   5: 68.42% -----  Neutral
   
   6: 78.95%
   
   7: 84.62%

---

## 🔍 Pre-processing Methodology

1. **Audio Loading & Trimming**
   - `.wav` files are loaded using `librosa`.
   - Silence at the start and end is trimmed.

2. **Feature Extraction**
   - MFCCs (40)
   - Delta & Delta²
   - RMS energy
   - For selected classes (e.g., fearful, sad), slight enhancements like pitch shift or pre emphasis were applied to improve model separability.

3. **Padding & Encoding**
   - Variable-length features are padded for batch training.
   - Emotions are label-encoded using `LabelEncoder`.

4. **Train-Test Split**
   - Stratified split (80-20) to preserve class distribution.
   - Balanced sampling ensures fair representation of all classes.

---

##  Model Pipeline

- **Architecture**
  - 1D Convolution Layer
  - MaxPooling
  - BiLSTM (Bidirectional LSTM)
  - Attention Mechanism
  - Fully Connected Layer (7 output classes)

- **Loss Function**
  - `CrossEntropyLoss` with optional **class weighting** for minority classes

- **Optimizer**
  - `Adam` with optional `ReduceLROnPlateau` for dynamic learning rate control

- **Training Strategy**
  - Trained until loss ≤ best threshold (~0.6199)
  - Model saved only when accuracy & F1 > 80%

---

##  Accuracy Metrics

| Metric                  | Result         |
|-------------------------|----------------|
| ✅ Overall Accuracy      | **~83.84%**     |
| ✅ Weighted F1 Score     | **~83.73%**     |
| ✅ Per-Class Accuracy    | **≥ 75%** (All) |

### Example per-class breakdown:

✅ Overall Accuracy: 83.4061135371179

✅ Weighted F1 Score: 83.41116487853209

✅ Per Class Accuracy:

   0: 89.47%
   
   1: 96.05%
   
   2: 89.74%
   
   3: 76.32%
   
   4: 77.63%
   
   5: 77.63%
   
   6: 76.92%


##  Project Structure

 speech-emotion-recognizer/

├── web-app.py                   # Streamlit frontend
├── Main.py                      # Model + predict logic
├── final_emotion_model.pth      # Trained PyTorch model
├── Bg_image.jpg                 # Background image for web-app
├── MAIN_MODEL.ipynb             # Jupyter Notebook with model training
├── DEMO_VIDEO.mp4               # 2 min Video showing the use of the app
├── requirements.txt             # Python dependencies
└── README.md                    # This file

## TECH STACK

| Component            | Description                                                            |
| -------------------- | ---------------------------------------------------------------------- |
|  **PyTorch**       | Deep learning framework used to build and train the CNN + BiLSTM model |
|  **Librosa**       | Audio feature extraction and preprocessing                             |
|  **Streamlit**     | Real-time web app framework for deploying the model                    |
|  **Scikit-learn**  | Label encoding, stratified splitting, and performance metrics          |
|  **NumPy / Torch** | Numerical operations, tensor handling, and sequence padding            |

##  How to Use This App

Follow the steps below to set up, run, and interact with the Speech Emotion Recognizer locally.

---

### Install Dependencies

pip install -r requirements.txt

###  Run the App

streamlit run web-app.py

###  Upload and Predict

- Use the upload button to select a `.wav` audio file
- The model will:
  - Play back the file
  - Show the predicted **emotion**
  - Display use cases and tech stack in a beautiful UI

##  Acknowledgements

- [RAVDESS Dataset](https://zenodo.org/record/1188976)
- [PyTorch](https://pytorch.org/)
- [Librosa](https://librosa.org/)
- [Streamlit](https://streamlit.io/)

---

##  Author

**Swapnil Sharan**  

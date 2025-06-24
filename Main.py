import torch
import torch.nn as nn
import librosa # type: ignore
import numpy as np
import os
from torch.nn.utils.rnn import pad_sequence
from io import BytesIO

# ----------------- MODEL DEFINITION -----------------
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim * 2, 1)

    def forward(self, lstm_out):
        attn_weights = torch.softmax(self.attn(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        return context

class EmotionCNNBiLSTM(nn.Module):
    def __init__(self, input_dim=121, cnn_out=64, lstm_hidden=128, num_classes=7):
        super(EmotionCNNBiLSTM, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, cnn_out, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(cnn_out),
            nn.MaxPool1d(kernel_size=2)
        )
        self.lstm = nn.LSTM(input_size=cnn_out, hidden_size=lstm_hidden,
                            num_layers=1, batch_first=True, bidirectional=True)
        self.attention = Attention(hidden_dim=lstm_hidden)
        self.fc = nn.Linear(lstm_hidden * 2, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        context = self.attention(lstm_out)
        out = self.fc(context)
        return out

# ----------------- FEATURE EXTRACTION -----------------
def extract_features(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    rms = librosa.feature.rms(y=y)
    return np.vstack([mfcc, delta, delta2, rms]).T

def preprocess_audio(file_path, max_length=400):
    y, sr = librosa.load(file_path, sr=22050)
    y_trimmed, _ = librosa.effects.trim(y)

    features = extract_features(y_trimmed, sr)
    features_tensor = torch.tensor(features, dtype=torch.float32)

    # Pad sequence to match training input shape
    if features_tensor.shape[0] < max_length:
        pad_len = max_length - features_tensor.shape[0]
        pad = torch.zeros(pad_len, features_tensor.shape[1])
        features_tensor = torch.cat([features_tensor, pad], dim=0)
    else:
        features_tensor = features_tensor[:max_length, :]

    return features_tensor.unsqueeze(0)  # Add batch dimension

# ----------------- PREDICTION -----------------
def predict_emotion(file_path, model_path='final_emotion_model.pth'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = EmotionCNNBiLSTM()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Preprocess and predict
    input_tensor = preprocess_audio(file_path).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        predicted_label = torch.argmax(output, dim=1).item()

    label_map = {
        0: 'angry',
        1: 'calm',
        2: 'disgust',
        3: 'fearful',
        4: 'happy',
        5: 'sad',
        6: 'surprised'
    }

    return label_map[predicted_label]


# FOR TESTING

if __name__ == '__main__':
    file_path = "03-02-06-02-02-02-21.wav"
    result = predict_emotion(file_path)
    print("Predicted Emotion:", result)

#Emotion (02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).



def predict_emotion_from_uploaded_file(uploaded_file, model_path='final_emotion_model.pth'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EmotionCNNBiLSTM()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Read from uploaded in-memory file
    y, sr = librosa.load(BytesIO(uploaded_file.read()), sr=22050)
    y_trimmed, _ = librosa.effects.trim(y)
    features = extract_features(y_trimmed, sr)

    tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    padded_tensor = pad_sequence(tensor, batch_first=True).to(device)

    with torch.no_grad():
        output = model(padded_tensor)
        predicted = torch.argmax(output, dim=1).item()

    label_map = {
        0: 'angry',
        1: 'calm',
        2: 'disgust',
        3: 'fearful',
        4: 'happy',
        5: 'sad',
        6: 'surprised'
    }

    return label_map[predicted]

#Emotion (02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
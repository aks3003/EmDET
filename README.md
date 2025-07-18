# 🎭 EmDet – Emotion Detection using Deep Learning  

EmDet is a deep learning-based project that performs **emotion detection** from audio and other inputs using **TensorFlow/Keras**. It also integrates advanced audio processing and augmentation techniques for better model generalization.

---

## ✅ Features
- **Audio-based emotion detection** with feature extraction using **Librosa**.
- Data augmentation using **audiomentations** (e.g., background noise, Gaussian SNR).
- Pre-trained `.keras` models for continued training and evaluation.
- Tracks **validation accuracy** using `ModelCheckpoint`.
- Stores multiple versions of models including `BestModel.keras` and `UpdatedModel.keras`.

---

## 📂 Project Structure
EmDet/
│
├── ALL/ # Contains data and model files
│ ├── data/ # Raw and processed emotion data
│ └── working/ # Saved models
│ ├── ACCURACY*.keras # Models with different accuracies
│ ├── BestModel.keras # Best performing model
│ └── UpdatedModel.keras # Model after further training
│
├── Audio.ipynb # Audio emotion detection notebook
├── Kitchen_2_Dishes_4_1_white_80.wav # Example audio file
├── requirements.txt # Python dependencies
└── README.md # Project documentation

yaml
Copy code

---

## 🛠 Tech Stack
- **Python 3.x**
- **TensorFlow / Keras** (Model training)
- **Librosa** (Audio feature extraction)
- **audiomentations** (Data augmentation)
- **torchaudio**
- **NumPy**, **Pandas**, **Matplotlib**, **Seaborn**
- **TQDM** (Progress bars)

---

## 📦 Installation
Clone the repository:
```bash
git clone https://github.com/aks3003/EmDet.git
cd EmDet
Install dependencies:

bash
Copy code
pip install -r requirements.txt
🚀 Training / Continue Training
To load a pre-trained model and continue training:

python
Copy code
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint

# Load model
model = load_model("ALL/working/ACCURACY0.3747.keras")

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Save best model during training
checkpoint = ModelCheckpoint('ALL/working/BestModel.keras', monitor='val_accuracy', save_best_only=True, mode='max')

# Train
history = model.fit(X_train, y_train, epochs=10, callbacks=[checkpoint], validation_data=(X_val, y_val), initial_epoch=1)

# Save updated model
model.save('ALL/working/UpdatedModel.keras')
🔍 Evaluation
python
Copy code
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
🎧 Audio Emotion Detection
The Audio.ipynb notebook demonstrates:

Loading audio files (e.g., Kitchen_2_Dishes_4_1_white_80.wav)

Extracting features (MFCCs) using Librosa

Augmenting audio using audiomentations:

python
Copy code
from audiomentations import Compose, AddBackgroundNoise, AddGaussianSNR

augment = Compose([
    AddBackgroundNoise(sounds_path="path/to/noise/folder"),
    AddGaussianSNR(min_SNR_dB=10, max_SNR_dB=50)
])
✅ Future Improvements
Real-time audio + video emotion detection using webcam and mic.

Integration with EfficientNet/ResNet for better accuracy.

Deploy via Flask / Streamlit.

Convert to TensorFlow Lite for mobile apps.

👤 Author
Akshith Mynampati

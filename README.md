# Deepfake-Audio-Detection

# Project Title: Audio Classification using Deep Learning

## Overview

This project implements an **audio classification** model using **MFCC feature extraction** and a deep learning model built with **TensorFlow/Keras**. The model is trained on an audio dataset to classify different sounds.

## Requirements

To run this project, install the following dependencies:

```bash
pip install tensorflow librosa numpy pandas scikit-learn matplotlib
```

## Dataset: UrbanSound8K

This project uses the UrbanSound8K dataset, which is around 5GB in size. You need to manually download it from:

[UrbanSound8K Dataset](https://urbansounddataset.weebly.com/urbansound8k.html)

After downloading, extract the dataset and place it in the project directory.

## Project Workflow

### 1. Data Preprocessing

- Load audio files using `librosa`
- Extract **MFCC features** (40 coefficients per audio file)
- Normalize and reshape data for model input

### 2. Label Encoding

- Convert class labels into numerical format using `LabelEncoder`
- Use `to_categorical()` to convert labels into one-hot encoded vectors

### 3. Train-Test Split

- Split dataset into training and testing sets using `train_test_split()`

### 4. Model Training

- A deep learning model (likely a **CNN** or fully connected neural network) is trained using:
  - `model.fit()` with training data
  - `ModelCheckpoint` to save the best model
  - `validation_data` for performance tracking

### 5. Model Evaluation

- The trained model is evaluated using `model.evaluate()` on test data
- Accuracy is printed as the final metric

### 6. Prediction on New Audio

- Load an audio file (`.wav` format)
- Extract MFCC features and reshape
- Predict class label using `model.predict()`
- Convert numerical prediction back to the original class name using `inverse_transform()`

## Running the Project

1. **Train the model**
   ```python
   model.fit(X_train, y_train, batch_size=32, epochs=100, validation_data=(X_test, y_test))
   ```
2. **Evaluate the model**
   ```python
   test_accuracy = model.evaluate(X_test, y_test, verbose=0)
   print("Test Accuracy:", test_accuracy[1])
   ```
3. **Predict a new audio file**
   ```python
   filename = "UrbanSound8K/drill.wav"
   audio, sample_rate = librosa.load(filename, res_type='kaiser_fast')
   mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
   mfccs_scaled_features = np.mean(mfccs_features.T, axis=0).reshape(1, -1)
   predicted_probs = model.predict(mfccs_scaled_features)
   predicted_label = np.argmax(predicted_probs, axis=1)
   prediction_class = labelencoder.inverse_transform(predicted_label)
   print("Predicted Class:", prediction_class)
   ```

## Expected Output

- The model predicts the class of an input audio file.
- Accuracy metric is displayed after evaluation.

## Notes

- Ensure all dependencies are installed before running.
- Modify dataset paths accordingly before execution.




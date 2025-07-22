#  Emotion Classification: Happy vs Disgust (Custom CNN Model)

##  Objective
This project aims to classify facial expressions as **Happy** or **Disgust** using a **custom-built Convolutional Neural Network (CNN)** trained on grayscale images from a manually curated subset of the **FER-2013** dataset.

---

##  Dataset Preparation & Preprocessing

###  Data Selection
- Downloaded FER-2013 dataset from Kaggle.
- Selected only **150 images of Happy** and **150 images of Disgust** manually for training (total: 300 images).
- Ensured a **balanced dataset** with equal samples per class.

### Preprocessing Steps

- All images resized to **48x48 pixels**. 
- Images are converted to grayscale 
- Normalization is performed  :Pixel values scaled from **[0, 255] → [0, 1]** for stable and faster training. 
- Label Encoding : Used `class_mode='binary'` for binary classification (0 = Disgust, 1 = Happy). 
- Batch_size=16 is fixed that Feeds model with 16 images at a time. 
- Shuffle is enabled for better generalization during training. 


---

##  Model Architecture:  CNN

A lightweight yet powerful **Convolutional Neural Network (CNN)** was designed from scratch

###  Input Layer
- **Shape**: `(48, 48, 1)` → Resized grayscale image with 1 channel.

---

###  Convolutional Block 1

- Conv2D: Applies 32 filters of size 3x3 to learn low-level features like edges and textures.
-  ReLU Activation: Introduces non-linearity to help the model learn complex patterns.
-   BatchNormalization: Stabilizes training by normalizing the output of the convolutional layer.
-    MaxPooling: Downsamples feature maps by selecting maximum values in 2x2 windows, reducing spatial size and computation

---

###  Convolutional Block 2
Same as of Convolution Block 1 with 64 filters

---

###  Convolutional Block 3
Same as of Convolution block 1 with 128 filters that captures high-level features such as facial structure and emotion patterns.

---

###  Fully Connected Layers (Classifier)
- `Flatten` → Converts feature maps into a 1D vector
- `Dense(128)` → Captures complex patterns and combinations
- `Dropout(0.3)` → Prevents overfitting by randomly disabling 30% of neurons
- `Dense(1, activation='sigmoid')` → Outputs probability for binary classification (0 = Disgust, 1 = Happy)

---

##  Training

- Training time: ⏱ **Under 5 minutes**
- Optimizer: `Adam`
- Loss: `Binary Crossentropy`
- Evaluation Metrics: **Accuracy, Precision, Recall, F1-Score**

---

##  Results

Accuracy **81%**

---


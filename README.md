# Handwritten_Digit_Recognition

This project focuses on developing a **multi-class image classification system** using machine learning techniques to **accurately recognize handwritten digits (0–9)** from image data.  
It includes a training pipeline using the **MNIST dataset** and a user-friendly **web app** built with **Streamlit** to upload and classify custom digit images.

---

## Contents of the Project

1. **Dataset Overview**
   - Uses the `mnist_784` dataset from `sklearn.datasets.fetch_openml`
   - Images are grayscale and sized 28x28 pixels

2. **Model Training**
   - Model used: `LogisticRegression` from scikit-learn
   - Normalized pixel values to range [0, 1]
   - Split data into training (80%) and testing (20%) sets
   - Trained on 60,000+ images

3. **Model Evaluation**
   - Achieved high accuracy on test data
   - Evaluated using:
     - Accuracy score
     - Classification report
     - Confusion matrix (visualized using seaborn)

4. **Custom Image Prediction**
   - User can upload their own digit images
   - Image is resized to 28x28 and inverted if needed
   - Model predicts the digit and displays it

5. **Web App Interface (Streamlit)**
   - Interactive interface for uploading images
   - Displays prediction results and processed image
   - Real-time model inference using a saved `.pkl` model

---

##  How the Analysis Was Performed

### ➤ Step 1: Loading the Dataset  
- The dataset was loaded using `fetch_openml` with `as_frame=False`.

### ➤ Step 2: Preprocessing  
- Pixel values were normalized (divided by 255.0)  
- Labels were converted to integers

### ➤ Step 3: Model Training  
- Used `LogisticRegression` with `max_iter=2000`  
- Fitted the model on training data

### ➤ Step 4: Evaluation  
- Predictions were made on the test set  
- Calculated accuracy and confusion matrix  
- Generated a classification report

### ➤ Step 5: Saving the Model  
- The model was saved to `mnist_logistic_model.pkl` using `joblib`

### ➤ Step 6: Streamlit App  
- User uploads an image (28x28, white digit on black background)  
- Image is preprocessed, inverted if needed  
- Model predicts the digit and result is displayed in the browser

---

## Requirements
streamlit
scikit-learn
joblib
opencv-python-headless
numpy
Pillow
matplotlib

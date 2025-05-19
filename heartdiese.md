# Heart Disease Prediction Using Multilayer Perceptron (MLP)

This document explains a Python implementation of a neural network model for heart disease prediction using the Kaggle Heart Disease dataset. The code creates, trains, and evaluates a Multilayer Perceptron (MLP) model to predict the presence of heart disease based on various health metrics.

## Table of Contents
1. [Libraries and Dependencies](#libraries-and-dependencies)
2. [Data Loading and Exploration](#data-loading-and-exploration)
3. [Data Preprocessing](#data-preprocessing)
4. [Neural Network Model Architecture](#neural-network-model-architecture)
5. [Model Training](#model-training)
6. [Evaluation and Performance Metrics](#evaluation-and-performance-metrics)
7. [Visualization](#visualization)

## Libraries and Dependencies

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
```

The code uses several key libraries:
- **NumPy & Pandas**: For data manipulation and analysis
- **Matplotlib & Seaborn**: For data visualization
- **Scikit-learn**: For data preprocessing and evaluation metrics
- **TensorFlow/Keras**: For building and training the neural network model

## Data Loading and Exploration

```python
# Load the Heart Disease dataset from Kaggle
df = pd.read_csv('heart.csv')

# Step 1: Check for missing values and handle if any
print("Step 1: Checking for missing values")
print(df.isnull().sum())

# Step 2: Display input and output features of the dataset
print("\nStep 2: Displaying dataset information")
print(f"Dataset shape: {df.shape}")
print("\nFeature descriptions:")
print(df.describe())

# Display first few rows of the dataset
print("\nFirst 5 rows of the dataset:")
print(df.head())
```

This section:
1. Loads the heart disease dataset from a CSV file
2. Checks for any missing values in the dataset
3. Displays basic information about the dataset, including its shape and statistical descriptions
4. Shows the first 5 rows of the dataset for a quick overview

## Data Preprocessing

```python
# Input features and output
X = df.drop('target', axis=1)  # Input features (all columns except 'target')
y = df['target']               # Output feature (presence of heart disease)

print("\nInput features:")
print(X.columns.tolist())
print("\nOutput feature: target (0 = no disease, 1 = disease)")

# Step 3: Encode non-numeric input attributes using Label Encoder
print("\nStep 3: Encoding non-numeric attributes")
# Checking data types
print("Data types of features:")
print(X.dtypes)

# In case there are categorical features
categorical_cols = X.select_dtypes(include=['object']).columns
encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le
    print(f"Encoded column: {col}")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

The preprocessing steps include:
1. Separating the features (X) from the target variable (y)
   - The target variable is 'target' (0 = no heart disease, 1 = heart disease)
   - Features include all other columns
2. Encoding any categorical features using LabelEncoder
   - The code identifies columns with non-numeric data types
   - Each categorical column is transformed into numeric values
3. Splitting the data into training (80%) and testing (20%) sets using a fixed random seed (42) for reproducibility

## Neural Network Model Architecture

```python
# Step 4: Constructing MLP model with configuration 11x128x64x32x1
print("\nStep 4: Constructing MLP model with configuration 11x128x64x32x1")

# Create the MLP model
model = Sequential([
    # Input layer
    Dense(128, activation='relu', input_shape=(X.shape[1],)),
    Dropout(0.2),  # Adding dropout for regularization
    
    # Hidden layers
    Dense(64, activation='relu'),
    Dropout(0.2),
    
    Dense(32, activation='relu'),
    Dropout(0.2),
    
    # Output layer (sigmoid for binary classification)
    Dense(1, activation='sigmoid')
])

# Compile the model with Adam optimizer
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Display model summary
model.summary()
```

The MLP architecture follows a 11x128x64x32x1 configuration:
- **Input Layer**: The number of features (11) as input
- **First Hidden Layer**: 128 neurons with ReLU activation and 20% dropout
- **Second Hidden Layer**: 64 neurons with ReLU activation and 20% dropout
- **Third Hidden Layer**: 32 neurons with ReLU activation and 20% dropout
- **Output Layer**: 1 neuron with sigmoid activation (for binary classification)

The model is compiled with:
- **Optimizer**: Adam with a learning rate of 0.001
- **Loss Function**: Binary crossentropy (standard for binary classification)
- **Metrics**: Accuracy

## Model Training

```python
# Train the model
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)
```

The model is trained for 50 epochs with:
- A batch size of 32 samples
- 20% of the training data is used as validation data
- Training progress is displayed (verbose=1)
- The training history is saved for later visualization and analysis

## Evaluation and Performance Metrics

```python
# Step 5: Predict heart disease for test data and evaluate performance
print("\nStep 5: Evaluating model performance")

# Predict on test data
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)  # Convert probabilities to binary predictions

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\nAccuracy: {accuracy:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Precision: {precision:.4f}")
print(f"F1-score: {f1:.4f}")

# Print detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```

After training, the model is evaluated using:
1. **Predictions on test data**:
   - The model outputs probabilities which are converted to binary predictions (0 or 1) using a threshold of 0.5
2. **Confusion Matrix**: Shows true positives, false positives, true negatives, and false negatives
3. **Performance Metrics**:
   - **Accuracy**: Overall prediction accuracy
   - **Recall**: Ability to find all positive cases (sensitivity)
   - **Precision**: Accuracy of positive predictions
   - **F1-score**: Harmonic mean of precision and recall
4. **Classification Report**: Detailed breakdown of precision, recall, and F1-score for each class

## Visualization

```python
# Plot training history
plt.figure(figsize=(12, 5))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Plot confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()
```

The code produces three key visualizations:
1. **Training and Validation Accuracy** over epochs
   - Helps identify if the model is learning and when it starts to overfit
2. **Training and Validation Loss** over epochs
   - Shows how the loss decreases during training and when overfitting begins
3. **Confusion Matrix Heatmap**
   - Visual representation of model predictions vs. actual values
   - True positives and true negatives on the diagonal
   - False positives and false negatives off the diagonal

These visualizations help assess model performance and identify areas for improvement.
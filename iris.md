# Iris Flower Classification Using MLP

This document explains a Python implementation of a neural network model for classifying iris flowers using the famous Iris dataset. The code creates, trains, and evaluates a Multilayer Perceptron (MLP) model to classify iris flowers into three species based on their measurements.

## Table of Contents
1. [Libraries and Dependencies](#libraries-and-dependencies)
2. [Data Loading and Exploration](#data-loading-and-exploration)
3. [Data Preprocessing](#data-preprocessing)
4. [Feature Engineering](#feature-engineering)
5. [Neural Network Model Architecture](#neural-network-model-architecture)
6. [Model Training](#model-training)
7. [Evaluation and Performance Metrics](#evaluation-and-performance-metrics)
8. [Visualization](#visualization)

## Libraries and Dependencies

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
```

The code uses several key libraries:
- **NumPy & Pandas**: For data manipulation and analysis
- **Matplotlib & Seaborn**: For data visualization
- **Scikit-learn**: For data preprocessing, feature engineering, and evaluation metrics
- **TensorFlow/Keras**: For building and training the neural network model

## Data Loading and Exploration

```python
# Load the Iris dataset from Kaggle or use the built-in one from sklearn
try:
    # Try loading from file first
    df = pd.read_csv('iris.csv')
except FileNotFoundError:
    # If file not found, use sklearn's built-in dataset
    from sklearn.datasets import load_iris
    iris = load_iris()
    df = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                      columns=iris['feature_names'] + ['species'])
    # Convert numeric target to species name for better visualization
    species_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    df['species'] = df['species'].astype(int).map(species_map)

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
1. Attempts to load the Iris dataset from a CSV file
2. If the file is not found, it loads the dataset from scikit-learn's built-in datasets
3. Checks for any missing values in the dataset
4. Displays basic information about the dataset, including its shape and statistical descriptions
5. Shows the first 5 rows of the dataset for a quick overview

## Data Preprocessing

```python
# Identify input features and output feature
if 'species' in df.columns:
    target_col = 'species'
elif 'target' in df.columns:
    target_col = 'target'
elif 'class' in df.columns:
    target_col = 'class'
else:
    # If none of the expected column names are found, assume the last column is the target
    target_col = df.columns[-1]

X = df.drop(target_col, axis=1)  # Input features (all columns except target)
y = df[target_col]               # Output feature (species)

print("\nInput features:")
print(X.columns.tolist())
print("\nOutput feature:", target_col)

# Step 3: Encode output attribute using one hot encoder
print("\nStep 3: Encoding output attribute using one hot encoder")
# Fix: Use sparse_output=False instead of sparse=False
encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(y.values.reshape(-1, 1))
print("One-hot encoded output:")
print(y_encoded[:5])  # Display first 5 encoded values

# Get class names
class_names = encoder.categories_[0]
print(f"Classes: {class_names}")
```

The preprocessing steps include:
1. Identifying the target column (trying common names like 'species', 'target', or 'class')
2. Separating the features (X) from the target variable (y)
3. One-hot encoding the target variable (converting categorical labels to binary vectors)
   - This is necessary for multi-class classification with neural networks
4. Extracting class names from the encoder for later use

## Feature Engineering

```python
# Step 4: Shuffle the dataset and count tuples in each class
print("\nStep 4: Shuffling dataset and counting tuples in each class")

# Create a combined dataset with inputs and one-hot encoded outputs
combined_df = pd.DataFrame(np.column_stack([X, y_encoded]), 
                           columns=list(X.columns) + [f'class_{i}' for i in range(len(class_names))])

# Shuffle the dataset
shuffled_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Extract shuffled features and targets
X_shuffled = shuffled_df[X.columns]
y_shuffled = shuffled_df[[f'class_{i}' for i in range(len(class_names))]]

# Count and display number of tuples in each class (using original labels for clarity)
class_counts = y.value_counts()
print("\nNumber of samples in each class:")
for class_name, count in class_counts.items():
    print(f"{class_name}: {count}")

# Step 5: Normalize input attributes using standard scalar
print("\nStep 5: Normalizing input attributes using StandardScaler")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_shuffled)
print("Scaled features (first 5 samples):")
print(X_scaled[:5])
```

The feature engineering process includes:
1. Creating a combined dataset with features and one-hot encoded targets
2. Shuffling the dataset to ensure randomness
3. Extracting the shuffled features and targets
4. Counting the number of samples in each class
5. Normalizing the input features using StandardScaler
   - This scales features to have zero mean and unit variance
   - Helps the neural network converge faster and perform better

## Data Splitting

```python
# Step 6: Split dataset into training/validation/test sets in 70:15:15 ratio
print("\nStep 6: Splitting dataset into training/validation/test sets in 70:15:15 ratio")

# First split: 70% training, 30% remaining
X_train, X_temp, y_train, y_temp = train_test_split(
    X_scaled, y_shuffled.values, test_size=0.3, random_state=42
)

# Second split: Split the remaining 30% into 15% validation and 15% test (which is 50% of the 30%)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
```

The data is split into three sets:
1. Training set (70%): Used to train the model
2. Validation set (15%): Used for monitoring training progress
3. Test set (15%): Used for final evaluation of model performance

## Neural Network Model Architecture

```python
# Step 7: Construct an MLP with configuration 4x32x16x8x3
print("\nStep 7: Constructing MLP model with configuration 4x32x16x8x3")

# Create the MLP model
model = Sequential([
    # Input layer
    Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),  # Adding dropout for regularization
    
    # Hidden layers
    Dense(16, activation='relu'),
    Dropout(0.2),
    
    Dense(8, activation='relu'),
    Dropout(0.2),
    
    # Output layer (softmax for multi-class classification)
    Dense(len(class_names), activation='softmax')
])

# Compile the model with Adam optimizer
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Display model summary
model.summary()
```

The MLP architecture follows a 4×32×16×8×3 configuration:
- **Input Layer**: 4 features (sepal length, sepal width, petal length, petal width)
- **First Hidden Layer**: 32 neurons with ReLU activation and 20% dropout
- **Second Hidden Layer**: 16 neurons with ReLU activation and 20% dropout
- **Third Hidden Layer**: 8 neurons with ReLU activation and 20% dropout
- **Output Layer**: 3 neurons (one for each species) with softmax activation

The model is compiled with:
- **Optimizer**: Adam with a learning rate of 0.001
- **Loss Function**: Categorical crossentropy (standard for multi-class classification)
- **Metrics**: Accuracy

## Model Training

```python
# Train the model
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=16,
    validation_data=(X_val, y_val),
    verbose=1
)
```

The model is trained for 50 epochs with:
- A batch size of 16 samples
- Validation data for monitoring training progress
- Training progress is displayed (verbose=1)
- Training history is saved for later visualization and analysis

## Evaluation and Performance Metrics

```python
# Step 8: Predict species of Iris flower for test data and evaluate performance
print("\nStep 8: Evaluating model performance")

# Predict on test data
y_pred_prob = model.predict(X_test)
y_pred_classes = np.argmax(y_pred_prob, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

# Convert class indices to class names for better readability
y_pred_labels = [class_names[idx] for idx in y_pred_classes]
y_test_labels = [class_names[idx] for idx in y_test_classes]

# Confusion matrix
cm = confusion_matrix(y_test_classes, y_pred_classes)
print("\nConfusion Matrix:")
print(cm)

# Calculate metrics
accuracy = accuracy_score(y_test_classes, y_pred_classes)
macro_recall = recall_score(y_test_classes, y_pred_classes, average='macro')
micro_recall = recall_score(y_test_classes, y_pred_classes, average='micro')
macro_precision = precision_score(y_test_classes, y_pred_classes, average='macro')
micro_precision = precision_score(y_test_classes, y_pred_classes, average='micro')
macro_f1 = f1_score(y_test_classes, y_pred_classes, average='macro')
micro_f1 = f1_score(y_test_classes, y_pred_classes, average='micro')

print(f"\nWeighted Avg. Accuracy: {accuracy:.4f}")
print(f"Macro Recall: {macro_recall:.4f}")
print(f"Micro Recall: {micro_recall:.4f}")
print(f"Macro Precision: {macro_precision:.4f}")
print(f"Micro Precision: {micro_precision:.4f}")
print(f"Macro F1-score: {macro_f1:.4f}")
print(f"Micro F1-score: {micro_f1:.4f}")

# Print detailed classification report
print("\nClassification Report:")
print(classification_report(y_test_classes, y_pred_classes, target_names=class_names))
```

After training, the model is evaluated using:
1. **Predictions on test data**:
   - The model outputs probabilities which are converted to class indices using argmax
   - Class indices are mapped back to class names for better readability
2. **Confusion Matrix**: Shows how many samples of each true class were predicted as each class
3. **Performance Metrics**:
   - **Accuracy**: Overall prediction accuracy
   - **Recall**: Ability to find all positive cases (macro and micro averaged)
   - **Precision**: Accuracy of positive predictions (macro and micro averaged)
   - **F1-score**: Harmonic mean of precision and recall (macro and micro averaged)
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
plt.savefig('training_history.png')
plt.show()

# Plot confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('confusion_matrix.png')
plt.show()
```

The code produces two key visualizations:
1. **Training and Validation Metrics** over epochs
   - Accuracy and Loss curves
   - Helps identify if the model is learning and when overfitting begins
2. **Confusion Matrix Heatmap**
   - Visual representation of model predictions vs. actual values
   - Shows how many samples of each true class were predicted as each class
   - Class names are used as labels for better readability
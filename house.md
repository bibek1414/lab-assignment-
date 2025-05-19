# California Housing Price Prediction Using MLP

This document explains a Python implementation of a neural network model for predicting housing prices using the California Housing dataset from Kaggle. The code creates, trains, and evaluates a Multilayer Perceptron (MLP) regression model to predict median house values.

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
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
```

The code uses several key libraries:
- **NumPy & Pandas**: For data manipulation and analysis
- **Matplotlib & Seaborn**: For data visualization
- **Scikit-learn**: For data preprocessing, feature engineering, and evaluation metrics
- **TensorFlow/Keras**: For building and training the neural network model

## Data Loading and Exploration

```python
# Load the California Housing dataset from Kaggle
df = pd.read_csv('housing.csv')
price_col = 'median_house_value'  # California Housing price column

# Step 1: Check for missing values and handle if any
print("Step 1: Checking for missing values")
print(df.isnull().sum())

# Handle missing values if any
if df.isnull().sum().sum() > 0:
    # Fill numeric columns with mean values
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        # Avoid chained assignment warning by using loc
        df.loc[:, col] = df[col].fillna(df[col].mean())
    
    print("Missing values after handling:")
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
1. Loads the California Housing dataset from a CSV file
2. Checks for any missing values in the dataset
3. Handles missing values in numeric columns using mean imputation
4. Displays basic information about the dataset, including its shape and statistical descriptions
5. Shows the first 5 rows of the dataset for a quick overview

## Data Preprocessing

```python
# Input features and output
X = df.drop(price_col, axis=1)  # Input features (all columns except price)
y = df[price_col]               # Output feature (house price)

print("\nInput features:")
print(X.columns.tolist())
print("\nOutput feature:", price_col)

# Step 3: Handle categorical columns
print("\nStep 3: Handling categorical attributes")
# Check for categorical columns
categorical_cols = X.select_dtypes(include=['object']).columns
numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
print(f"Categorical columns: {categorical_cols.tolist()}")
print(f"Numeric columns: {numeric_cols.tolist()}")
```

The preprocessing steps include:
1. Separating the features (X) from the target variable (y)
   - The target variable is 'median_house_value' (the price column)
   - Features include all other columns
2. Identifying categorical and numerical columns for appropriate preprocessing

## Feature Engineering

```python
# Set up a column transformer to handle both numeric and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# Apply the preprocessing to the features
X_processed = preprocessor.fit_transform(X)

# For categorical features, get the names of one-hot encoded columns
if len(categorical_cols) > 0:
    # Get the one-hot encoded feature names
    cat_encoder = preprocessor.named_transformers_['cat']
    encoded_features = cat_encoder.get_feature_names_out(categorical_cols)
    
    # All feature names (numeric + encoded categorical)
    feature_names = list(numeric_cols) + list(encoded_features)
    print(f"\nProcessed features shape: {X_processed.shape}")
    print(f"Features after one-hot encoding: {len(feature_names)}")
else:
    feature_names = list(numeric_cols)

# Step 4: Scale the output variable
print("\nStep 4: Scaling the output variable")
y_scaler = StandardScaler()
y_scaled = y_scaler.fit_transform(y.values.reshape(-1, 1)).flatten()

print("Processed features (first 5 samples):")
print(X_processed[:5])
print("\nScaled target (first 5 samples):")
print(y_scaled[:5])
```

The feature engineering process includes:
1. Setting up a `ColumnTransformer` to apply different transformations to different column types
   - Numerical columns: Standardization (zero mean, unit variance)
   - Categorical columns: One-hot encoding
2. Applying transformations to create processed features
3. Extracting feature names from the transformed data
4. Scaling the target variable (house prices) using standardization
   - This is common in regression problems to help the model converge

## Data Splitting

```python
# Step 5: Split dataset into training/validation/test sets in 70:15:15 ratio
print("\nStep 5: Splitting dataset into training/validation/test sets in 70:15:15 ratio")

# First split: 70% training, 30% remaining
X_train, X_temp, y_train, y_temp = train_test_split(
    X_processed, y_scaled, test_size=0.3, random_state=42
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
2. Validation set (15%): Used for hyperparameter tuning and early stopping
3. Test set (15%): Used for final evaluation of model performance

## Neural Network Model Architecture

```python
# Step 6: Construct an MLP with configuration [input]x128x64x32x16x1
print("\nStep 6: Constructing MLP model")
# Note: input size now includes one-hot encoded features

# Create the MLP model
model = Sequential([
    # Input layer - using X_train.shape[1] for proper input size with encoded features
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),  # Adding dropout for regularization
    
    # Hidden layers
    Dense(64, activation='relu'),
    Dropout(0.2),
    
    Dense(32, activation='relu'),
    Dropout(0.2),
    
    Dense(16, activation='relu'),
    Dropout(0.2),
    
    # Output layer (linear activation for regression)
    Dense(1, activation='linear')
])

# Compile the model with Adam optimizer
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mean_squared_error',
    metrics=['mae']  # Mean Absolute Error
)

# Display model summary
model.summary()
```

The MLP architecture follows a [input]×128×64×32×16×1 configuration:
- **Input Layer**: The number of features (including one-hot encoded features)
- **First Hidden Layer**: 128 neurons with ReLU activation and 20% dropout
- **Second Hidden Layer**: 64 neurons with ReLU activation and 20% dropout
- **Third Hidden Layer**: 32 neurons with ReLU activation and 20% dropout
- **Fourth Hidden Layer**: 16 neurons with ReLU activation and 20% dropout
- **Output Layer**: 1 neuron with linear activation (for regression)

The model is compiled with:
- **Optimizer**: Adam with a learning rate of 0.001
- **Loss Function**: Mean Squared Error (standard for regression problems)
- **Metrics**: Mean Absolute Error (MAE)

## Model Training

```python
# Set up early stopping to prevent overfitting
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True,
    verbose=1
)

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=20,  # More epochs with early stopping
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping],
    verbose=1
)
```

The model is trained with:
- A maximum of 200 epochs
- A batch size of 32 samples
- Early stopping mechanism to prevent overfitting
  - Monitors validation loss
  - Has patience of 20 epochs (stops if no improvement for 20 consecutive epochs)
  - Restores best weights from the epoch with lowest validation loss
- Validation data for monitoring training progress
- Training history is saved for later visualization and analysis

## Evaluation and Performance Metrics

```python
# Step 7: Predict house price for test data
print("\nStep 7: Predicting house prices for test data")
y_pred_scaled = model.predict(X_test).flatten()

# Step 8: Perform inverse transformation of predicted and actual house price
print("\nStep 8: Performing inverse transformation of predicted and actual house prices")
y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
y_test_actual = y_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

# Display actual vs predicted prices for first few samples
comparison_df = pd.DataFrame({
    'Actual Price': y_test_actual,
    'Predicted Price': y_pred,
    'Difference': y_test_actual - y_pred
})
print("\nActual vs Predicted Prices (first 10 samples):")
print(comparison_df.head(10))

# Step 9: Compute and display RMSE, MAE and MAPE
print("\nStep 9: Computing RMSE, MAE, and MAPE")

# Calculate RMSE (Root Mean Squared Error)
rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))

# Calculate MAE (Mean Absolute Error)
mae = mean_absolute_error(y_test_actual, y_pred)

# Calculate MAPE (Mean Absolute Percentage Error)
# Adding small epsilon to avoid division by zero
mape = np.mean(np.abs((y_test_actual - y_pred) / (y_test_actual + 1e-10))) * 100

print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"MAPE: {mape:.2f}%")
```

After training, the model is evaluated using:
1. **Predictions on test data**:
   - The model outputs scaled predictions which are then inversely transformed to the original scale
2. **Comparison of actual vs. predicted prices**:
   - A dataframe showing actual prices, predicted prices, and the difference
3. **Performance Metrics**:
   - **RMSE (Root Mean Squared Error)**: Square root of the average of squared differences between predicted and actual values
   - **MAE (Mean Absolute Error)**: Average of absolute differences between predicted and actual values
   - **MAPE (Mean Absolute Percentage Error)**: Average percentage difference between predicted and actual values

## Visualization

```python
# Plot training history
plt.figure(figsize=(12, 5))

# Plot loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()

# Plot MAE
plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Model MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png')
plt.show()

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test_actual, y_pred, alpha=0.7)
plt.plot([min(y_test_actual), max(y_test_actual)], [min(y_test_actual), max(y_test_actual)], 'r--')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted House Prices')
plt.grid(True)
plt.savefig('actual_vs_predicted.png')
plt.show()

# Plot prediction error distribution
plt.figure(figsize=(10, 6))
errors = y_test_actual - y_pred
plt.hist(errors, bins=30, alpha=0.7)
plt.xlabel('Prediction Error')
plt.ylabel('Count')
plt.title('Distribution of Prediction Errors')
plt.grid(True)
plt.savefig('error_distribution.png')
plt.show()
```

The code produces three key visualizations:
1. **Training and Validation Metrics** over epochs
   - Loss (MSE) and Mean Absolute Error (MAE)
   - Helps identify if the model is learning and when overfitting begins
2. **Actual vs. Predicted Prices Scatter Plot**
   - Perfect predictions would lie on the diagonal red line
   - Shows how well the model predicts across the price range
3. **Prediction Error Distribution**
   - Histogram of prediction errors
   - Ideally should be centered around zero and approximately normally distributed
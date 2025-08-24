import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Dense, Dropout, Conv1D, Flatten, BatchNormalization, Input,
                                     LeakyReLU, Add, GlobalAveragePooling1D, Multiply, Reshape, MultiHeadAttention)
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from imblearn.over_sampling import SMOTE
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Load datasets
lidar_df = pd.read_csv(r"cnn\data set\training data\synthetic_lidar_threats.csv", low_memory=False)
radar_df = pd.read_csv(r"cnn\data set\training data\synthetic_radar_threats.csv", low_memory=False)

# Select relevant features
lidar_features = ['X', 'Y', 'Z', 'Intensity']
radar_features = ['Azimuth', 'Elevation', 'Range', 'Speed', 'RCS', 'Power', 'Noise']

# Merge datasets
combined_df = pd.concat([lidar_df[lidar_features], radar_df[radar_features]], axis=1)

# Feature Engineering
combined_df['Velocity_Magnitude'] = np.sqrt(combined_df['Speed']**2 + combined_df['Azimuth']**2)
combined_df['Velocity_Range_Ratio'] = combined_df['Speed'] / (combined_df['Range'] + 1e-5)
combined_df['Log_Intensity'] = np.log1p(combined_df['Intensity'])
combined_df['Power_Noise_Ratio'] = combined_df['Power'] / (combined_df['Noise'] + 1e-5)

# Target variable
label_col = 'Threat_Type'
if label_col in lidar_df.columns:
    y = lidar_df[label_col]
else:
    raise KeyError(f"'{label_col}' not found in lidar_df")

# Encode target variable
encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(y.values.reshape(-1, 1))

# Handle missing values
imputer = IterativeImputer(max_iter=10, random_state=42)
combined_df[:] = imputer.fit_transform(combined_df)

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(combined_df)

# Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)

# Add Gaussian noise for augmentation
X += np.random.normal(0, 0.02, X.shape)

# Reshape for CNN
X = X.reshape(X.shape[0], X.shape[1], 1)

# Stratified K-Fold Cross Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train_index, val_index in skf.split(X, np.argmax(y, axis=1)):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

# Define Squeeze-and-Excitation Block
def se_block(input_tensor, ratio=8):
    channels = input_tensor.shape[-1]
    squeeze = GlobalAveragePooling1D()(input_tensor)
    excitation = Dense(channels // ratio, activation='relu', use_bias=False)(squeeze)
    excitation = Dense(channels, activation='sigmoid', use_bias=False)(excitation)
    excitation = Reshape((1, channels))(excitation)
    return Multiply()([input_tensor, excitation])

# Define Transformer-based Attention Layer
def transformer_block(x, num_heads=4, key_dim=64):
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x, x)
    return Add()([x, attn_output])

# Define Residual Block
def residual_block(x, filters, kernel_size=3):
    res = Conv1D(filters, kernel_size=kernel_size, padding="same")(x)
    res = BatchNormalization()(res)
    res = LeakyReLU()(res)
    res = Conv1D(filters, kernel_size=kernel_size, padding="same")(res)
    res = BatchNormalization()(res)
    return Add()([x, res])

# Define CNN Model
def build_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    x = Conv1D(128, kernel_size=5, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0005))(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(0.2)(x)

    x = residual_block(x, 128)
    x = se_block(x)
    x = transformer_block(x)

    x = Conv1D(64, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(0.2)(x)

    x = Conv1D(32, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(0.2)(x)

    x = Flatten()(x)
    x = Dense(256, kernel_regularizer=tf.keras.regularizers.l2(0.0005))(x)
    x = LeakyReLU()(x)
    x = Dropout(0.3)(x)

    x = Dense(128, kernel_regularizer=tf.keras.regularizers.l2(0.0005))(x)
    x = LeakyReLU()(x)
    x = Dropout(0.3)(x)

    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    return model

# Build and compile model
optimizer = AdamW(learning_rate=0.0001, weight_decay=1e-5)
cnn_model = build_model((X_train.shape[1], 1), y_train.shape[1])
cnn_model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=['accuracy'])

# Callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train model
cnn_model.fit(X_train, y_train, epochs=100, batch_size=64, validation_data=(X_val, y_val), callbacks=[early_stop, reduce_lr])

# Save model and tools
cnn_model.save("cnn_model_lidar_radar_enhanced.h5")
joblib.dump(scaler, "scaler_lidar_radar_enhanced.pkl")
joblib.dump(encoder, "encoder_lidar_radar_enhanced.pkl")

# ---- Model Evaluation ----

# Get model predictions
y_pred_probs = cnn_model.predict(X_val)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_val, axis=1)

# Calculate Performance Metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_true, y_pred)
class_report = classification_report(y_true, y_pred)

# Print Results
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)

# Plot Confusion Matrix
plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=encoder.categories_[0], yticklabels=encoder.categories_[0])
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()

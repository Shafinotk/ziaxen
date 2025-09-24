import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Load datasets
can_bus_df = pd.read_csv(r"newagent\lstm\dataset\training data\synthetic_can_bus.csv")
gps_imu_df = pd.read_csv(r"newagent\lstm\dataset\training data\synthetic_gps_imu.csv")
egomotion_df = pd.read_csv(r"newagent\lstm\dataset\training data\synthetic_egomotion_with_bias.csv")

# Extract target
y = can_bus_df['Threat_Type'].copy()

# Drop label columns from features
can_bus_df.drop(columns=['Threat_Type', 'Attack_Severity'], inplace=True, errors='ignore')
gps_imu_df.drop(columns=['Threat_Type', 'Attack_Severity'], inplace=True, errors='ignore')
egomotion_df.drop(columns=['Threat_Type', 'Attack_Severity'], inplace=True, errors='ignore')

# Encode Gear_Position if categorical
encoder = LabelEncoder()
if 'Gear_Position' in can_bus_df.columns and can_bus_df['Gear_Position'].dtype == 'object':
    can_bus_df['Gear_Position'] = encoder.fit_transform(can_bus_df['Gear_Position'])

# Encode the labels
y = encoder.fit_transform(y)

# Combine all features
combined_df = pd.concat([can_bus_df, gps_imu_df, egomotion_df], axis=1)

# Handle missing values
combined_df.fillna(combined_df.median(), inplace=True)

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(combined_df)

# Handle class imbalance
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)

# Reshape for LSTM (1 timestep)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Data augmentation with Gaussian noise
noise = np.random.normal(0, 0.05, X_train.shape)
X_train_aug = X_train + noise

# Model settings
num_classes = len(np.unique(y))
activation = "sigmoid" if num_classes == 2 else "softmax"
loss_fn = "binary_crossentropy" if num_classes == 2 else "sparse_categorical_crossentropy"

# Build LSTM model
model = Sequential([
    Bidirectional(LSTM(128, return_sequences=True, input_shape=(X.shape[1], 1))),
    LayerNormalization(),
    Dropout(0.3),

    Bidirectional(LSTM(64, return_sequences=True)),
    LayerNormalization(),
    Dropout(0.3),

    Bidirectional(LSTM(32)),
    LayerNormalization(),
    Dense(32, activation='relu'),
    Dropout(0.2),

    Dense(num_classes, activation=activation)
])

# Compile
model.compile(optimizer=Adam(learning_rate=0.001), loss=loss_fn, metrics=['accuracy'])

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

# Train the model
history = model.fit(X_train_aug, y_train,
                    validation_data=(X_test, y_test),
                    epochs=25,
                    batch_size=64,
                    callbacks=[early_stop, reduce_lr])

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ Model trained successfully!")
print(f"📊 Validation Accuracy: {accuracy * 100:.2f}%")

# Predict and print classification metrics
y_pred = np.argmax(model.predict(X_test), axis=1) if num_classes > 2 else (model.predict(X_test) > 0.5).astype("int32").flatten()

print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=encoder.classes_))

print("🔍 Confusion Matrix:")
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

# Optional: Visualize confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# Save model and tools
model.save("lstm_model_optimized.h5")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(encoder, "label_encoder.pkl")

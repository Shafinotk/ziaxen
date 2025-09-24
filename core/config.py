import os

# ================== CONFIG ==================
ART_DIR = os.getenv("ARTIFACT_DIR", "./artifacts")
DB_PATH = os.getenv("DB_PATH", "./db/events.db")
CSV_MIRROR = os.getenv("CSV_MIRROR", "false").lower() == "true"
CSV_PATH = os.getenv("CSV_PATH", "./data/events.csv")

EGOMOTION_MODEL_PATH = os.getenv("EGOMOTION_MODEL", os.path.join(ART_DIR, "egomotion_model.h5"))
EGOMOTION_SCALER_PATH = os.getenv("EGOMOTION_SCALER", os.path.join(ART_DIR, "egomotion_scaler.pkl"))
EGOMOTION_ENCODER_PATH = os.getenv("EGOMOTION_ENCODER", os.path.join(ART_DIR, "egomotion_label_encoder.pkl"))

LIDAR_RADAR_MODEL_PATH = os.getenv("LIDAR_RADAR_MODEL", os.path.join(ART_DIR, "lidar_radar_model.h5"))
LIDAR_RADAR_SCALER_PATH = os.getenv("LIDAR_RADAR_SCALER", os.path.join(ART_DIR, "lidar_radar_scaler.pkl"))
LIDAR_RADAR_ENCODER_PATH = os.getenv("LIDAR_RADAR_ENCODER", os.path.join(ART_DIR, "lidar_radar_label_encoder.pkl"))

# --- Labels (YOUR models) ---
EGO_LABELS = ["Acceleration_Anomaly", "Normal", "Sensor_Spoofing", "Steering_Mismatch"]
LR_LABELS  = ["Normal", "intensity_manipulation", "jamming", "spoofing"]

# --- Human-readable descriptions ---
EGO_DESC = {
    "Acceleration_Anomaly": "Longitudinal acceleration inconsistent with throttle/brake/velocity dynamics.",
    "Normal": "No anomalous ego-motion detected.",
    "Sensor_Spoofing": "IMU/GNSS/pose signals inconsistent with vehicle kinematics (potential spoofing).",
    "Steering_Mismatch": "Steering angle vs yaw/heading response inconsistent (possible EPS command injection/sensor fault)."
}
LR_DESC = {
    "Normal": "No anomaly in LiDAR/Radar signatures.",
    "intensity_manipulation": "Return intensities/power inconsistent with range/geometry; potential fake returns.",
    "jamming": "Wideband noise or saturation typical of jamming.",
    "spoofing": "Coherent false targets/ghosts suggesting sensor spoofing."
}

# --- Threat thresholds ---
EGO_LOW  = float(os.getenv("EGO_LOW", 0.50))
EGO_HIGH = float(os.getenv("EGO_HIGH", 0.90))
LR_LOW   = float(os.getenv("LR_LOW", 0.50))
LR_HIGH  = float(os.getenv("LR_HIGH", 0.90))

COMBINED_POLICY = os.getenv("COMBINED_POLICY", "max_severity")

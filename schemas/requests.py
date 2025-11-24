from typing import List, Optional, Literal
from pydantic import BaseModel, Field

class EgoReading(BaseModel):
    Throttle_Position: float
    Brake_Pressure: float
    Steering_Angle: float
    Vehicle_Speed: float
    Gear_Position: float
    Rotation_X: float
    Rotation_Y: float
    Rotation_Z: float
    Position_X: float
    Rotation_Q1: float
    Rotation_Q2: float
    Rotation_Q3: float
    Position_Y: float
    Velocity_X: float
    Velocity_Y: float
    Velocity_Z: float
    Position_Z: float
    Bias1: float
    Bias2: float
    Bias3: float
    Bias4: float
    Latitude: float
    Longitude: float
    Speed: float
    Heading: float
    Acceleration_X: float
    Acceleration_Y: float
    Acceleration_Z: float
    Gyroscope_X: float
    Gyroscope_Y: float
    Gyroscope_Z: float
    Yaw_Rate: float

class LRPoint(BaseModel):
    X: float; Y: float; Z: float
    Intensity: float
    Azimuth: float
    Elevation: float
    Range: float
    Speed: float
    RCS: float
    Power: float
    Noise: float

class PredictUnified(BaseModel):
    model: Literal["egomotion","lidar_radar","both"]
    egomotion_data: Optional[List[List[float]]] = Field(None)
    lidar_radar_data: Optional[List[List[float]]] = Field(None)

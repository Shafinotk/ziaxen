from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import threading
from typing import List
from agent.fusion_agent import IntrusionDetectionAgent
from generator.realtime_generator import RealTimeDataGenerator
from agent.fusion_agent import IntrusionDetectionAgent  # Add this for prediction

app = FastAPI()

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Use specific origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = IntrusionDetectionAgent()
predictor = IntrusionDetectionAgent()  # Initialize prediction model
generator = RealTimeDataGenerator(hack_frequency=10)

class StreamResponse(BaseModel):
    gps_output: dict
    lidar_output: dict
    hacked: bool

class DrivingInput(BaseModel):
    Throttle_Position: float
    Brake_Pressure: float
    Steering_Angle: float
    Vehicle_Speed: float
    Gear_Position: float
    Latitude: float
    Longitude: float
    Speed: float
    Heading: float
    Acceleration_X: float
    Acceleration_Y: float
    Rotation_X: float
    Rotation_Y: float
    Rotation_Z: float
    Position_X: float
    Velocity_X: float
    Velocity_Y: float

results: List[StreamResponse] = []
streaming_thread = None
streaming_active = False

def handle_data(gps_df, lidar_df, is_hacked_label):
    report = agent.analyze(gps_df, lidar_df)
    if report:
        results.append(StreamResponse(
            gps_output=report["GPS Feature"],
            lidar_output=report["Lidar Feature"],
            hacked=report["Hacking Detected"]
        ))
        # Limit results list length to avoid memory bloat
        if len(results) > 1000:
            results.pop(0)

@app.get("/")
def root():
    return {"message": "Intrusion Detection API is running."}

@app.get("/start-stream")
def start_stream():
    global streaming_thread, streaming_active, results
    if streaming_active:
        return {"message": "Streaming is already running."}

    results = []
    streaming_active = True

    def run_stream():
        try:
            generator.stream_data(callback=handle_data, interval=1.0)
        except Exception as e:
            print(f"Streaming error: {e}")

    streaming_thread = threading.Thread(target=run_stream, daemon=True)
    streaming_thread.start()

    return {"message": "Streaming started."}

@app.get("/get-results", response_model=List[StreamResponse])
def get_results():
    return results

@app.post("/predict")
def predict(data: DrivingInput):
    input_dict = data.dict()
    prediction = predictor.analyze(gps_df, lidar_df)
    return {"prediction": prediction}

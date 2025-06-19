from agent.fusion_agent import IntrusionDetectionAgent
from generator.realtime_generator import RealTimeDataGenerator

def handle_data(gps_df, lidar_df, is_hacked_label):
    report = agent.analyze(gps_df, lidar_df)
    if report and report['Hacking Detected']:
        print("\n=== Intrusion Detected! ===")
        print(f"GPS Feature Output: {report['GPS Feature']}")
        print(f"Lidar Feature Output: {report['Lidar Feature']}")
    else:
        print(".", end="", flush=True)  # Keep alive signal

if __name__ == "__main__":
    agent = IntrusionDetectionAgent()
    generator = RealTimeDataGenerator(hack_frequency=10)
    print("Starting real-time monitoring... Press Ctrl+C to stop.")
    generator.stream_data(callback=handle_data, interval=1.0)

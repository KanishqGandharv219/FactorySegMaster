"""
FactorySegMaster Day 6: Train Custom PPE Model

Fine-tunes the base YOLOv8 nano model on a custom Hardhat & Vest dataset.
"""

from ultralytics import YOLO

def train_custom_model():
    print("Loading base YOLOv8n model...")
    # Load the base COCO model to start fine-tuning from
    model = YOLO("yolov8n.pt") 
    
    print("\nStarting Training Loop...")
    print("NOTE: Make sure you downloaded the dataset and extracted it.")
    print("Update the 'data' path below to point to your dataset's data.yaml file.")
    
    # We assume the user downloaded data to day6_ppe_training/datasets/dataset/data.yaml
    # Modify this path if your dataset extracts differently.
    data_yaml_path = "datasets/data.yaml" 
    
    # Train the model
    # epochs=10: Short run for demonstration. A real production model needs 50-100+
    # imgsz=640: Standard YOLO image size
    # device='cpu': Safe default, will use GPU if available and PyTorch is configured for CUDA
    try:
        results = model.train(
            data=data_yaml_path,
            epochs=10, 
            imgsz=640,
            plots=True, # Generate training curves
            name="ppe_detector" # The outputs will be saved to runs/detect/ppe_detector/
        )
        print("\nTraining Complete! 🎯")
        print("Your new custom weights are saved at: runs/detect/ppe_detector/weights/best.pt")
        
    except FileNotFoundError:
        print(f"\nERROR: Could not find dataset config at {data_yaml_path}")
        print("Please run python download_dataset.py first and ensure the data.yaml is in the correct folder.")
        
if __name__ == "__main__":
    train_custom_model()

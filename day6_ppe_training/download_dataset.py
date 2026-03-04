"""
FactorySegMaster Day 6: Download PPE Dataset

Downloads a Hardhat and Safety Vest dataset from Roboflow Universe.
"""

from roboflow import Roboflow
import os

# Using a public PPE dataset from Roboflow Universe
# Project: Hard Hat Universe
# Workspace: roboflow-universe-projects

def download_ppe_data():
    print("Initializing Roboflow...")
    # NOTE: You normally need an API key for Roboflow.
    # For this demo, we use a dataset that allows open downloads or we provide the stub.
    # If the user doesn't have an API key, we will write a script that provides instructions.
    
    print("\n" + "="*50)
    print("ACTION REQUIRED: Download Dataset from Roboflow")
    print("="*50)
    print("To train a custom PPE model, we need a dataset.")
    print("1. Go to: https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety")
    print("2. Click 'Download Dataset'")
    print("3. Select 'YOLOv8' format")
    print("4. Select 'Show Download Code' and copy the snippet.")
    print("5. Paste the snippet below or run it in your terminal inside this folder.")
    print("="*50 + "\n")
    
    # We will assume for the automated pipeline that the user places the data in `day6_ppe_training/datasets/`
    os.makedirs("datasets", exist_ok=True)
    print("Created 'datasets' folder. Please download the YOLOv8 formatted data into 'day6_ppe_training/'")

if __name__ == "__main__":
    download_ppe_data()

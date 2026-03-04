"""
FactorySegMaster Day 6: Generate Dummy PPE Dataset

Since the local network/firewall is blocking downloads from Roboflow,
this script generates a small, synthetic dataset of colored boxes
to simulate Hardhats (Yellow) and Safety Vests (Orange) so we can
verify the YOLOv8 fine-tuning pipeline actually runs.
"""

import cv2
import numpy as np
import os
import yaml

def generate_dummy_dataset():
    print("Generating synthetic PPE dataset...")
    
    # We will generate 20 training images and 5 validation images
    splits = {"train": 20, "val": 5}
    base_dir = "datasets"
    
    # Classes: 0 -> Hardhat, 1 -> Vest
    
    for split, count in splits.items():
        img_dir = os.path.join(base_dir, split, "images")
        lbl_dir = os.path.join(base_dir, split, "labels")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)
        
        for i in range(count):
            # Create a 640x640 gray background (simulating a factory wall)
            img = np.ones((640, 640, 3), dtype=np.uint8) * 128
            
            labels = []
            
            # --- Draw a Dummy Worker (Just a rectangle) ---
            worker_x, worker_y = np.random.randint(100, 300), np.random.randint(100, 300)
            cv2.rectangle(img, (worker_x, worker_y), (worker_x+200, worker_y+300), (200, 200, 200), -1)
            
            # --- Draw a Dummy Hardhat (Yellow rectangle on top) ---
            if np.random.rand() > 0.2: # 80% chance to have a hardhat
                hh_w, hh_h = 80, 40
                hh_x = worker_x + 60
                hh_y = worker_y - 20
                cv2.rectangle(img, (hh_x, hh_y), (hh_x+hh_w, hh_y+hh_h), (0, 255, 255), -1)
                
                # YOLO format: class x_center y_center width height (Normalized)
                xc = (hh_x + hh_w/2) / 640.0
                yc = (hh_y + hh_h/2) / 640.0
                w = hh_w / 640.0
                h = hh_h / 640.0
                labels.append(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
                
            # --- Draw a Dummy Vest (Orange rectangle on chest) ---
            if np.random.rand() > 0.3: # 70% chance to have a vest
                v_w, v_h = 100, 120
                v_x = worker_x + 50
                v_y = worker_y + 50
                cv2.rectangle(img, (v_x, v_y), (v_x+v_w, v_y+v_h), (0, 165, 255), -1) # BGR
                
                xc = (v_x + v_w/2) / 640.0
                yc = (v_y + v_h/2) / 640.0
                w = v_w / 640.0
                h = v_h / 640.0
                labels.append(f"1 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
                
            # Save image and label
            cv2.imwrite(os.path.join(img_dir, f"dummy_{i:03d}.jpg"), img)
            with open(os.path.join(lbl_dir, f"dummy_{i:03d}.txt"), "w") as f:
                f.write("\n".join(labels))
                
    # Create the data.yaml file YOLO needs with STRICT absolute paths for Windows
    yaml_content = {
        "path": os.path.abspath(base_dir).replace("\\", "/"),
        "train": os.path.abspath(os.path.join(base_dir, "train", "images")).replace("\\", "/"),
        "val": os.path.abspath(os.path.join(base_dir, "val", "images")).replace("\\", "/"),
        "nc": 2,
        "names": ["Hardhat", "Safety_Vest"]
    }
    
    with open(os.path.join(base_dir, "data.yaml"), "w") as f:
        yaml.dump(yaml_content, f)
        
    print(f"Dataset generated at {os.path.abspath(base_dir)}")
    print("Run `python train_ppe.py` to train on it!")

if __name__ == "__main__":
    generate_dummy_dataset()

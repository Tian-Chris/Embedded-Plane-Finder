from ultralytics import YOLO
import torch
import gc

def main():
    trained_model = YOLO("runs/yolov8_train/weights/last.pt")
    trained_model.train(
        data="data2.yaml",  
        epochs=100, 
        batch=16,  
        workers=0,   
        project="runs",
        name="yolov8_train2"
    )

if __name__ == "__main__":
    main()

from ultralytics import YOLO
import torch
import gc

def main():
    model = YOLO("runs/yolov8_train/weights/last.pt")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model.train(
        data="data.yaml",
        imgsz=416,
        epochs=100, 
        batch=16,
        workers=0,
        device=device,
        resume=True
    )

    model.val()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    trained_model = YOLO("runs/yolov8_train/weights/last.pt")
    trained_model.predict(
        source="test/images",
        conf=0.25,
        save=True 
    )

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

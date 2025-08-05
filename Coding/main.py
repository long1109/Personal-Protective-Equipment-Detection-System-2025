from ultralytics import YOLO

# Load a model
#model = YOLO('yolov8s.yaml')
model = YOLO('/content/gdrive/MyDrive/Train_Yolov8/Ketqua_ppe/last_org_yolov8s.pt')

# Use the model
results = model.train(data="data1.yaml", project='/content/gdrive/MyDrive/Train_Yolov8/Ketqua_ppe', pretrained= "/content/gdrive/MyDrive/Train_Yolov8/Ketqua_ppe/last_org_yolov8s.pt",imgsz=640 , batch = 32 ,epochs=100)  # train the model

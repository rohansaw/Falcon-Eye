from ultralytics import YOLO

img_sz = (2688,1512)
model = "/data/results/yolo/yolo/sw/weights/best.pt"

yolo = YOLO(model)
yolo.export(format="onnx", imgsz=img_sz)  # onnx for raspi
yolo.export(format="tflite", imgsz=img_sz, int8=True)  # tflite for raspi
yolo.export(format="edgetpu", imgsz=img_sz)  # edgetpu for google coral
from ultralytics import YOLO

# Load the exported TFLite model
tflite_model = YOLO('yolov8n_saved_model/yolov8n_float32.tflite')

# Run inference
results = tflite_model.predict('bus.jpg', show=True)
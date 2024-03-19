from ultralytics import YOLO

# Load the exported TFLite model
tflite_model = YOLO('C:/Users/gabri/OneDrive/Desktop/Universita/PMI/Weights/best_saved_model/best_float32.tflite')

# Run inference
results = tflite_model('https://ultralytics.com/images/bus.jpg')
import cv2
import time
import os
from ultralytics import YOLO

# --- Configuration ---
# Directory where your YOLO model files (.pt) are stored
YOLO_MODELS_DIR = "models" 
DEFAULT_MODEL_NAME = "models/yolov8n.pt"  # Default model if no selection is made
DEFAULT_MODEL_CONFIDENCE = 0.55  # Default minimum confidence score for a detection (0.0 to 1.0)
DEFAULT_RESOLUTION_WIDTH = 1280
DEFAULT_RESOLUTION_HEIGHT = 720

# Function to list available YOLO model files
def get_available_models(directory):
    models = []
    if not os.path.exists(directory):
        print(f"Warning: Directory '{directory}' not found. No local models available.")
        return models
    for filename in os.listdir(directory):
        if filename.endswith(".pt"):
            models.append(filename)
    return sorted(models)

# Get available models
available_models = get_available_models(YOLO_MODELS_DIR)

selected_model_path = ""
if not available_models:
    print(f"No YOLO model files found in '{YOLO_MODELS_DIR}'. Use default '{DEFAULT_MODEL_NAME}'.")
    selected_model_path = DEFAULT_MODEL_NAME
else:
    print(f"\nAvailable YOLO models in '{YOLO_MODELS_DIR}':")
    for i, model_file in enumerate(available_models):
        print(f"  [{i+1}] {model_file}")
    print(f"  [Default] (Press Enter) {DEFAULT_MODEL_NAME}")

    while True:
        choice = input("Enter the number of the model to use press Enter for default: ").strip()
        if not choice:
            selected_model_path = DEFAULT_MODEL_NAME
            print(f"Using default model: {DEFAULT_MODEL_NAME}") # Corrected variable name here
            break
        try:
            choice_index = int(choice) - 1
            if 0 <= choice_index < len(available_models):
                selected_model_path = os.path.join(YOLO_MODELS_DIR, available_models[choice_index])
                print(f"Using selected model: {selected_model_path}")
                break
            else:
                print("Invalid choice. Please enter a valid number or press Enter.")
        except ValueError:
            print("Invalid input. Please enter a number or press Enter.")

# Prompt for Model Confidence
model_confidence = DEFAULT_MODEL_CONFIDENCE
while True:
    conf_input = input(f"Enter model confidence (0.0-1.0, default {DEFAULT_MODEL_CONFIDENCE}): ").strip()
    if not conf_input:
        model_confidence = DEFAULT_MODEL_CONFIDENCE
        print(f"Using default confidence : {DEFAULT_MODEL_CONFIDENCE}")
        break
    try:
        parsed_conf = float(conf_input)
        if 0.0 <= parsed_conf <= 1.0:
            model_confidence = parsed_conf
            print(f"Using confidence: {model_confidence}")
            break
        else:
            print("Invalid confidence. Please enter a value between 0.0 and 1.0.")
    except ValueError:
        print("Invalid input. Please enter a correct value.")

# Prompt for Screen Resolution
resolution_width = DEFAULT_RESOLUTION_WIDTH
resolution_height = DEFAULT_RESOLUTION_HEIGHT
while True:
    res_input = input(f"Enter screen resolution (default {DEFAULT_RESOLUTION_WIDTH}x{DEFAULT_RESOLUTION_HEIGHT}): ").strip().lower()
    if not res_input:
        print(f"Using default resolution: {resolution_width}x{resolution_height}")
        break
    try:
        parts = res_input.split('x')
        if len(parts) == 2:
            width = int(parts[0])
            height = int(parts[1])
            if width > 0 and height > 0:
                resolution_width = width
                resolution_height = height
                print(f"Using resolution: {resolution_width}x{resolution_height}")
                break
            else:
                print("Resolution dimensions must be positive integers.")
        else:
            print("Invalid format. Please use WxH (e.g., 1920x1080).")
    except ValueError:
        print("Invalid input. Please enter numerical values for resolution.")


# Load the YOLO model
try:
    model = YOLO(selected_model_path)
except FileNotFoundError:
    print(f"Error: Model file '{selected_model_path}' not found. Please ensure it exists.")
    exit()
except Exception as e:
    print(f"An error occurred while loading the model: {e}")
    exit()

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Set the desired resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution_height)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

print(f"\nCamera resolution set to: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
print(f"Confidence threshold set to: {model_confidence}")

# Initialize variables for FPS calculation
prev_frame_time = 0
new_frame_time = 0

while True:
    # Read a frame from the webcam
    success, frame = cap.read()

    if not success:
        print("Failed to read frame from camera.")
        break

    # Calculate FPS
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    
    # Run YOLO detection on the frame with specified confidence threshold
    # The 'conf' argument filters out detections below the specified value
    results = model(frame, conf=model_confidence, stream=True)

    annotated_frame = frame.copy() # Start with a clean copy for annotations
    
    # Process and display the results
    for r in results:
        # Use Ultralytics' 'plot()' method for drawing boxes and labels
        annotated_frame = r.plot()
    
    # Add FPS text to the annotated frame
    fps_text = f"FPS: {fps:.2f}" # Format FPS to 2 decimal places
    cv2.putText(annotated_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show the annotated frame
    cv2.imshow("YOLO Real-Time Detection", annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and destroy all windows
cap.release()
cv2.destroyAllWindows()

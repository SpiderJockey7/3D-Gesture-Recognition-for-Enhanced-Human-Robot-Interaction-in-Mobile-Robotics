import cv2
import torch
import torchvision
import torch.nn as nn
from PIL import Image
import mediapipe as mp
from torchvision import transforms
import sys
import os

# Get the device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Get model file name , if not set as default
model_name = sys.argv[1] if len(sys.argv) > 1 else "resnet18_model.pth"
if not os.path.exists(model_name):
    print(f"Error: Model file '{model_name}' not found.")
    sys.exit(1)

# Initialize MediaPipe hand detection model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,  
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.5)

# Initialize ResNet-18 model for gesture classification
model = torchvision.models.resnet18(weights='DEFAULT')
model.fc = nn.Linear(model.fc.in_features, 8)  # Adjust for 8 classes

# Load model weights
try:
    model.load_state_dict(torch.load(model_name, map_location=device))  # Load model directly to the correct device
    model.to(device)  # Ensure the model is on the correct device
    model.eval()
    print(f"Model '{model_name}' successfully loaded.")
except Exception as e:
    print(f"Error loading model '{model_name}': {e}")
    sys.exit(1)

# Define preprocessing steps
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Gesture label mapping
gesture_labels = ['STOP', 'Forward', 'BACK', 'LEFT', 'RIGHT', 'GRAB', 'action_ONE', 'action_TWO']

def recognize_gesture():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Flip frame for a mirror view
        frame = cv2.flip(frame, 1)

        # Convert BGR frame to RGB for MediaPipe processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)  # Hand detection

        if results.multi_hand_landmarks:  # If hands are detected
            for hand_landmarks in results.multi_hand_landmarks:
                # Get bounding box around the hand
                h, w, c = frame.shape
                x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * w)
                y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * h)
                x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * w)
                y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * h)

                # Expand bounding box margins
                margin = 20
                x_min = max(0, x_min - margin)
                y_min = max(0, y_min - margin)
                x_max = min(w, x_max + margin)
                y_max = min(h, y_max + margin)

                # Crop the hand region from the frame
                hand_img = frame[y_min:y_max, x_min:x_max]

                # Check if hand image is valid
                if hand_img.shape[0] == 0 or hand_img.shape[1] == 0:
                    continue

                # Convert cropped hand image to PIL format
                hand_img_pil = Image.fromarray(cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB))

                # Preprocess the hand image
                img_tensor = preprocess(hand_img_pil).unsqueeze(0).to(device)  # Move tensor to the correct device

                # Predict gesture with the model
                with torch.no_grad():
                    outputs = model(img_tensor)
                    _, predicted_class = torch.max(outputs, 1)
                    predicted_class = predicted_class.item()

                # Get the predicted gesture name
                predicted_class_name = gesture_labels[predicted_class]

                # Display prediction on the frame
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(frame, f"Gesture: {predicted_class_name}", 
                            (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the frame with annotations
        cv2.imshow('Hand Gesture Recognition', frame)

        # Press ESC to exit
        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

recognize_gesture()

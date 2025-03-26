import cv2
import pandas as pd
import torch
from torchvision import transforms
from timm import create_model
import pyttsx3

# Load the trained model and database CSV file
model = create_model('vit_base_patch16_224', pretrained=False, num_classes=512).cuda()
model.load_state_dict(torch.load('face_recognition_model.pth'))
model.eval()

df = pd.read_csv('face_database.csv')
known_face_names = df['name'].tolist()
known_face_branches = df['branch'].tolist()
known_face_images = df['image_path'].tolist()

# Initialize text-to-speech engine 
engine = pyttsx3.init()

# Image transformation for input to the model 
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def speak(text):
    engine.say(text)
    engine.runAndWait()

def recognize_face(frame):
    image_tensor = transform(frame).unsqueeze(0).cuda()  # Add batch dimension and move to GPU
    
    with torch.no_grad():
        output = model(image_tensor)
    
    # Get predicted class index (highest probability)
    _, predicted_idx = torch.max(output.data, 1)
    
    return known_face_names[predicted_idx.item()], known_face_branches[predicted_idx.item()]

# Start video capture 
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    
    if not ret:
        break
    
    name, branch = recognize_face(frame)  # Recognize face
    
    if name == "unknown person":
        speak("Unknown person detected")
    else:
        speak(f"Detected {name} from {branch}")
    
    cv2.imshow('Video', frame)  # Display video feed
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

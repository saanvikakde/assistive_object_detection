import cv2 
import torch 
import pyttsx3
from tkinter import *
import speech_recognition as sr
import threading 
import time
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

# Suppress specific warnings, e.g., DeprecationWarning
warnings.filterwarnings("ignore", category=DeprecationWarning)

engine = pyttsx3.init() 
recognizer = sr.Recognizer()
model = torch.hub.load('ultralytics/yolov5', 'yolov5n') 
cam = cv2.VideoCapture(0)

objects = [] 

def listen_for_command(): 
    print("Listening for command...")
    with sr.Microphone() as source: 
        recognizer.adjust_for_ambient_noise(source, duration=1) 
        try: 
            audio = recognizer.listen(source) 
            command = recognizer.recognize_google(audio).lower() 
            print(f"Command recognized: {command}")
            return command
        except sr.UnknownValueError:
            print("Could not understand audio")
            return None
        except sr.RequestError: 
            print("Request error with the speech recognition service") 
            return None

def speak_objects(objects):
    global objects_spoken
    if objects and not objects_spoken:
        engine.say("Detecting nearby surroundings")
        for obj in objects:
            engine.say(f"Detected: {obj}")
            print(f"Detected: {obj}")
        engine.runAndWait()
        objects_spoken = True  # Set the flag to indicate objects have been spoken

def listen_in_background(): 
    global objects_spoken
    print("Background listening thread started.")
    while True:
        command = listen_for_command()
        if command == "start":
            print("Start command received, resetting spoken flag.")
            objects_spoken = False  # Reset the flag when "start" command is received
    
listening_thread = threading.Thread(target=listen_in_background, daemon=True)
listening_thread.start()
while True: #loop
    ret, frame = cam.read()
    if not ret: #no frame was captured 
        break

    results = model(frame)  # Get results
    detected_objects = results.names  # Extract object labels
    predictions = results.pred[0]  # Predicted bounding boxes

    for pred in predictions:
        if pred[4] > 0.3:
            object_label = detected_objects[int(pred[5])]
            objects.append(object_label)
    
    renderedframes = results.render()[0]

    # Display the resulting frame
    cv2.imshow('YOLOv5 Assistive Object Detection', renderedframes)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Quitting...")
        break

    if cv2.waitKey(1) & 0xFF == ord('d'):
        print("Manual detection command ('d') received.")
        speak_objects(objects)


# Release the capture and close OpenCV windows
cam.release()
cv2.destroyAllWindows()


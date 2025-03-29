import warnings
warnings.simplefilter("ignore", category=FutureWarning)
import torch
import cv2
import numpy as np
import os
import RPi.GPIO as GPIO
import time
import sys
import threading

# Load YOLOv5 model
model_path = "/home/pi/yolov5/best_merged.pt"
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)

# Set device (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()


# GPIO Pin Assignments
TRIG = 5  # Trigger Pin of Ultrasonic Sensor
ECHO = 6  # Echo Pin of Ultrasonic Sensor
IN1 = 17    # Motor 1 Forward
IN2 = 27    # Motor 1 Backward
IN3 = 22   # Motor 2 Forward
IN4 = 23   # Motor 2 Backward
ENA = 18   # PWM Speed Control Motor 1
ENB = 19   # PWM Speed Control Motor 2



# Setup GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)
GPIO.setup(IN1, GPIO.OUT)
GPIO.setup(IN2, GPIO.OUT)
GPIO.setup(IN3, GPIO.OUT)
GPIO.setup(IN4, GPIO.OUT)
GPIO.setup(ENA, GPIO.OUT)
GPIO.setup(ENB, GPIO.OUT)

# PWM for motor speed
pwm_a = GPIO.PWM(ENA, 50)  # 50 Hz frequency
pwm_b = GPIO.PWM(ENB, 50)
pwm_a.start(40)  # Set initial speed (0-100)
pwm_b.start(40)

def play_audio(file_path):
    """Plays the specified .wav audio file."""
    os.system(f"aplay {file_path}")  # Uses aplay (default on Raspberry Pi)


class VideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        if not self.stream.isOpened():
            print("Error: Could not open webcam.")
            sys.exit()
        self.ret, self.frame = self.stream.read()
        self.stopped = False
        self.lock = threading.Lock()

    def start(self):
        threading.Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            ret, frame = self.stream.read()
            if not ret:
                self.stop()
                return
            with self.lock:
                self.ret, self.frame = ret, frame

    def read(self):
        with self.lock:
            return self.ret, self.frame.copy()

    def stop(self):
        self.stopped = True
        self.stream.release()

# Start video stream in a separate thread
video_stream = VideoStream().start()
time.sleep(0.5)

def get_distance():
    GPIO.output(TRIG, True)
    time.sleep(0.00001)  # 10µs pulse
    GPIO.output(TRIG, False)

    start_time = time.time()
    stop_time = time.time()

    while GPIO.input(ECHO) == 0:
        start_time = time.time()
    while GPIO.input(ECHO) == 1:
        stop_time = time.time()

    elapsed_time = stop_time - start_time
    distance = (elapsed_time * 34300) / 2  # Speed of sound = 343m/s
    return distance

# Movement functions

#def obstacle_right():,,,,,,,,,,,,,,,,,,
def move_forward():
    GPIO.output([IN1, IN3], GPIO.LOW)
    GPIO.output([IN2, IN4], GPIO.HIGH)
    pwm_a.ChangeDutyCycle(40)
    pwm_b.ChangeDutyCycle(45)

def move_backward():
    GPIO.output([IN1, IN3], GPIO.HIGH)
    GPIO.output([IN2, IN4], GPIO.LOW)

def turn_left():
    GPIO.output([IN1, IN3], GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN4, GPIO.HIGH)
    pwm_a.ChangeDutyCycle(15)
    pwm_b.ChangeDutyCycle(75)

def turn_right():
    GPIO.output([IN1, IN3], GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN4, GPIO.HIGH)
    pwm_a.ChangeDutyCycle(60)
    pwm_b.ChangeDutyCycle(25)

def stop():
    GPIO.output([IN1, IN2, IN3, IN4], GPIO.LOW)
    
def slight_right():
    GPIO.output([IN1, IN3], GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN4, GPIO.HIGH)
    pwm_a.ChangeDutyCycle(65)
    pwm_b.ChangeDutyCycle(0)
    
def avoid_left():
    GPIO.output([IN1, IN3], GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN4, GPIO.HIGH)
    pwm_a.ChangeDutyCycle(10)
    pwm_b.ChangeDutyCycle(80)

no_chair_start_time = None
TIMEOUT = 60

while True:
    # Read a frame from the webcam
    ret, frame = video_stream.read()
    if not ret:
        print("Error: Failed to capture image.")
        break
    # Convert frame to RGB (YOLO expects RGB images)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform inference
    results = model(img)

    # Extract detections
    detections = results.xyxy[0].cpu().numpy()  # Bounding boxes format: (x1, y1, x2, y2, confidence, class)

    # Get frame width and divide into three sections
    frame_width = frame.shape[1]
    left_boundary = frame_width // 3
    right_boundary = 2 * (frame_width // 3)

    detected_objects = []  # Store detected objects and their positions
    most_confident_chair = None  # To store the most confident chair
    max_conf = 0  # Track highest confidence
    # Draw detections and determine position
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        object_name = model.names[int(cls)]
    
        if object_name.lower() == "unoccupied" and conf > max_conf:
            max_conf = conf
            most_confident_chair = det
    
    if most_confident_chair is not None and max_conf > 0.40:
        no_chair_start_time = None
        
        x1, y1, x2, y2, conf, cls = most_confident_chair
        center_x = (x1 + x2) / 2
        bbox_width = x2 - x1  # Get width for the most confident chair


        # Determine object position
        if center_x < left_boundary:
            position = "Left"
        elif center_x > right_boundary:
            position = "Right"
        else:
            position = "Middle"

        # Store detected object and position
        detected_objects.append((object_name, position, conf))

        # Draw bounding box and label
        label = f"{object_name} {conf:.2f} ({position})"
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        distance = get_distance()
        print(f"Distance: {distance:.2f} cm")
        print(f"most confident width: {bbox_width:.2f} cm")
        print(f"FWidth: {frame_width:.2f} cm")
        
        if bbox_width > 0.5 * frame_width:
            print("Object is close enough, stopping robot.")
            print(f"chair is on your {position}")
            play_audio("reached.wav")
            image_path = "detected_chair.jpg"
            cv2.imwrite(image_path, frame)
            print(f"Image saved as {image_path}")
            stop()
            video_stream.stop()
            cv2.destroyAllWindows()
            sys.exit()
        if distance >= 20:
            if position == "Left":
                print("Turning Left")
                turn_left()
            elif position == "Right":
                print("Turning Right")
                turn_right()
            elif position == "Middle":
                print("Moving Forward")
                move_forward()
          # Exit the 10-second detection loop if a chair is found
            
        else:
            print("Obstacle detected on the path, taking a left")
            stop()
            play_audio("Obstacle.wav")
            time.sleep(4)
            while distance < 20:
                avoid_left()
                time.sleep(1)  # Small delay to prevent excessive turning
                distance = get_distance()  # Update distance after turning
                
            avoid_left()
            time.sleep(2)
            move_forward()
            time.sleep(1)
            stop()
            time.sleep(0.2)
            print("path cleared and continuing the movement")
    else:
        if no_chair_start_time is None:
            no_chair_start_time = time.time()
            
        elapsed_time = time.time() - no_chair_start_time
        if elapsed_time >= TIMEOUT:
#             print("No chair detected for 1 minute. Stopping rotation and detecting the door")
#             play_audio("door-navigate.wav")
            print("No chair detected stopping the rotation")
            play_audio("nodetect.py")
            stop()
            time.sleep(2)
            stop()
            video_stream.stop()
            cv2.destroyAllWindows()
            sys.exit()
            #os.system("Python3 navigate_door.py")
            #break
        slight_right()
        print("No chair detected taking slight right")
        time.sleep(0.1)

    
    # Show the output stream
    cv2.imshow("YOLO Object Detection", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        stop_robot()
        break

# Release resources
video_stream.stop()
cv2.destroyAllWindows()

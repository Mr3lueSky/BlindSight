import RPi.GPIO as GPIO
import os
import time

# Define switch pin
SWITCH_PIN = 4  # Change to the GPIO pin where your switch is connected

# Setup GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(SWITCH_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)  # Pull-up resistor

# File paths
SCRIPT_ON = "/home/pi/yolov5/video-text.py"  # OCR-to-speech
SCRIPT_OFF = "/home/pi/yolov5/run_audio.py"  # Voice recognition and navigation

# Function to start the script
def run_script(script):
    os.system(f"python3 {script}")

# Main loop to check switch state
try:
    current_state = None
    while True:
        switch_state = GPIO.input(SWITCH_PIN)

        if switch_state != current_state:  # Detect state change
            current_state = switch_state
            if switch_state == GPIO.HIGH:  # Switch is ON
                print("Switch ON: Running OCR-to-Speech script...")
                run_script(SCRIPT_ON)
            else:  # Switch is OFF
                print("Switch OFF: Running Voice Recognition script...")
                run_script(SCRIPT_OFF)

        time.sleep(0.5)  # Debounce delay

except KeyboardInterrupt:
    print("Exiting...")

finally:
    GPIO.cleanup()
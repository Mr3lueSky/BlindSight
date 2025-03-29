import cv2
import pytesseract
import time
from gtts import gTTS
import os

# Ensure pytesseract points to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

def extract_text_from_video(output_file, duration=10, capture_at_second=5):
    # Open the USB camera
    cap = cv2.VideoCapture(0)  # 0 means the first camera, use 1 or 2 for other connected cameras

    # Check if the camera is opened successfully
    if not cap.isOpened():
        print("Error: Couldn't open camera.")
        return

    # Open the text file to save the output
    with open(output_file, 'w') as file:
        start_time = time.time()

        while True:
            ret, frame = cap.read()  # Read a frame from the camera
            if not ret:
                print("Error: Couldn't fetch frame.")
                break

            # Get the current elapsed time
            elapsed_time = time.time() - start_time

            # Capture only at the 5th second
            if elapsed_time >= capture_at_second:
                # Convert frame to grayscale (Tesseract works better on grayscale images)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Use Tesseract to extract text from the frame
                text = pytesseract.image_to_string(gray)

                # Write the extracted text to the file (if not empty)
                if text.strip():
                    file.write(text)
                    file.write("\n")  # Add a new line for readability

                print("Text extracted and saved to", output_file)
                break  # Exit after capturing and processing the 5th second frame

            # Optional: Display the frame for debugging
            cv2.imshow('Frame', frame)

            # Wait for 1 ms, break on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the video capture and close any OpenCV windows
        cap.release()
        cv2.destroyAllWindows()

def text_to_speech(input_file, output_audio="output.mp3"):
    # Read text from file
    with open(input_file, "r") as file:
        text = file.read().strip()

    if text:
        # Convert text to speech
        tts = gTTS(text, lang="en")
        tts.save(output_audio)

        # Play the generated speech
        os.system(f"mpg321 {output_audio}")  # Requires mpg321 to be installed
        print("Speech generated and played.")
    else:
        print("No text found to convert.")

# Example usage
output_file = "output_text.txt"  # Path to save the extracted text
extract_text_from_video(output_file)
text_to_speech(output_file)
# BlindSight
- BlindSight is a chained robot that is the first prototype of our vision in building something that will reduce the dependence of the visually impaired on other human beings for their need
- The current prototype has 2 modes:
  Mode 1: In this mode the robot will look for a chair for the blind person to sit on the request of the blind person. We used YOLOv5 to train a model that can identify occupied and unoccupied chairs. This model is used to identify the unoccupied chairs and once detected the robot gives a voice feedback and moves towards the chair.
  Mode 2: In this mode the robot uses OCR to read texts that are present in front of it.

## Technologies Used
- YOLOv5
- Python
- Google Text To Speech
- Tesseract OCR

## Hardware Used
- Rapsberry pi as the main processing unit.
- Motors and motor controller to facilitate movement.
- Mic and Speaker to take in audio input and give out voice feedback.
- Webcam to detect chair and read text.

## Output
![Output](/detected_chair.jpg)*Identified chair reached successfully!*

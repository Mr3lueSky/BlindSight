import speech_recognition as sr
import os
import subprocess

def play_audio(file_path):
    """Plays the specified .wav audio file."""
    os.system(f"aplay {file_path}")  # Uses aplay (default on Raspberry Pi)

recognizer = sr.Recognizer()
audio_file = "hello.wav"

with sr.AudioFile(audio_file) as source:
    audio = recognizer.record(source)

try:
    text = recognizer.recognize_google(audio)
    print("Transcribed Text:", text)

    # Save transcribed text to file (overwrite previous data)
    with open("result.txt", "w") as file:
        file.write(text)
        file.write("\n")  # Add newline for readability

    # Compare transcribed text with target phrase
    if "chair" in text.lower():
        print("Match found! Playing 'searching.wav'...")
        play_audio("searching.wav")
        subprocess.run(['python3', 'object-find.py'])
    else:
        print("No match found")

except sr.UnknownValueError:
    print("Google Speech Recognition could not understand the audio")
except sr.RequestError:
    print("Could not request results from Google Speech Recognition")

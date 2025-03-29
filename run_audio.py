import pyaudio
import wave
import subprocess

def get_device_index():
    """Find the device index for the USB microphone (Card 3)."""
    p = pyaudio.PyAudio()
    device_index = None

    print("Available audio input devices:")
    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        print(f"Device {i}: {dev['name']} ({dev['maxInputChannels']} channels)")
        
        # Check if the device matches the known name (modify if needed)
        if "USB PnP Sound Device" in dev['name'] and dev['maxInputChannels'] > 0:
            device_index = i
            print(f"Selected Device Index: {device_index}")
            break  # Stop searching after finding the correct device

    p.terminate()

    if device_index is None:
        raise ValueError("USB Microphone (Card 3) not found! Check `arecord -l` output.")

    return device_index


def record_audio(duration, output_file):
    """Records audio from the USB microphone and saves it as a WAV file."""
    p = pyaudio.PyAudio()
    device_index = get_device_index()

    rate = 44100  # CD quality
    channels = 1  # Mono audio
    frames_per_buffer = 1024  # Buffer size

    # Open the stream with the selected device index
    stream = p.open(format=pyaudio.paInt16, channels=channels, rate=rate,
                    input=True, frames_per_buffer=frames_per_buffer,
                    input_device_index=device_index)

    print(f"Recording for {duration} seconds...")

    frames = []
    for _ in range(0, int(rate / frames_per_buffer * duration)):
        data = stream.read(frames_per_buffer, exception_on_overflow=False)
        frames.append(data)

    print("Recording finished.")

    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save the recorded audio
    with wave.open(output_file, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))

    print(f"Audio saved to {output_file}")


# Record audio from USB microphone (Card 3) and save as hello.wav
record_audio(10, 'hello.wav')

# Call the conversion.py script
subprocess.run(['python3', 'conversion.py'])
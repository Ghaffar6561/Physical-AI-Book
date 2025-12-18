"""
This file provides a code example for converting speech to text, a crucial
first step in a voice-to-action pipeline.

Learning Goals:
- Understand how to capture microphone input.
- Use a speech recognition library to transcribe audio to text.
- See the conceptual place of this module in a larger robotics system.

Dependencies:
- speech_recognition: `pip install SpeechRecognition`
- pyaudio: `pip install PyAudio` (or use your system's package manager)

For an alternative using a more powerful local model, you could use OpenAI's
Whisper. That would require `pip install openai-whisper`.
"""

import speech_recognition as sr

def listen_and_transcribe(recognizer, microphone):
    """
    Listens for a single utterance from the microphone and transcribes it.

    Args:
        recognizer: An instance of sr.Recognizer.
        microphone: An instance of sr.Microphone.

    Returns:
        The transcribed text as a string, or None if it fails.
    """
    with microphone as source:
        print("Calibrating for ambient noise...")
        recognizer.adjust_for_ambient_noise(source)
        print("Listening for your command...")
        audio = recognizer.listen(source)

    try:
        print("Transcribing audio...")
        # Using Google's free web speech API
        text = recognizer.recognize_google(audio)
        print(f"User said: '{text}'")
        return text
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio.")
        return None
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return None

def main():
    """
    Main function to set up the recognizer and microphone and start listening.
    """
    recognizer = sr.Recognizer()
    # You might need to select a different device index for the microphone
    microphone = sr.Microphone()

    # In a real robotics application, this would be a loop in a ROS 2 node,
    # publishing the transcribed text to a topic.
    
    command = listen_and_transcribe(recognizer, microphone)

    if command:
        # In a real system, you would publish this command to the next
        # stage of the VLA pipeline (the LLM planner).
        print(f"\nCommand to be sent to planner: '{command}'")

    # Example of how you might use Whisper instead:
    # -----------------------------------------------
    # import whisper
    # model = whisper.load_model("base")
    # with microphone as source:
    #     print("Listening...")
    #     audio = recognizer.listen(source)
    # with open("command.wav", "wb") as f:
    #     f.write(audio.get_wav_data())
    # result = model.transcribe("command.wav")
    # print(f"Whisper transcription: {result['text']}")
    # -----------------------------------------------


if __name__ == '__main__':
    main()

import speech_recognition as sr

recognizer = sr.Recognizer()
mic = sr.Microphone()

def listen_once(timeout: int = 5) -> str:
    """Listen once from the microphone and return recognized speech (lowercase)."""
    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        print("Listening...")
        try:
            audio = recognizer.listen(source, timeout=timeout)
            text = recognizer.recognize_google(audio)
            print(f"🗣️ You said: {text}")
            return text.lower()
        except sr.WaitTimeoutError:
            return ""
        except sr.UnknownValueError:
            return ""
        except sr.RequestError as e:
            print("Speech recognition error:", e)
            return ""

if __name__ == "__main__":
    print("🎤 Testing speech recognition... (say something!)")
    text = listen_once(timeout=7)

    if text:
        print(f"Recognized: {text}")
    else:
        print(" No speech detected or recognition failed.")
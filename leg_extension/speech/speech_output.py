import pyttsx3

def speak(text: str):
    """Speak text aloud and print it using Microsoft Guy voice."""
    if not text:
        return

    print(f"🤖 Therapist: {text}")
    engine = pyttsx3.init()  # recreate engine each call
    engine.setProperty('rate', 180)

    # List available voices
    voices = engine.getProperty('voices')

    # Try to select 'Microsoft Guy' voice (case-insensitive)
    for v in voices:
        if "guy" in v.name.lower():
            engine.setProperty('voice', v.id)
            break

    engine.say(text)
    engine.runAndWait()
    engine.stop()

if __name__ == "__main__":
    print("🔊 Testing text-to-speech...")
    speak("Hello! This is your friendly physical therapist speaking. Let's make sure your voice output works properly.")
    speak("If you can hear me clearly, then text to speech is working perfectly!")

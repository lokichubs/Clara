import cv2
import threading
import time
from vision.leg_extension_detector import LegExtensionDetector
from speech.speech_input import listen_once
from speech.speech_output import speak
from agent.therapist_agent import TherapistAgent

# Global flag to safely stop background listener
stop_listening = False


def voice_listener(agent, detector):
    """Continuously listens for user voice input while camera runs."""
    global stop_listening
    while not stop_listening:
        user_text = listen_once()
        if not user_text:
            continue

        user_text_lower = user_text.lower()

        # Allow user to manually stop the session
        if any(word in user_text_lower for word in ["stop", "end", "quit", "finish"]):
            speak("Alright, ending the session early. Great work today!")
            stop_listening = True
            break

        reps = detector.reps
        angle = getattr(detector, "last_angle", 0)
        response = agent.get_response(user_text, angle, reps)
        speak(response)
        # time.sleep(0.1)  # avoid overloading the mic loop


def main():
    global stop_listening
    agent = TherapistAgent()
    detector = LegExtensionDetector()

    # Intro conversation
    speak(agent.get_intro())
    while True:
        user_reply = listen_once()
        if "yes" in user_reply.lower():
            speak("Great! Let’s start your leg extensions. Try to complete 10 reps!")
            break
        elif user_reply:
            speak("No problem, let me know when you’re ready.")
        else:
            speak("I didn’t catch that. Are you ready to begin?")

    # Small pause before starting video (lets TTS finish)
    time.sleep(0.5)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        speak("I can’t access your camera. Please check it and try again.")
        return

    cv2.namedWindow('Leg Extension Detector', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Leg Extension Detector', 980, 720)

    # Start background listener
    threading.Thread(target=voice_listener, args=(agent, detector), daemon=True).start()

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Failed to grab frame.")
                continue

            frame, reps, angle = detector.process_frame(frame)
            detector.last_angle = angle  # store latest angle for thread access
            cv2.imshow('Leg Extension Detector', frame)
            cv2.setWindowProperty('Leg Extension Detector', cv2.WND_PROP_TOPMOST, 1)

            # Stop automatically after 10 reps
            if reps >= 10:
                speak("Good job! Congratulations, you’ve completed your set!")
                stop_listening = True
                break

            # Allow manual quit with 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                speak("Session ended manually. Great effort today!")
                stop_listening = True
                break

            # Early voice stop check
            if stop_listening:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        speak("Session ended. Well done today!")


if __name__ == "__main__":
    main()

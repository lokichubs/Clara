import requests
import json

class TherapistAgent:
    """
    TherapistAgent using local LLaMA-3 through the Ollama API.
    Default endpoint: http://127.0.0.1:11434/api/generate
    """

    def __init__(self, host: str = "http://127.0.0.1:11434"):
        # Ensure we store a clean base URL (no trailing slash)
        self.api_base = host.rstrip("/")

    def _call_model(self, prompt: str, model: str = "llama3", max_tokens: int = 256):
        """
        Internal helper to call the local Ollama API.
        The 'stream': False flag ensures we get a complete JSON response
        instead of line-by-line streaming output.
        """
        url = f"{self.api_base}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "stream": False
        }

        try:
            resp = requests.post(url, json=payload, timeout=45)
            resp.raise_for_status()
            data = resp.json()

            # Handle common Ollama response formats
            if isinstance(data, dict) and "response" in data:
                return data["response"]
            if "text" in data:
                return data["text"]
            if "results" in data and len(data["results"]) > 0:
                return data["results"][0].get("text") or data["results"][0].get("content")

            # Fallback: return raw JSON for debugging
            return json.dumps(data)

        except Exception as e:
            print("[TherapistAgent] Model call failed:", e)
            return "Sorry, I’m having trouble thinking right now."

    def get_intro(self) -> str:
        """
        Ask the model to greet the user and prompt them to begin.
        """
        prompt = (
            "You are a friendly physical therapist named Clara. tell the user to be seated comfortably."
            "Ask them if they are ready to begin their leg extension exercise."
            "End on the question asking them if they are ready, keep it short"
        )
        return self._call_model(prompt)

    def get_response(self, user_text: str, knee_angle: float, reps: int) -> str:
        """
        Generate a therapist-style response given the current knee angle,
        rep count, and what the user said (e.g. 'it hurts'). The goal is to get to 10 reps
        """
        prompt = (
            f"You are a supportive physical therapist guiding a patient doing seated leg extensions.\n"
            f"Knee angle: {knee_angle:.1f} degrees\n"
            f"Reps completed: {reps}\n"
            f"User said: \"{user_text}\"\n\n"
            "Respond in one short, spoken-style sentence that is encouraging, safe, and easy to understand."
            "If they say they are in pain or equivalent, advise them to adjust their form rather than pause."
            "For example if their knee hurts, suggest they slow down or ensure their knee is facing up."
        )
        return self._call_model(prompt)
    
if __name__ == "__main__":
    agent = TherapistAgent()

    print("Testing TherapistAgent with local LLaMA 3...")
    
    # Test 1: Intro message
    intro = agent.get_intro()
    print("\n[Intro Message]\n", intro)

    # Test 2: Example response
    response = agent.get_response("my knee hurts a little, what should I do?", knee_angle=130, reps=5)
    print("\n[Model Response]\n", response)
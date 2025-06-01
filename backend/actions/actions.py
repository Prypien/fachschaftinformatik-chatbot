from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
import requests
import json

OLLAMA_URL = "http://localhost:11434"

class ActionSmartResponse(Action):
    def name(self) -> Text:
        return "action_smart_response"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:
        user_input = tracker.latest_message.get("text")

        try:
            response = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": "phi3:mini",
                    "prompt": (
                        "Du bist ein smarter, hilfsbereiter Chatbot der Fachschaft Informatik der Hochschule Reutlingen. Deine Aufgabe ist es, den utter_default zu ersetzen das heißt du kommst erst zum Einsatz wenn es zum default kommt. Antworte lustig darauf und dann poste unseren Linktree: https://linktr.ee/inf.fachschaft\n\n "
                        f"Frage: {user_input}"
                    ),
                    "stream": False
                },
                stream=True
            )

            chunks = []
            for line in response.iter_lines():
                if line:
                    data = json.loads(line.decode("utf-8"))
                    chunks.append(data.get("response", ""))

            reply = "".join(chunks).strip()
            if reply:
                # Füge den Linktree-Hinweis immer ans Ende der Antwort, wenn noch nicht enthalten
                if "linktr.ee/inf.fachschaft" not in reply:
                    reply = f"{reply}\n\nWeitere Infos findest du auch hier: https://linktr.ee/inf.fachschaft"
            else:
                reply = "Ich konnte leider keine passende Antwort generieren. Schau gerne auf https://linktr.ee/inf.fachschaft für alle wichtigen Fachschaft-Links."

            dispatcher.utter_message(text=reply)

        except Exception as e:
            print(f"[Ollama-Fallback Error] {e}")
            dispatcher.utter_message(
                text="Sorry, ich konnte gerade keine Antwort vom internen Modell holen. Schau gerne auf https://linktr.ee/inf.fachschaft oder frag direkt per Mail an die Fachschaft!"
            )

        return []

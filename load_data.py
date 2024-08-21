import json
import os


def load_chatbot_arena_data():
    """Load the latest chatbot arena data."""
    with open(os.path.join(os.path.dirname(__file__), "chatbotarena_20240814.json")) as f:
        data = json.load(f)
    return data

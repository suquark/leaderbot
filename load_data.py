import json
import os


def load_chatbot_arena_data():
    """Load the latest chatbot arena data."""
    with open(os.path.join(os.path.dirname(__file__), "chatbotarena_20240814.json")) as f:
        data = json.load(f)
    return data


def load_chatbot_arena_data_strong_models():
    """Load the latest chatbot arena data (strong models only)."""
    with open(os.path.join(os.path.dirname(__file__), "chatbotarena_20240814.json")) as f:
        data = json.load(f)
    with open(os.path.join(os.path.dirname(__file__), "strong_models.json")) as f:
        strong_models = set(json.load(f))
    sorted_strong_models = []
    mapping = {}
    for i, model in enumerate(data["models"]):
        if model in strong_models:
            mapping[i] = len(sorted_strong_models)
            sorted_strong_models.append(model)
        else:
            mapping[i] = None

    new_X = []
    new_Y = []

    for x, y in zip(data["X"], data["Y"], strict=True):
        if mapping[x[0]] is None or mapping[x[1]] is None:
            continue
        new_X.append([mapping[x[0]], mapping[x[1]]])
        new_Y.append(y)

    return {
        "models": sorted_strong_models,
        "X": new_X,
        "Y": new_Y,
    }

import json
import pandas as pd
import tqdm


def generate_dataset(input_file: str):
    # load the JSON data from the local file
    with open(input_file, 'r') as file:
        battles = pd.read_json(file).sort_values(ascending=True, by=["tstamp"])

    # we use anony battles only for leaderboard
    battles = battles[battles["anony"]]

    # we de-duplicate top 0.1% redundant prompts
    # see https://lmsys.org/blog/2024-05-17-category-hard/#note-enhancing-quality-through-de-duplication
    print("Before dedup: ", len(battles))
    battles = battles[battles["dedup_tag"].apply(lambda x: x.get("sampled", False))]
    print("After dedup: ", len(battles))

    # get unique model names from "model_a" and "model_b" columns
    combined_series = pd.concat([battles["model_a"], battles["model_b"]]).drop_duplicates()
    # Convert to sorted list
    model_list = sorted(combined_series.tolist())

    data_dict = {}
    reverse_dict = {m: i for i, m in enumerate(model_list)}

    for _, row in tqdm.tqdm(battles.iterrows(), total=len(battles)):
        model_a, model_b = row["model_a"], row["model_b"]
        model_a_id = reverse_dict[model_a]
        model_b_id = reverse_dict[model_b]

        if row["winner"] == "model_a":
            index = 0
        elif row["winner"] == "model_b":
            index = 1
        elif row["winner"] == "tie":
            index = 2
        else:
            # both bad
            index = 3

        if model_a_id < model_b_id:
            pair = (model_a_id, model_b_id)
        else:
            pair = (model_b_id, model_a_id)
            # flip winner
            index = [1, 0, 2, 3][index]
        if pair not in data_dict:
            data_dict[pair] = [0, 0, 0, 0]

        data_dict[pair][index] += 1

    data = list(data_dict.items())
    data.sort(key=lambda x: x[0])
    return {"models": model_list, "X": [x[0] for x in data], "Y": [x[1][:3] for x in data]}


if __name__ == "__main__":
    import os

    # See https://colab.research.google.com/drive/1KdwokPjirkTmpO_P1WByFNFiqxWQquwH#scrollTo=EZvUIOhVZD27
    # Data file from https://storage.googleapis.com/arena_external_data/public/clean_battle_20240814_public.json
    input_file = os.path.join(os.path.dirname(__file__), "clean_battle_20240814_public.json")
    output_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "chatbotarena_20240814.json")
    dataset = generate_dataset(input_file)
    with open(output_file, "w") as f:
        json.dump(dataset, f)

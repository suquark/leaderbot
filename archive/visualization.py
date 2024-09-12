import numpy as np
import pandas as pd
import plotly.express as px

skip_models = ["bard-jan-24-gemini-pro", "gemini-1.5-pro-api-0409-preview"]


def pretty_print(weight, battle_record):
    # associate models with their optimized parameters
    result_dict = {}
    n_models = len(battle_record["models"])
    for i, name in enumerate(battle_record["models"]):
        if len(weight) >= 2 * n_models:
            result_dict[name] = (weight[i], weight[n_models + i])
        else:
            result_dict[name] = (weight[i], 0)
    # sort result by mean
    result_dict = dict(
        sorted(result_dict.items(), key=lambda x: x[1][0],
               reverse=True))  # print the result
    print("= Model ranking =".center(57, "="))
    print("=" * 57)
    print("Model".ljust(35), "Mean".rjust(10), "Std".rjust(10))
    print("-" * 57)
    for k, v in result_dict.items():
        print(k.ljust(35), f"{v[0]:.2f}".rjust(10), f"{v[1]:.2f}".rjust(10))
    print("=" * 57)


def plot_win_rate_bt(weight, battle_record, inference_func):
    models = battle_record["models"]
    n_models = len(models)
    n = min(n_models, 30)
    win_rate_matrix = np.zeros((n, n))

    result_dict = {}
    for i, name in enumerate(battle_record["models"]):
        result_dict[name] = weight[i]
    result_dict = dict(
        sorted(result_dict.items(), key=lambda x: x[1], reverse=True))
    models = list(result_dict.keys())

    for i in range(n):
        for j in range(n):
            if i == j:
                win_rate_matrix[i, j] = np.nan
                continue
            if models[i] in skip_models or models[j] in skip_models:
                win_rate_matrix[i, j] = np.nan
                continue
            model_id_i = battle_record["model_indices"][models[i]]
            model_id_j = battle_record["model_indices"][models[j]]

            # calculate the win rate of model i against model j
            pair = (models[i], models[j])
            pair_index = battle_record["pair_indices"].get(pair, None)
            if pair_index is None:
                pair_index = battle_record["pair_indices"][pair[::-1]]

            p_win, p_loss, p_tie = inference_func(weight, model_id_i,
                                                  model_id_j, pair_index,
                                                  n_models)
            del p_loss, p_tie

            # Here we use [j, i] to respect the order of chatbot arena
            win_rate_matrix[j, i] = p_win

    # Assuming you have your data in a DataFrame named 'df'
    # Replace this with your actual data loading code
    data = {'Model A': models[:n]}
    for i in range(n):
        data[models[i]] = win_rate_matrix[i, :n]

    df = pd.DataFrame(data).set_index('Model A')

    # Create the heatmap
    fig = px.imshow(
        df,
        color_continuous_scale='RdBu',
        text_auto=".2f",
        title=
        "Predicted Win Rate Using Elo Ratings for Model A in an A vs. B Battle",
        labels=dict(x="Model B", y="Model A", color="Predicted Win Rate"))

    # Customize the plot
    fig.update_layout(xaxis_title="Model B",
                      yaxis_title="Model A",
                      xaxis_side="top",
                      height=900,
                      width=900,
                      title_y=0.07,
                      title_x=0.5)

    fig.update_traces(
        hovertemplate=
        "Model A: %{y}<br>Model B: %{x}<br>Win Rate: %{z}<extra></extra>")
    # Show the plot
    fig.show()


def plot_win_rate(weight, battle_record):
    models = battle_record["models"]
    n_models = len(models)
    n = min(n_models, 30)
    win_rate_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                win_rate_matrix[i, j] = np.nan
                continue
            if models[i] in skip_models or models[j] in skip_models:
                win_rate_matrix[i, j] = np.nan
                continue
            model_id_i = battle_record["model_indices"][models[i]]
            model_id_j = battle_record["model_indices"][models[j]]

            # calculate the win rate of model i against model j
            xi = weight[model_id_i]
            xj = weight[model_id_j]
            pair = (models[i], models[j])
            pair_index = battle_record["pair_indices"].get(pair, None)
            if pair_index is None:
                pair_index = battle_record["pair_indices"][pair[::-1]]
            rij = weight[n_models + pair_index]
            win_rate = (np.exp(xi) + 0.5 * np.exp(rij + 0.5 * (xi + xj))) / (
                np.exp(xi) + np.exp(xj) + np.exp(rij + 0.5 * (xi + xj)))
            # Here we use [j, i] to respect the order of chatbot arena
            win_rate_matrix[j, i] = win_rate

    # Assuming you have your data in a DataFrame named 'df'
    # Replace this with your actual data loading code
    data = {'Model A': models[:n]}
    for i in range(n):
        data[models[i]] = win_rate_matrix[i, :n]

    df = pd.DataFrame(data).set_index('Model A')

    # Create the heatmap
    fig = px.imshow(
        df,
        color_continuous_scale='RdBu',
        text_auto=".2f",
        title=
        "Predicted Win Rate Using Elo Ratings for Model A in an A vs. B Battle",
        labels=dict(x="Model B", y="Model A", color="Predicted Win Rate"))

    # Customize the plot
    fig.update_layout(xaxis_title="Model B",
                      yaxis_title="Model A",
                      xaxis_side="top",
                      height=900,
                      width=900,
                      title_y=0.07,
                      title_x=0.5)

    fig.update_traces(
        hovertemplate=
        "Model A: %{y}<br>Model B: %{x}<br>Win Rate: %{z}<extra></extra>")
    # Show the plot
    fig.show()


def plot_correlation(weight, battle_record):
    models = battle_record["models"]
    n_models = len(models)
    n = min(n_models, 30)
    correction = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                correction[i, j] = np.nan
                continue
            if models[i] in skip_models or models[j] in skip_models:
                correction[i, j] = np.nan
                continue
            pair = (models[i], models[j])
            pair_index = battle_record["pair_indices"].get(pair, None)
            if pair_index is None:
                pair_index = battle_record["pair_indices"][pair[::-1]]
            rij = weight[n_models + pair_index]
            # Here we use [j, i] to respect the order of chatbot arena
            correction[j, i] = rij

    # Assuming you have your data in a DataFrame named 'df'
    # Replace this with your actual data loading code
    data = {'Model A': models[:n]}
    for i in range(n):
        data[models[i]] = correction[i, :n]

    df = pd.DataFrame(data).set_index('Model A')

    # Create the heatmap
    fig = px.imshow(
        df,
        color_continuous_scale='RdBu',
        text_auto=".2f",
        title="Predicted Correlation for Model A in an A vs. B Battle",
        labels=dict(x="Model B", y="Model A", color="Predicted Correlation"))

    # Customize the plot
    fig.update_layout(xaxis_title="Model B",
                      yaxis_title="Model A",
                      xaxis_side="top",
                      height=900,
                      width=900,
                      title_y=0.07,
                      title_x=0.5)

    fig.update_traces(
        hovertemplate=
        "Model A: %{y}<br>Model B: %{x}<br>Correlation: %{z}<extra></extra>")
    # Show the plot
    fig.show()

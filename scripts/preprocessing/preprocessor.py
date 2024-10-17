from . import preprocessor
import pandas as pd
import nltk
from nltk import word_tokenize, pos_tag
from collections import Counter


class Preprocessor:

    def extract_additional_data(data: pd.DataFrame):
        print(type(data["instructions"]))
        data["instruction_length"] = data["instructions"].apply(len)
        data["mean_alcohol_percentage"] = data["ingredients"].apply(
            lambda row: int(
                sum(
                    ingredient["percentage"]
                    for ingredient in row
                    if ingredient["percentage"] is not None or not "null"
                )
                / len(row)
            )
        )
        data["ingredients"] = data["ingredients"].apply(
            lambda row: [ingredient["name"] for ingredient in row]
        )

        data["complexity_score"] = data["instructions"].apply(calculate_complexity)

        data = data.drop(["instructions", "alcoholic", "tags"], axis=1)
        data["mean_alcohol_percentage"] = data["mean_alcohol_percentage"].replace(
            0,
            int(
                data[data["mean_alcohol_percentage"] != 0][
                    "mean_alcohol_percentage"
                ].median()
            ),
        )
        data.fillna("[]", inplace=True)
        data.to_csv("data.csv")
        return data


def calculate_complexity(instructions: str) -> float:

    nltk.download("punkt", quiet=True)
    nltk.download("averaged_perceptron_tagger", quiet=True)

    tokens = word_tokenize(instructions.lower())
    tagged = pos_tag(tokens)

    verbs = [word for word, pos in tagged if pos.startswith("VB")]

    bartender_imperatives = [
        word
        for word, pos in tagged
        if pos == "VB"
        or (pos == "NN" and word in ["stir", "garnish", "serve", "shake", "mix"])
    ]

    all_action_words = list(set(verbs + bartender_imperatives))

    verb_count = len(all_action_words)
    unique_verbs_count = len(set(all_action_words))

    if verb_count == 0:
        return 1
    complexity_score = (verb_count * 0.7 + unique_verbs_count * 0.3) * 2
    normalized_complexity = min(10, max(1, complexity_score))

    return normalized_complexity

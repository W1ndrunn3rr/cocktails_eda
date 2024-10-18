from . import preprocessor
import pandas as pd
import nltk
from nltk import word_tokenize, pos_tag
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer


class Preprocessor:

    def extract_additional_data(data: pd.DataFrame):
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
        data = _encode_data(data)

        data = ingredients_PCA(data)

        data.to_csv("data.csv")

        return data


def _encode_data(encode_data: pd.DataFrame):
    cocktail_glass_dict = {
        "Cocktail glass": 1,
        "Old-fashioned glass": 2,
        "Highball glass": 3,
        "Whiskey sour glass": 4,
        "Collins glass": 5,
        "Champagne flute": 6,
        "Pousse cafe glass": 7,
        "Copper Mug": 8,
        "Whiskey Glass": 9,
        "Brandy snifter": 10,
        "White wine glass": 11,
    }
    glass_label_coding = encode_data["glass"].apply(
        lambda glass: cocktail_glass_dict[glass]
    )

    category_one_hot = pd.get_dummies(encode_data["category"], dtype="i")
    encode_data = encode_data.drop(["category"], axis=1)
    encode_data = encode_data.join(category_one_hot)
    encode_data["glass"] = glass_label_coding

    return encode_data


def ingredients_PCA(data: pd.DataFrame):

    data["ingredients"] = data["ingredients"].apply(lambda i: " ".join(i))

    tfidf_vectorizer = TfidfVectorizer(
        tokenizer=lambda ingredient: ingredient.split(", ")
    )
    X_tfidf = tfidf_vectorizer.fit_transform(data["ingredients"])

    pca = PCA(n_components=10)
    X_pca = pca.fit_transform(X_tfidf.toarray())

    pca_df = pd.DataFrame(
        X_pca, columns=[f"ingredient_pc{i+1}" for i in range(X_pca.shape[1])]
    )

    final_data = pd.concat([data, pca_df], axis=1)

    return final_data.drop(["ingredients"], axis=1)


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

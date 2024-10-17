from dataclasses import dataclass
import pandas as pd


@dataclass
class Loader:

    def __init__(self):
        self.data = pd.read_json("../data/cocktail_dataset.json")
        self._prepare_data()

    def _prepare_data(self):
        self.data = self.data.drop(
            ["id", "name", "imageUrl", "createdAt", "updatedAt"], axis=1
        )
        self.data["ingredients"] = self.data["ingredients"].apply(
            self._clean_ingredients
        )

    def _clean_ingredients(self, ingredient_list: list) -> list:
        keys = ["type", "percentage", "name"]
        return [
            {k: v for k, v in ingredient.items() if k in keys}
            for ingredient in ingredient_list
        ]

    def get_data(self) -> pd.DataFrame:
        return self.data

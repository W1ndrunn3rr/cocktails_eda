from load_prepare.loader import Loader
from preprocessing.preprocessor import Preprocessor


def main():
    loader = Loader()
    data = Preprocessor.extract_additional_data(loader.get_data())
    print(data["ingredients"])


if __name__ == "__main__":
    main()

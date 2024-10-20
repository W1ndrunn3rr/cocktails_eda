from load_prepare.loader import Loader
from preprocessing.preprocessor import Preprocessor
from clustering.cluster import Cluster


def main():
    loader = Loader()
    data = Preprocessor.extract_additional_data(loader.get_data())
    cluster = Cluster(data)


if __name__ == "__main__":
    main()

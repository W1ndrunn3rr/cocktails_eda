from load_prepare.loader import Loader
from preprocessing.preprocessor import Preprocessor
from clastering.claster import Claster


def main():
    loader = Loader()
    data = Preprocessor.extract_additional_data(loader.get_data())
    claster = Claster(data)

    claster._transform_data()
    print(claster.get_X())


if __name__ == "__main__":
    main()

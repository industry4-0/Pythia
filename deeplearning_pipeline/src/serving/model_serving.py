from tensorflow import keras
import argparse


def main(args):
    """
    Main function
    """

    model = keras.models.load_model(args.keras_model)
    keras.experimental.export_saved_model(model, args.output_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--keras_model', type=str, help='Path to keras h5 model file')
    parser.add_argument('--output_folder', type=str, help='Path to directory of exported serving files')
    args = parser.parse_args()

    main(args)
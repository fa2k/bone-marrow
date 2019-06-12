import argparse
import sys
import tensorflow as tf
import numpy as np

def main(model_path, input_path, input_norm, output_norm, delimiter):
    model = tf.keras.models.load_model(model_path)
    data = np.transpose(np.loadtxt(input_path, delimiter=delimiter))
    predictions = model.predict(data / input_norm)
    rounded_predictions = np.round(predictions * output_norm)
    np.savetxt(sys.stdout, rounded_predictions, fmt="%1.0f", delimiter=delimiter)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Make predictions for start and end of bone marrow",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument("--model", default="model.h5", help="Model file to use in Keras HDF5 format")
    parser.add_argument("--output-norm", default=40, type=float, help="Normalisation constant to use for outputs (bounds)")
    parser.add_argument("--input-norm", default=255, type=float, help="Normalisation constant for inputs (intensities)")
    parser.add_argument("--delimiter", default=";", help="Field delimiter used in the program's input and output")
    parser.add_argument("INPUT", help="Input file containing the intensity values")
    args = parser.parse_args()
    main(args.model, args.INPUT, args.input_norm, args.output_norm, args.delimiter)
    
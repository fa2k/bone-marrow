import argparse
import sys
import re
import tensorflow as tf
import numpy as np

NUM_INTENSITY_VALUES = 50
MAX_INTENSITY_VALUE = 255

parser = argparse.ArgumentParser(
    description="Run a Keras model to identify layer boundaries based on a "
                "vector of intensities.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
parser.add_argument("--model", default="model.h5",
        help="Model file to use in Keras HDF5 format")
parser.add_argument("--delimiter", default=";",
        help="Field delimiter used in the program's input and output")
parser.add_argument("--output-file", default=None,
        help="Output file name. Default is INPUTNAME_ObBmIb_bounds.txt, "
             "where INPUTNAME is the input file name without extension.")
parser.add_argument("INPUTS", nargs="+",
        help="Any number of input files containing the intensity values.")

args = parser.parse_args()

def load_input_and_orient(input_file):
    data = np.loadtxt(input_file, delimiter=args.delimiter)
    is_transposed = False
    if data.shape[1] != NUM_INTENSITY_VALUES:
        if data.shape[0] == NUM_INTENSITY_VALUES:
            data = np.transpose(data)
            is_transposed = True
        else:
            print("Error: The input data has dimensions {}, but expected to have {}"
                    " intensity values per observation.".format(data.shape,
                        NUM_INTENSITY_VALUES)
                    )
            sys.exit(1)
    return data, is_transposed

def main():
    model = tf.keras.models.load_model(args.model)

    if len(args.INPUTS) > 1 and args.output_file:
        print("Error: output file cannot be specified when there are multiple inputs.")
        sys.exit(1)

    for input_filename in args.INPUTS:
        data, is_transposed = load_input_and_orient(input_filename)

        norm_inputs = data / MAX_INTENSITY_VALUE
        predictions = model.predict(norm_inputs)
        predictions *= NUM_INTENSITY_VALUES
        predictions = np.clip(predictions, a_min=1, a_max=NUM_INTENSITY_VALUES,
                                out=predictions)
        num_predictions = len(predictions)
        if is_transposed:
            predictions = np.transpose(predictions)
        if args.output_file:
            output_file = args.output_file
        else:
            output_file = re.sub(r"\.\w+$", "", input_filename)
            output_file += "_ObBmIb_bounds.txt"
        np.savetxt(output_file, predictions, fmt="%1.0f", delimiter=args.delimiter)
        print("{} results written to file {}.".format(num_predictions, output_file))

if __name__ == "__main__":
    main()

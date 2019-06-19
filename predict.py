import argparse
import sys
import tensorflow as tf
import numpy as np

# TODO:
# File naming if not provided by the user the default could simply be the input file name stripped of its extension + "_ObBmIb_bounds" + .txt


NUM_LAYERS = 50

parser = argparse.ArgumentParser(
    description="Make predictions for start and end of bone marrow",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
parser.add_argument("--model", default="model.h5",
        help="Model file to use in Keras HDF5 format")
parser.add_argument("--input-norm", default=255, type=float,
        help="Normalisation constant for inputs (intensities)")
parser.add_argument("--delimiter", default=";",
        help="Field delimiter used in the program's input and output")
parser.add_argument("--output-file", default=None,
        help="Output file name. Default is INPUTNAME_ObBmIb_bounds.txt, "
             "where INPUTNAME is the input file name without extension.")
parser.add_argument("INPUT",
        help="Input file containing the intensity values, read from "
             "standard input if not given.")

args = parser.parse_args()

def load_input_and_orient(input_file):
    np.loadtxt(input_file, delimiter=args.delimiter)
    transposed = False
    if data.shape[1] != NUM_LAYERS:
        if data.shape[0] == NUM_LAYERS:
            data = np.transpose(data)
            transposed = True
        else:
            print("Error: The input data has dimensions {}, but expected to have {}"
                    " intensity values per observation.".format(data.shape,
                        NUM_LAYERS)
                    )
            sys.exit(1)
    return data, transposed

def main():
    model = tf.keras.models.load_model(args.model)
    data, transposed = load_input_and_orient(args.INPUT)

    # TODO: Change the normalisation constant to max of all the current inputs?
    norm_inputs = data / args.input_norm
    predictions = model.predict(norm_inputs)
    rounded_predictions = np.round(predictions * args.output_norm)
    predictions = np.clip(predictions, a_min=1, a_max=NUM_LAYERS, out=predictions)
    num_predictions = len(predictions)
    if transposed:
        predictions = np.transpose(predictions)
    if not args.output_file:
        args.output_file = re.sub(args.INPUT, r"(\.\w+)?$", "_ObBmIb_bounds.txt")
    np.savetxt(args.output_file, predictions, fmt="%1.0f", delimiter=args.delimiter)
    print("{} predictions written to file {}.".format(
        num_predictions,
        args.output_file
        ))

if __name__ == "__main__":
    main()

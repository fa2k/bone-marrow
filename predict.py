import argparse
import sys
import tensorflow as tf
import numpy as np

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

model = tf.keras.models.load_model(args.model)
data = np.transpose(np.loadtxt(args.INPUT, delimiter=args.delimiter))
predictions = model.predict(data / args.input_norm)
rounded_predictions = np.round(predictions * args.output_norm)
np.savetxt(sys.stdout, rounded_predictions, fmt="%1.0f", delimiter=args.delimiter)


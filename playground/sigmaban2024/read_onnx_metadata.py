import onnx
import argparse
import json

parser = argparse.ArgumentParser(description="Read ONNX model metadata")
parser.add_argument("onnx_file", type=str, help="Path to the ONNX model file")
args = parser.parse_args()

model_onnx = onnx.load(args.onnx_file)
model_metadata = model_onnx.metadata_props[0]
metadata = json.loads(model_metadata.value)

for k, v in metadata.items():
    print(f"{k}: {v}")

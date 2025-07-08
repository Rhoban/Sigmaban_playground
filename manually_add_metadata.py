import onnx
import json

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True, help="Path to the ONNX model file")
args = parser.parse_args()

metadata = {
    "kind": "vel",
    "feet_spacing": None,
    "dx_range": [-0.1, 0.12],
    "dy_range": [-0.1, 0.1],
    "dtheta_range": [0.6, 0.6],
    "mapping_matrix": None
}



def add_metadata_to_onnx(model_path, kind, dx_range, dy_range, dtheta_range):
    model_onnx = onnx.load(model_path)
    metadata = model_onnx.metadata_props.add()
    # metadata = model_onnx.metadata_props[0]
    metadata.key = "metadata"
    data = {
        "kind" : kind,
        "feet_spacing" : None,
        "dx_range" : dx_range,
        "dy_range" : dy_range,
        "dtheta_range" : dtheta_range,
        "mapping_matrix" : None
    }
    data_json = json.dumps(data)
    metadata.value = data_json

    name = model_path.strip(".onnx")
    name += "_with_metadata.onnx"

    print(f"Saving ONNX model with metadata to {name}")
    onnx.save(model_onnx, name)

# model_onnx = onnx.load(args.model_path)
# print(model_onnx.metadata_props)
# exit()

add_metadata_to_onnx(
    args.model_path,
    metadata["kind"],
    metadata["dx_range"],
    metadata["dy_range"],
    metadata["dtheta_range"]
)

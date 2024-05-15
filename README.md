# lookdeep-Pytorch_Retinaface

Tools to export, convert, and train the RetinaFace face detection model for use the with [lightfoot](https://github.com/lookdeep/lightfoot) edge pipeline [faces process](https://github.com/lookdeep/lightfoot/tree/ai/retinaface-cpp/ld_edge_pipeline/src/lookdeep/ml/process/faces).

## Overview

The production implementation uses the torch weights provided in the [forked repo](https://github.com/biubug6/Pytorch_Retinaface). Those weights are converted to from torch to ONNX, then ONNX to RKNN whereby the RKNN models are exported in both floating point (`fp`) and quantized (`i8`) formats. All intermediate and final RKNN models converted by the processes described herein can be found in Google Cloud Storage (GCS) [`gs://ld-models/RetinaFace`](https://console.cloud.google.com/storage/browser/ld-models/RetinaFace). The full conversion process is reproduced here for completeness.

Note that the ONNX to RKNN conversion script is adapted from [https://github.com/airockchip/rknn_model_zoo/blob/main/examples/RetinaFace/python/convert.py](https://github.com/airockchip/rknn_model_zoo/blob/main/examples/RetinaFace/python/convert.py).

## Install

```bash
git clone https://github.com/lookdeep/lookdeep-Pytorch_Retinaface/tree/master
cd lookdeep-Pytorch_Retinaface
python3.10 -m venv venv  # Ensure that requirements.txt refers to the rknn-toolkit2 wheel commensurate with your python version.
. venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```
## Download
For pretrained weights from the [forked repo](https://github.com/biubug6/Pytorch_Retinaface) are available in GCS.

```bash
./download_weights.sh
```

## Export

### `.pth` to `.onnx`

```
python convert_to_onnx.py -m weights/Resnet50_Final.pth --network resnet50 --long_side 320
python convert_to_onnx.py -m weights/Resnet50_Final.pth --network resnet50 --long_side 640
python convert_to_onnx.py -m weights/mobilenet0.25_Final.pth --network mobile0.25 --long_side 320
python convert_to_onnx.py -m weights/mobilenet0.25_Final.pth--network mobile0.25 --long_side 640
```

### `.onnx` to `.rknn`


**Note**: If converting from a newly trained model, ensure specification of `--mlflow-run-id` and `--mlflow-run-name`.

```
# Resnet50
python convert_to_rknn.py weights/RetinaFace_resnet50_320_opset17.onnx
python convert_to_rknn.py weights/RetinaFace_resnet50_320_opset17.onnx --quantize
python convert_to_rknn.py weights/RetinaFace_resnet50_640_opset17.onnx
python convert_to_rknn.py weights/RetinaFace_resnet50_640_opset17.onnx --quantize

# Mobilenet0.25
python convert_to_rknn.py weights/RetinaFace_mobile0.25_320_opset17.onnx
python convert_to_rknn.py weights/RetinaFace_mobile0.25_320_opset17.onnx --quantize
python convert_to_rknn.py weights/RetinaFace_mobile0.25_640_opset17.onnx
python convert_to_rknn.py weights/RetinaFace_mobile0.25_640_opset17.onnx --quantize
```


## Train

See [README.OG.md](https://github.com/lookdeep/lookdeep-Pytorch_Retinaface/blob/master/README.OG.md)  

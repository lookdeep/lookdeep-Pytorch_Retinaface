#!/bin/bash

[ ! -d "./weights" ] && mkdir ./weights
gsutil -m cp gs://ld-models/RetinaFace/Resnet50_Final.pth ./weights/
gsutil -m cp gs://ld-models/RetinaFace/mobilenetV1X0.25_pretrain.tar ./weights/
gsutil -m cp gs://ld-models/RetinaFace/Resnet50_Final.pth ./weights/


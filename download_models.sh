#!/usr/bin/env bash
# Downloads the two OpenCV Zoo ONNX models (~40 MB total) into ./models
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p models
base=https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models
curl -L --fail -o models/face_detection_yunet_2023mar.onnx  $base/face_detection_yunet/face_detection_yunet_2023mar.onnx
curl -L --fail -o models/face_recognition_sface_2021dec.onnx $base/face_recognition_sface/face_recognition_sface_2021dec.onnx
ls -lh models
# expected: yunet ~230K, sface ~37M. A file of a few hundred bytes means the download failed.

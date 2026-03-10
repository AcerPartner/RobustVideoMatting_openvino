# Robust Video Matting OpenVino version

Robust Video Matting model is a powerful model that can provide high-quality matting images.

For details see the [repository](https://github.com/PeterL1n/RobustVideoMatting). For model convert tool see [openvinotoolkit
open_model_zoo](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/robust-video-matting-mobilenetv3)

## How to convert Robust Video Matting to openvino
1. Download robust-video-matting-mobilenetv3 from openvino model zoo
```
pip install openvino-dev
omz_downloader --name robust-video-matting-mobilenetv3
omz_converter --name robust-video-matting-mobilenetv3 --precisions FP16
```
2. Reshap model
```
pip install openvino==2025.3
python .\reshape_ov.py
```
## Notice
Based on RobustVideoMatting

Modified by JoelanHsiehAcer

Date: 2026-03-06

## Legal Information

The original model is distributed under the
[GPL-3.0 License](https://github.com/DmitriySidnev/RobustVideoMatting/blob/master/LICENSE).

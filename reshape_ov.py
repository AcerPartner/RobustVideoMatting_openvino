# Copyright (C) 2026 Acer Corporation
# Licensed under the GNU General Public License v3.0
# Modified by JoelanHsiehAcer
# Date: 2026-03-06

from openvino.runtime.passes import Manager, MakeStateful
import openvino as ov

#download model from runing 
#  omz_downloader --name robust-video-matting-mobilenetv3
#  omz_converter --name robust-video-matting-mobilenetv3 --precisions FP16
#https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/robust-video-matting-mobilenetv3

HEIGHT = 360
WIDTH = 640
core = ov.Core()

ov_input = {
    'src': [1, 3, HEIGHT, WIDTH],
    'r1': [1, 16, HEIGHT // 5, WIDTH // 5],
    'r2': [1, 20, HEIGHT // 10, WIDTH // 10],
    'r3': [1, 40, HEIGHT // 20, WIDTH // 20],
    'r4': [1, 64, HEIGHT // 40, WIDTH // 40],
}

ov_model = core.read_model(model="./public/robust-video-matting-mobilenetv3/FP16/robust-video-matting-mobilenetv3.xml")

ov_model.reshape(ov_input)
ov.save_model(ov_model, f"./public/robust-video-matting-mobilenetv3/FP16/robust-video-matting-mobilenetv3_{WIDTH}x{HEIGHT}.xml")
print("model done")
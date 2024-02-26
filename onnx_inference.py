'''
Ronak Bhanushali and Ruohe Zhou
Spring 2024
'''
import torch
import onnxruntime as ort
import numpy as np

torch.manual_seed(100)
data = torch.rand((10, 3, 256, 128)).cuda()

# Infer from ONNX
data = data.cpu().numpy()
ort_session = ort.InferenceSession("/home/ronak/Downloads/siamese_net_market_20.onnx")
vector_onnx = ort_session.run(None, {"input": data})
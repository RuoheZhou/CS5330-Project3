'''
Ronak Bhanushali and Ruohe Zhou
Spring 2024
'''
import torch
import onnxruntime as ort
import numpy as np
from PIL import Image
import os
import torchvision.transforms as T

def save_query_vectors():
    image_path = "/home/ronak/cs5330/project_3/DNN/query/"
    transform = T.Compose(
            [
                T.Resize([256, 128]),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    input_images = []
    labels = []
    for file in sorted(os.listdir(image_path)):
        labels.append(file.split('_')[0])
        image = Image.open(os.path.join(image_path,file)).convert('RGB')
        input = transform(image)
        input = torch.unsqueeze(input, dim=0)
        input_images.append(input)

    num_input_images = len(input_images)
    num_filler_images = 20-num_input_images


    filler_images = torch.zeros((num_filler_images,3, 256, 128))
    input_images.append(filler_images)

    input_tensors = torch.cat(input_images, dim=0)

    # Infer from ONNX
    data = input_tensors.cpu().numpy()
    ort_session = ort.InferenceSession("/home/ronak/cs5330/project_3/rouhe/mobilenetv3_modified.onnx")
    vector_onnx = ort_session.run(None, {"input": data})

    numpy_vector_onnx = np.asarray(vector_onnx).squeeze()[:num_input_images]

    file_path = "/home/ronak/cs5330/project_3/rouhe/data/features_query_dnn.csv"

    data_with_labels = np.column_stack((np.array(labels)[:, np.newaxis], numpy_vector_onnx))

    np.savetxt(file_path, data_with_labels, delimiter=',', fmt='%s') 

def save_anchor_vectors():
    image_path = "/home/ronak/cs5330/project_3/DNN/anchors/"
    transform = T.Compose(
            [
                T.Resize([256, 128]),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    input_images = []
    labels = []
    for file in sorted(os.listdir(image_path)):
        labels.append(file.split('_')[0])
        image = Image.open(os.path.join(image_path,file)).convert('RGB')
        input = transform(image)
        input = torch.unsqueeze(input, dim=0)
        input_images.append(input)

    num_input_images = len(input_images)
    num_filler_images = 20-num_input_images


    filler_images = torch.zeros((num_filler_images,3, 256, 128))
    input_images.append(filler_images)

    input_tensors = torch.cat(input_images, dim=0)

    # Infer from ONNX
    data = input_tensors.cpu().numpy()
    ort_session = ort.InferenceSession("/home/ronak/Downloads/siamese_net_market_20.onnx")
    vector_onnx = ort_session.run(None, {"input": data})

    numpy_vector_onnx = np.asarray(vector_onnx).squeeze()[:num_input_images]

    file_path = "/home/ronak/cs5330/project_3/rouhe/data/features_dnn.csv"

    data_with_labels = np.column_stack((np.array(labels)[:, np.newaxis], numpy_vector_onnx))

    np.savetxt(file_path, data_with_labels, delimiter=',', fmt='%s')
    
    
save_anchor_vectors()
save_query_vectors()
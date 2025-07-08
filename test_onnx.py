import onnxruntime as ort
import numpy as np

import cv2
import torch
import os
import argparse


def run_inference(onnx_dir, img_path, datatype='fp32'):
    """
    Run inference using the ONNX model.
    
    Args:
        model_path (str): Path to the ONNX model file.
        img_path (str): Path to the input image.
        datatype (str): Data type for the input tensor ('float32' or 'float16').
    """
    # 检查输入数据类型
    if datatype not in ['fp32', 'fp16']:
        raise ValueError("datatype must be 'fp32' or 'fp16'")

    # 设置数据类型
    if datatype == 'fp16':
        torch.set_default_dtype(torch.float16)
        precision = 'fp16'
        np_dtype = np.float16
    elif datatype == 'fp32':
        torch.set_default_dtype(torch.float32)
        precision = 'fp32'
        np_dtype = np.float32
    else:
        raise ValueError("Unsupported datatype. Use 'fp32' or 'fp16'.")

    onnx_path = os.path.join(onnx_dir, precision, f"s3diff_{precision}.onnx")
    # 创建 ONNX 推理会话（确保目录下有 s3diff.onnx 和所有碎片文件）
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    # 获取模型输入名称
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    # 准备输入 (例如 [-1,1] normalized, float32)
    # dummy_input = np.random.uniform(-1, 1, size=(1, 3, 256, 256)).astype(np.float32)

    img = cv2.imread(img_path)

    img = cv2.resize(img, (512, 512))
    lr_img = img[128:384, 128:384, :]
    # print('lr_img shape:', lr_img.shape)
    # cv2.imwrite(f"input_lr_image.png", lr_img)
    lr_img_rgb = cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB)
    lr_img_rgb = lr_img_rgb / 255.0  # Convert to range [0, 1]
    lr_img_rgb = lr_img_rgb.astype(np_dtype)  # Convert to specified dtype
    lr_img_rgb = lr_img_rgb * 2.0 - 1.0
    dummy_input = lr_img_rgb.unsqueeze(0)  # Add batch dimension
    dummy_input = np.transpose(dummy_input, (0, 3, 1, 2))  # Convert to NCHW format



    # 推理
    output = sess.run([output_name], {input_name: dummy_input})[0]

    print("ONNX 推理输出 shape:", output.shape)

    # 处理输出
    hr_img = output.squeeze(0).transpose(1, 2, 0)
    hr_img = ((hr_img * 0.5 + 0.5) * 255.0).clip(0, 255).astype(np.uint8)
    hr_img = cv2.cvtColor(hr_img, cv2.COLOR_RGB2BGR)
    print('hr_img shape:', hr_img.shape)
    cv2.imwrite(f"onnx_hr_image_{precision}.png", hr_img)



if __name__ == "__main__":
    
    argparser = argparse.ArgumentParser(description="Run ONNX inference on an image.")
    argparser.add_argument("--onnx_dir", type=str, required=True, help="Directory containing ONNX model files.")
    argparser.add_argument("--img_path", type=str, required=True, help="Path to the input image.")
    argparser.add_argument("--datatype", type=str, default='fp32', choices=['fp32', 'fp16'], help="Data type for the input tensor.")
    args = argparser.parse_args()   

    run_inference(args.onnx_dir, args.img_path, args.datatype)
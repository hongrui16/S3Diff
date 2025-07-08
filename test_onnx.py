import onnxruntime as ort
import numpy as np

import cv2
import torch
import os
import argparse


def run_inference(onnx_dir, onnx_path, img_path, precision='fp32'):
    """
    Run inference using the ONNX model.
    
    Args:
        model_path (str): Path to the ONNX model file.
        img_path (str): Path to the input image.
        datatype (str): Data type for the input tensor ('float32' or 'float16').
    """
    # 检查输入数据类型
    if precision not in ['fp32', 'fp16']:
        raise ValueError("datatype must be 'fp32' or 'fp16'")
    
    if onnx_dir is not None:
        onnx_path = os.path.join(onnx_dir, precision, f"s3diff_{precision}.onnx")
    elif onnx_path is not None:
        onnx_path = onnx_path
        if 'fp16' in onnx_path:
            precision = 'fp16'
        elif 'fp32' in onnx_path:
            precision = 'fp32'
    else:
        raise ValueError("Please provide either onnx_dir or onnx_path.")

    # 设置数据类型
    if precision == 'fp16':
        torch.set_default_dtype(torch.float16)
        np_dtype = np.float16
    elif precision == 'fp32':
        torch.set_default_dtype(torch.float32)
        np_dtype = np.float32
    else:
        raise ValueError("Unsupported datatype. Use 'fp32' or 'fp16'.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    img = cv2.imread(img_path)

    img = cv2.resize(img, (512, 512))
    lr_img = img[128:384, 128:384, :]
    # print('lr_img shape:', lr_img.shape)
    # cv2.imwrite(f"input_lr_image.png", lr_img)
    lr_img_rgb = cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB)
    lr_img_rgb = lr_img_rgb / 255.0  # Convert to range [0, 1]
    lr_img_rgb = lr_img_rgb.astype(np_dtype)  # Convert to specified dtype
    lr_img_rgb = lr_img_rgb * 2.0 - 1.0
    lr_img_rgb = np.expand_dims(lr_img_rgb, axis=0)  # shape: [1, C, H, W]
    dummy_input = np.transpose(lr_img_rgb, (0, 3, 1, 2))  # Convert to NCHW format

    dummy_input = np.ascontiguousarray(dummy_input)
    # 创建 ONNX 推理会话（确保目录下有 s3diff.onnx 和所有碎片文件）

    if torch.cuda.is_available():
        sess = ort.InferenceSession(onnx_path, providers=["CUDAExecutionProvider"])
    else:
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    # 获取模型输入名称
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name



    # 推理
    output = sess.run([output_name], {input_name: dummy_input})[0]

    print("ONNX 推理输出 shape:", output.shape)

    # 处理输出
    hr_img = output.squeeze(0).transpose(1, 2, 0)
    hr_img = hr_img.clip(-1.0, 1.0)  # 确保输出在 [-1, 1] 范围内
    hr_img = ((hr_img * 0.5 + 0.5) * 255.0).clip(0, 255).astype(np.uint8)
    hr_img = cv2.cvtColor(hr_img, cv2.COLOR_RGB2BGR)
    print('hr_img shape:', hr_img.shape)
    cv2.imwrite(f"onnx_hr_image_{precision}.png", hr_img)



if __name__ == "__main__":
    
    argparser = argparse.ArgumentParser(description="Run ONNX inference on an image.")
    argparser.add_argument("--onnx_dir", type=str, default=None, help="Directory containing ONNX model files.")
    argparser.add_argument("--onnx_path", type=str, default=None, help="Name of the ONNX model (without extension).")
    argparser.add_argument("--img_path", type=str, required=True, help="Path to the input image.")
    argparser.add_argument("--precision", type=str, default='fp32', choices=['fp32', 'fp16'], help="Data precision for the input tensor.")
    args = argparser.parse_args()   

    run_inference(args.onnx_dir, args.onnx_path, args.img_path, args.precision)
<h2 align="center">Degradation-Guided One-Step Image Super-Resolution with Diffusion Priors</h2>

<div align="center">

<a href="https://arxiv.org/abs/2409.17058"><img src="https://img.shields.io/badge/ArXiv-2409.17058-red"></a>  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<img src="https://visitor-badge.laobi.icu/badge?page_id=ArcticHare105/S3Diff" alt="visitors">
</p>

[Aiping Zhang]()<sup>1\*</sup>, [Zongsheng Yue]()<sup>2,\*</sup>, [Renjing Pei]()<sup>3</sup>, [Wenqi Ren]()<sup>1</sup>, [Xiaochun Cao]()<sup>1</sup>

<sup>1</sup>School of Cyber Science and Technology, Shenzhen Campus of Sun Yat-sen University<br> <sup>2</sup>S-Lab, Nanyang Technological University<br> <sup>3</sup>Huawei Noah's Ark Lab<br>* Equal contribution.
</div>

:fire::fire::fire: We have released the code, cheers!


:star: If S3Diff is helpful for you, please help star this repo. Thanks! :hugs:



## <a name="setup"></a> ⚙️ Setup
```bash
conda create -n s3diff python=3.10
conda activate s3diff
pip install -r requirements.txt
```
Or use the conda env file that contains all the required dependencies.

```bash
conda env create -f environment.yaml
```

:star: Since we employ peft in our code, we highly recommend following the provided environmental requirements, especially regarding diffusers.

## <a name="training"></a> :wrench: Training

#### Step1: Download the pretrained models
We enable automatic model download in our code, if you need to conduct offline training, download the pretrained model [SD-Turbo](https://huggingface.co/stabilityai/sd-turbo)

#### Step2: Prepare training data
We train the S3Diff on [LSDIR](https://github.com/ofsoundof/LSDIR) + 10K samples from [FFHQ](https://github.com/NVlabs/ffhq-dataset), following [SeeSR](https://github.com/cswry/SeeSR) and [OSEDiff](https://github.com/cswry/OSEDiff).

#### Step3: Training for S3Diff

Please modify the paths to training datasets in `configs/sr.yaml`
Then run:

```bash
sh run_training.sh
```

If you need to conduct offline training, modify `run_training.sh` as follows, and fill in `sd_path` with your local path:

```bash
accelerate launch --num_processes=4 --gpu_ids="0,1,2,3" --main_process_port 29300 src/train_s3diff.py \
    --sd_path="path_to_checkpoints/sd-turbo" \
    --de_net_path="assets/mm-realsr/de_net.pth" \
    --output_dir="./output" \
    --resolution=512 \
    --train_batch_size=4 \
    --enable_xformers_memory_efficient_attention \
    --viz_freq 25
```

## <a name="run inference on your own data"></a> 💫 Inference

#### Step1: Download the pretrained models

We enable automatic model download in our code, if you need to conduct offline inference, download the pretrained model [SD-Turbo](https://huggingface.co/stabilityai/sd-turbo) and S3Diff [[HuggingFace](https://huggingface.co/zhangap/S3Diff) | [GoogleDrive](https://drive.google.com/drive/folders/1cWYQYRFpadC4K2GuH8peg_hWEoFddZtj?usp=sharing)]

Download 'de_net.pth' and 's3diff.pkl', and put them into 'checkpoints' folder

#### Step2: Inference on your own test data.

```
!python inferenceOnly.py \
    --sd_path="stabilityai/sd-turbo" \
    --de_net_path="checkpoints/de_net.pth" \
    --pretrained_path="checkpoints/s3diff.pkl" \
    --output_dir="S3Diffx4" \
    --input_dir="ls_images" \
```
## <a name="run evaluation for S3Diff"></a> 💫 Evaluation
Run evaluation for S3Diff on a SR benchmark.
#### Step1: Download the pretrained models

We enable automatic model download in our code, if you need to conduct offline inference, download the pretrained model [SD-Turbo](https://huggingface.co/stabilityai/sd-turbo) and S3Diff [[HuggingFace](https://huggingface.co/zhangap/S3Diff) | [GoogleDrive](https://drive.google.com/drive/folders/1cWYQYRFpadC4K2GuH8peg_hWEoFddZtj?usp=sharing)]

Download 'de_net.pth' and 's3diff.pkl', and put them into 'checkpoints' folder

#### Step2: Download datasets for inference and run

Please add the paths to evaluate datasets in `configs/sr_test.yaml` and the path of GT folder in `run_inference.sh`
Then run:

```bash
sh run_inference.sh
```

If you need to conduct offline inference, modify `run_inference.sh` as follows, and fill in with your paths:

```bash
accelerate launch --num_processes=1 --gpu_ids="0," --main_process_port 29300 src/inference_s3diff.py \
    --sd_path="path_to_checkpoints/sd-turbo" \
    --de_net_path="assets/mm-realsr/de_net.pth" \
    --pretrained_path="path_to_checkpoints/s3diff.pkl" \
    --output_dir="./output" \
    --ref_path="path_to_ground_truth_folder" \
    --align_method="wavelet"
```

#### Gradio Demo

Please install Gradio first
```bash
pip install gradio
```

Please run the following command to interact with the gradio website, have fun. 🤗

```
python src/gradio_s3diff.py 
```
![s3diff](assets/pic/gradio.png)


<!-- - [Installation](#installation)
- [Inference](#inference) -->

## <a name="update"></a>:new: Update

- **2024.10.07**: Add gradio demo 🚀
- **2024.09.25**: The code is released :fire:
- **2024.09.25**: This repo is released :fire:
<!-- - [**History Updates** >]() -->

## <a name="todo"></a>:hourglass: TODO

- [x] Release Code :computer:
- [x] Release Checkpoints :link:


## <a name="abstract"></a>:fireworks: Abstract

> Diffusion-based image super-resolution (SR) methods have achieved remarkable success by leveraging large pre-trained text-to-image diffusion models as priors. However, these methods still face two challenges: the requirement for dozens of sampling steps to achieve satisfactory results, which limits efficiency in real scenarios, and the neglect of degradation models, which are critical auxiliary information in solving the SR problem. In this work, we introduced a novel one-step SR model, which significantly addresses the efficiency issue of diffusion-based SR methods. Unlike existing fine-tuning strategies, we designed a degradation-guided Low-Rank Adaptation (LoRA) module specifically for SR, which corrects the model parameters based on the pre-estimated degradation information from low-resolution images. This module not only facilitates a powerful data-dependent or degradation-dependent SR model but also preserves the generative prior of the pre-trained diffusion model as much as possible. Furthermore, we tailor a novel training pipeline by introducing an online negative sample generation strategy. Combined with the classifier-free guidance strategy during inference, it largely improves the perceptual quality of the super-resolution results. Extensive experiments have demonstrated the superior efficiency and effectiveness of the proposed model compared to recent state-of-the-art methods.

## <a name="framework_overview"></a>:eyes: Framework Overview

<img src=assets/pic/main_framework.jpg>

:star: Overview of S3Diff. We enhance a pre-trained diffusion model for one-step SR by injecting LoRA layers into the VAE encoder and UNet. Additionally, we employ a pre-trained Degradation Estimation Network to assess image degradation that is used to guide the LoRAs with the introduced block ID embeddings. We tailor a new training pipeline that includes an online negative prompting, reusing generated LR images with negative text prompts. The network is trained with a combination of a reconstruction loss and a GAN loss.

## <a name="visual_comparison"></a>:chart_with_upwards_trend: Visual Comparison

### Image Slide Results
[<img src="assets/pic/imgsli1.png" height="235px"/>](https://imgsli.com/MzAzNjIy) [<img src="assets/pic/imgsli2.png" height="235px"/>](https://imgsli.com/MzAzNjQ1)  [<img src="assets/pic/imgsli3.png" height="235px"/>](https://imgsli.com/MzAzNjU4)
[<img src="assets/pic/imgsli4.png" height="272px"/>](https://imgsli.com/MzAzNjU5) [<img src="assets/pic/imgsli5.png" height="272px"/>](https://imgsli.com/MzAzNjI2)
### Synthesis Dataset

<img src=assets/pic/div2k_comparison.jpg>

### Real-World Dataset

<img src=assets/pic/london2.jpg>
<img src=assets/pic/realsr_vis3.jpg>

<!-- </details> -->

## :smiley: Citation

Please cite us if our work is useful for your research.

```
@article{2024s3diff,
  author    = {Aiping Zhang, Zongsheng Yue, Renjing Pei, Wenqi Ren, Xiaochun Cao},
  title     = {Degradation-Guided One-Step Image Super-Resolution with Diffusion Priors},
  journal   = {arxiv},
  year      = {2024},
}
```

## :notebook: License

This project is released under the [Apache 2.0 license](LICENSE).


## :envelope: Contact

If you have any questions, please feel free to contact zhangaip7@mail2.sysu.edu.cn.


# Hand1000: Generating Realistic Hands from Text with Only 1,000 Images

[Haozhuo Zhang](https://haozhuo-zhang.github.io/), [Bin Zhu](https://binzhubz.github.io/), [Yu Cao](https://haozhuo-zhang.github.io/Hand1000-project-page/), [Yanbin Hao](https://haoyanbin918.github.io/)

[[`Paper`](https://arxiv.org/abs/2408.15461)] [[`Project Page`](https://haozhuo-zhang.github.io/Hand1000-project-page/)] [[`BibTeX`](#citing-hand1000)]

![Hand1000 architecture](imgs/training.png?raw=true)

Text-to-image generation models have achieved remarkable advancements in recent years, aiming to produce realistic images from textual descriptions. However, these models often struggle with generating anatomically accurate representations of human hands. The resulting images frequently exhibit issues such as incorrect numbers of fingers, unnatural twisting or interlacing of fingers, or blurred and indistinct hands. These issues stem from the inherent complexity of hand structures and the difficulty in aligning textual descriptions with precise visual depictions of hands. To address these challenges, we propose a novel approach named **Hand1000** that enables the generation of realistic hand images with target gesture using only 1,000 training samples. The training of Hand1000 is divided into three stages with the first stage aiming to enhance the model’s understanding of hand anatomy by using a pre-trained hand gesture recognition model to extract gesture representation. The second stage further optimizes text embedding by incorporating the extracted hand gesture representation, to improve alignment between the textual descriptions and the generated hand images. The third stage utilizes the optimized embedding to fine-tune the Stable Diffusion model to generate realistic hand images. In addition, we construct the first publicly available dataset specifically designed for text-to-hand image generation. Based on the existing hand gesture recognition dataset, we adopt advanced image captioning models and LLaMA3 to generate high-quality textual descriptions enriched with detailed gesture information. Extensive experiments demonstrate that Hand1000 significantly outperforms existing models in producing anatomically correct hand images while faithfully representing other details in the text, such as faces, clothing and colors.

![Results](imgs/results.jpg?raw=true)

## Getting Started

This project is largely based on [Stable Diffusion and Imagic](https://github.com/justinpinkney/stable-diffusion/blob/main/notebooks/imagic.ipynb).

```bash
git clone https://github.com/facebookresearch/sam2.git && cd Hand1000

pip install -r requirements.txt
```
<!--
If you are installing on Windows, it's strongly recommended to use [Windows Subsystem for Linux (WSL)](https://learn.microsoft.com/en-us/windows/wsl/install) with Ubuntu.

To use the SAM 2 predictor and run the example notebooks, `jupyter` and `matplotlib` are required and can be installed by:

```bash
pip install -e ".[notebooks]"
```

Note:
1. It's recommended to create a new Python environment via [Anaconda](https://www.anaconda.com/) for this installation and install PyTorch 2.5.1 (or higher) via `pip` following https://pytorch.org/. If you have a PyTorch version lower than 2.5.1 in your current environment, the installation command above will try to upgrade it to the latest PyTorch version using `pip`.
2. The step above requires compiling a custom CUDA kernel with the `nvcc` compiler. If it isn't already available on your machine, please install the [CUDA toolkits](https://developer.nvidia.com/cuda-toolkit-archive) with a version that matches your PyTorch CUDA version.
3. If you see a message like `Failed to build the SAM 2 CUDA extension` during installation, you can ignore it and still use SAM 2 (some post-processing functionality may be limited, but it doesn't affect the results in most cases).

Please see [`INSTALL.md`](./INSTALL.md) for FAQs on potential issues and solutions.
-->
## Dataset Construction

First, download HaGRID dataset from [here](https://github.com/hukenovs/hagrid). Select 1000 images of target gesture for training and testing respectively and randomly.

Then use BLIP or PaliGemma or VitGpt2 to caption the images.

```bash
python BLIP2.py --gesture {gesture name}
```

or

```bash
python PaliGemma.py --gesture {gesture name}
```

or

```bash
python VitGpt2.py --gesture {gesture name}
```

Finally, use [llama](https://www.llama.com/) or other LLMs to enrich the original caption with rich hand gesture information. Note that this task is very simple for existing LLMs, and there are many implementations available online, all of which yield similar results. 

The prompt is as follows:

```bash
The following sentence contains incorrect hand gesture information or lacks hand gesture information. If it contains incorrect hand gesture information, modify it to 'making phone call hand gesture.' If it lacks hand gesture information, append 'making phone call hand gesture' to the end of the sentence. The original sentence is: 'a girl in a green shirt and glasses giving the peace sign.' The final sentence must include 'making phone call hand gesture' exactly as it is. Provide the modified sentence directly and independently.
```

Evaluate the dataset construction quality by running:
```bash
python dataset_evaluate.py --image_folder {Path to the folder containing images} --image_filenames_path {Path to the file containing image filenames} --captions_path {Path to the file containing captions}
```

## Training

Start training by running (You can change Lambda from 0.0 to 1.0 to see what would happen to the generated images.):

```bash
python train.py --gesture {Gesture name} --Lambda {Hyperparameter Lambda}
```

## Image generation

After training and saving the finetuned model. Generate images by running (You can change Mu from 0.0 to 1.0 to see what would happen to the generated images. Note that the value of Mu doesn't need to be the same as that of Lambda):

```bash
python image_generation.py --image_filenames_file {Path to the file containing image filenames} --prompts_file {Path to the file containing prompts} --feature_npy_path {Path to the npy file containing gesture features} --ckpt {Path to the finetuned model} --img_save_path {Path to save the images} --mu {hyperparameter mu}
```

## Image evaluation

After the generation of images, evaluate the quality of the generated images by running the following commands:

```bash
python cal_fid.py --real_images {Path to the folder containing real images} --fake_images {Path to the folder containing generated images}

python cal_fid-h.py --real_images {Path to the folder containing real images} --fake_images {Path to the folder containing generated images}

python cal_kid.py --real_images {Path to the folder containing real images} --fake_images {Path to the folder containing generated images}

python cal_kid-h.py --real_images {Path to the folder containing real images} --fake_images {Path to the folder containing generated images}

python hand_confidence.py --image_folder {Path to the folder containing generated images}
```

## Citing Hand1000

If you use Hand1000 in your research, please use the following BibTeX entry.

```bibtex
@misc{zhang2024hand1000generatingrealistichands,
      title={Hand1000: Generating Realistic Hands from Text with Only 1,000 Images}, 
      author={Haozhuo Zhang and Bin Zhu and Yu Cao and Yanbin Hao},
      year={2024},
      eprint={2408.15461},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2408.15461}, 
}
```

from omegaconf import OmegaConf
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
import os
from tqdm import tqdm
from einops import rearrange
import numpy as np
from pathlib import Path
from torch import nn
from sklearn.metrics.pairwise import cosine_similarity
import argparse

import sys
sys.path.append('./stable-diffusion')

parser = argparse.ArgumentParser(description='Image Generation')
parser.add_argument('--image_filenames_file', type=str, required=True, help='Path to the file containing image filenames')
parser.add_argument('--prompts_file', type=str, required=True, help='Path to the file containing prompts')
parser.add_argument('--feature_npy_path', type=str, required=True, help='Path to the npy file containing gesture features')
parser.add_argument('--ckpt', type=str, required=True, help='Path to the checkpoint')
parser.add_argument('--img_save_path', type=str, required=True, help='Path to save the images')
parser.add_argument('--mu', type=float, required=True, help='hyperparameter mu')
args = parser.parse_args()

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config

def load_model_from_config(config, ckpt, device="cpu", verbose=False):
    """Loads a model from config and a ckpt
    if config is a path will use omegaconf to load
    """
    if isinstance(config, (str, Path)):
        config = OmegaConf.load(config)

    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    model.cond_stage_model.device = device
    return model

@torch.no_grad()
def sample_model(model, sampler, c, h, w, ddim_steps, scale, ddim_eta, start_code=None, n_samples=1):
    """Sample the model"""
    uc = None
    if scale != 1.0:
        uc = model.get_learned_conditioning(n_samples * [""])

    shape = [4, h // 8, w // 8]
    samples_ddim, _ = sampler.sample(S=ddim_steps,
                                     conditioning=c,
                                     batch_size=n_samples,
                                     shape=shape,
                                     verbose=False,
                                     start_code=start_code,
                                     unconditional_guidance_scale=scale,
                                     unconditional_conditioning=uc,
                                     eta=ddim_eta,
                                    )
    return samples_ddim

def load_img(path, target_size=512):
    """Load an image, resize and output -1..1"""
    image = Image.open(path).convert("RGB")
    
    
    tform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
        ])
    image = tform(image)
    return 2.*image - 1.

def load_img_2(img, target_size=512):
    """Load an image, resize and output -1..1"""
    image = img.convert("RGB")
    
    tform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
        ])
    image = tform(image)
    return 2.*image - 1.

def decode_to_im(samples, n_samples=1, nrow=1):
    """Decode a latent and return PIL image"""
    samples = model.decode_first_stage(samples)
    ims = torch.clamp((samples + 1.0) / 2.0, min=0.0, max=1.0)
    x_sample = 255. * rearrange(ims.cpu().numpy(), '(n1 n2) c h w -> (n1 h) (n2 w) c', n1=n_samples//nrow, n2=nrow)
    img = Image.fromarray(x_sample.astype(np.uint8))
    return img

# Read image filenames and prompts from text files
def read_lines_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.read().splitlines()
    return lines

#image_folder = "/home/ubuntu/disk1/hagrid_dataset_512_1000"
image_filenames_file = args.image_filenames_file
prompts_file = args.prompts_file

# Load image filenames and prompts
image_filenames = read_lines_from_file(image_filenames_file)
prompts = read_lines_from_file(prompts_file)

nums = 10
device = "cuda:0"
config="./stable-diffusion/configs/stable-diffusion/v1-inference.yaml"
ckpt = args.ckpt
fusion_model_ckpt_path = ckpt

#encode_folder = "/home/ubuntu/disk1/18encodes"
feature_npy_path = args.feature_npy_path

# Generation parameters
scale=3
h=512
w=512
ddim_steps=45
ddim_eta=0.0

model = load_model_from_config(config, ckpt, device)
sampler = DDIMSampler(model)

def find_nearest_encode(embedding, embeddings):
    if len(embedding.shape) == 3:
        embedding = embedding.reshape(embedding.shape[0], -1)
    
    if len(embeddings.shape) == 3:
        embeddings = embeddings.reshape(embeddings.shape[0], -1)
    similarities = cosine_similarity(embedding, embeddings)
    return np.argmax(similarities)

def load_encodes(file_paths):
    embeddings = []
    for file_path in file_paths:
        embeddings.append(np.load(file_path))
    return np.vstack(embeddings)

class FusionModel(nn.Module):
    def __init__(self, encode_size, feature_size):
        super(FusionModel, self).__init__()
        self.fc = nn.Linear(encode_size + feature_size, encode_size)

    def forward(self, encode, feature):
        # Unsqueeze feature to make it 2-dimensional
        feature = feature.unsqueeze(0)
        fused_feature = torch.cat((encode, feature), dim=1)
        result = self.fc(fused_feature)
        return result

'''
class FusionModel(nn.Module):
    def __init__(self, encode_size, feature_size):
        super(FusionModel, self).__init__()
        self.encode_size = encode_size
        self.conv = nn.Conv1d(1, 1, kernel_size=feature_size+1, stride=1)

    def forward(self, encode, feature):
        feature = feature.unsqueeze(0)
        combined = torch.cat((encode, feature), dim=1).unsqueeze(0)  # 添加 batch 和 channel 维度
        
        reduced_combined = self.conv(combined).squeeze(0)
        
        return reduced_combined
'''
fusion_model = FusionModel(encode_size=59136, feature_size=63).to(device)
fusion_model_ckpt = torch.load(fusion_model_ckpt_path, map_location=device)
fusion_model.load_state_dict(fusion_model_ckpt['fusion_model_state_dict'])
fusion_model.to(device)
fusion_model.eval()

# Define parameters
nums = 12
rows = 4
cols = 3
image_size = (400, 400)  # Adjust based on your images

# Load all npy encodes
#encode_file_paths = [os.path.join(encode_folder, f) for f in os.listdir(encode_folder)]
#encodes = load_encodes(encode_file_paths)

img_save_path = args.img_save_path
os.makedirs(img_save_path, exist_ok=True)

for num in tqdm(range(nums), desc="Processing images"):
    prompt = prompts[num]  
    torch.manual_seed(0)
    orig_emb = model.get_learned_conditioning([prompt])

    prompt_embedding = orig_emb.cpu().numpy()
    #nearest_idx = find_nearest_encode(prompt_embedding, encodes)
    #name = os.listdir(feature_folder)[nearest_idx]
    
    feature = torch.tensor(np.load(feature_npy_path), dtype=torch.float32).to(device)

    # Concatenate the gesture features with the original embeddings
    fused_feature = fusion_model(orig_emb.view(1, -1), feature)
    fused_feature = fused_feature.view(orig_emb.shape)

    #ema
    fused_feature = (1.0 - args.mu) * fused_feature + args.mu * orig_emb

    # Sample the model with a fixed code to see what it looks like
    quick_sample = lambda x, s, code: decode_to_im(sample_model(model, sampler, x, h, w, ddim_steps, s, ddim_eta, start_code=code))
    start_code = torch.randn((5, 4, 64, 64))
    img = quick_sample(fused_feature, scale, start_code)
    # Resize image to 100x100
    img = img.resize(image_size)
    save_path = f"{img_save_path}/{image_filenames[num]}"
    img.save(save_path)
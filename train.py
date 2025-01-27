from omegaconf import OmegaConf
import torch
from torch import nn
from PIL import Image
from torchvision import transforms
import os
from tqdm import tqdm
from einops import rearrange
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import argparse

import sys
sys.path.append('./stable-diffusion')

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config

parser = argparse.ArgumentParser(description='Train the model')
parser.add_argument('--gesture', type=str, required=True, help='Gesture name')
parser.add_argument('--Lambda', type=float, default=0.7, help='Hyperparameter Lambda')
args = parser.parse_args()

def load_model_from_config(config, ckpt, device="cpu", verbose=False):
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
    image = Image.open(path).convert("RGB")
    tform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
        ])
    image = tform(image)
    return 2.*image - 1.

def decode_to_im(samples, n_samples=1, nrow=1):
    samples = model.decode_first_stage(samples)
    ims = torch.clamp((samples + 1.0) / 2.0, min=0.0, max=1.0)
    x_sample = 255. * rearrange(ims.cpu().numpy(), '(n1 n2) c h w -> (n1 h) (n2 w) c', n1=n_samples//nrow, n2=nrow)
    img = Image.fromarray(x_sample.astype(np.uint8))
    return img

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
# Read image filenames and prompts from text files
def read_lines_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.read().splitlines()
    return lines
'''
def find_nearest_encode(embedding, embeddings):
    # check and reshape the embedding
    if len(embedding.shape) == 3:
        embedding = embedding.reshape(embedding.shape[0], -1)
    
    # check and reshape the embeddings
    if len(embeddings.shape) == 3:
        embeddings = embeddings.reshape(embeddings.shape[0], -1)
    similarities = cosine_similarity(embedding, embeddings)
    return np.argmax(similarities)
'''
def load_encodes(file_paths):
    embeddings = []
    for file_path in file_paths:
        embeddings.append(np.load(file_path))
    return np.vstack(embeddings)

image_folder = f"./{args.gesture}" 
image_filenames_file = f"./{args.gesture}_BLIP2_file_name.txt"
prompts_file = f"./{args.gesture}_BLIP2_modified.txt"
#encode_folder = "/home/ubuntu/disk1/18encodes" 
feature_npy_path = f'./{args.gesture}.npy'

image_filenames = read_lines_from_file(image_filenames_file)
prompts = read_lines_from_file(prompts_file)

nums = len(image_filenames)
device = "cuda:0"

config="./stable-diffusion/configs/stable-diffusion/v1-inference.yaml"
ckpt = "./stable-diffusion/models/ldm/stable-diffusion-v1/sd-v1-4-full-ema.ckpt"

scale=3
h=512
w=512
ddim_steps=45
ddim_eta=0.0

model = load_model_from_config(config, ckpt, device)
sampler = DDIMSampler(model)

fusion_model = FusionModel(encode_size=59136, feature_size=63).to(device)  # gesture feature size is 63

# Load all npy encodes
encode_file_paths = [os.path.join(encode_folder, f) for f in os.listdir(encode_folder)]
encodes = load_encodes(encode_file_paths)

for idx in tqdm(range(nums), desc="Processing images"):
    image_path = os.path.join(image_folder, image_filenames[idx])
    image = load_img(image_path).unsqueeze(0).to(device)
    prompt = prompts[idx]

    torch.manual_seed(0)

    init_latent = model.get_first_stage_encoding(model.encode_first_stage(image))
    orig_embs = model.get_learned_conditioning([prompt])

    prompt_embedding = orig_embs.cpu().numpy()
    #nearest_idx = find_nearest_encode(prompt_embedding, encodes)
    #name = os.listdir(feature_folder)[nearest_idx]
    
    #feature_npy_path = os.path.join(feature_folder, name)
    feature = torch.tensor(np.load(feature_npy_path), dtype=torch.float32).to(device)

    feature.requires_grad = False

    # Concatenation and FC mapping
    fused_feature_orig = fusion_model(orig_embs.view(1, -1), feature)
    fused_feature_orig = fused_feature_orig.view(orig_embs.shape)

    # linear interpolation
    fused_feature_1 = args.Lambda * orig_embs + (1.0 - args.Lambda) * fused_feature_orig
    fused_feature = fused_feature_1.clone().detach().requires_grad_(True)
    lr = 0.001
    it = 10
    opt = torch.optim.Adam([fused_feature], lr=lr)
    criteria = torch.nn.MSELoss()
    history = []

    start_code = torch.randn_like(init_latent)
    quick_sample = lambda x, s, code: decode_to_im(sample_model(model, sampler, x, h, w, ddim_steps, s, ddim_eta, start_code=code))
    pbar = tqdm(range(it), leave=False)
    # Text Embedding Optimization
    for i in pbar:
        opt.zero_grad()
        noise = torch.randn_like(init_latent)
        t_enc = torch.randint(1000, (1,), device=device)
        z = model.q_sample(init_latent, t_enc, noise=noise)

        pred_noise = model.apply_model(z, t_enc, fused_feature)
        loss = criteria(pred_noise, noise)
        loss.backward()
        history.append(loss.item())
        opt.step()

    fused_feature.requires_grad = False
    model.train()

    lr = 1e-6
    it = 20
    opt = torch.optim.Adam([{'params': model.model.parameters()}, {'params': fusion_model.parameters()}], lr=lr)
    criteria = torch.nn.MSELoss()
    history = []

    orig_latent = init_latent  # Use the fused feature

    pbar = tqdm(range(it), leave=False)
    # Stable Diffusion Fine-tuning
    for i in pbar:
        opt.zero_grad()
        noise = torch.randn_like(orig_latent)
        t_enc = torch.randint(model.num_timesteps, (1,), device=device)
        z = model.q_sample(orig_latent, t_enc, noise=noise)

        pred_noise = model.apply_model(z, t_enc, fused_feature)
        loss = criteria(pred_noise, noise)
        loss.backward()
        history.append(loss.item())
        opt.step()

    if (idx + 1) % 1000 == 0:
        ckpt_save_path = f"./model_finetuned.ckpt"
        torch.save({
            'state_dict': model.state_dict(),
            'fusion_model_state_dict': fusion_model.state_dict()
        }, ckpt_save_path)

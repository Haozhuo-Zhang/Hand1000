import torch
import numpy as np
from scipy.linalg import sqrtm
from torchvision import models, transforms
from torch.nn.functional import adaptive_avg_pool2d
from PIL import Image
import os
import argparse

parser = argparse.ArgumentParser(description='Calculate FID')
parser.add_argument('--real_images', type=str, required=True, help='Path to the folder containing real images')
parser.add_argument('--fake_images', type=str, required=True, help='Path to the folder containing fake images')
args = parser.parse_args()

def get_activations(images, model, batch_size=50, dims=2048, device='cpu'):
    model.eval()
    pred_arr = np.empty((len(images), dims))

    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        batch = torch.stack(batch).to(device)
        with torch.no_grad():
            pred = model(batch)

        # Ensure the output is 4D (batch_size, channels, height, width)
        if pred.dim() == 2:
            pred = pred.unsqueeze(-1).unsqueeze(-1)

        pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
        pred_arr[i:i + batch_size] = pred.cpu().data.numpy().reshape(pred.size(0), -1)

    return pred_arr

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    
    # Handling imaginary parts from the sqrtm function
    if np.iscomplexobj(covmean):
        covmean = covmean.real + eps * np.eye(covmean.shape[0])
    
    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))

def calculate_fid(real_images, fake_images, model, batch_size=50, device='cpu'):
    act1 = get_activations(real_images, model, batch_size, device=device)
    act2 = get_activations(fake_images, model, batch_size, device=device)

    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)

    fid_value = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    return fid_value

# Example usage
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1).to(device)
model.fc = torch.nn.Identity()

# Define transformations
transform = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_images_from_folder(folder, transform):
    images = []
    for filename in os.listdir(folder):
        img = Image.open(os.path.join(folder, filename))
        if img is not None:
            images.append(transform(img))
    return images

real_images_folder = args.real_images
fake_images_folder = args.fake_images

real_images = load_images_from_folder(real_images_folder, transform)
fake_images = load_images_from_folder(fake_images_folder, transform)

fid_value = calculate_fid(real_images, fake_images, model, device=device)
print(f"FID: {fid_value}")

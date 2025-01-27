import torch
import numpy as np
from torchvision import models, transforms
from torch.nn.functional import adaptive_avg_pool2d
from PIL import Image
import os
import argparse

parser = argparse.ArgumentParser(description='Calculate KID Score')
parser.add_argument('--real_images', type=str, required=True, help='Path to the folder containing real images')
parser.add_argument('--fake_images', type=str, required=True, help='Path to the folder containing fake images')
args = parser.parse_args()

def polynomial_kernel(X, Y, degree=3, gamma=None, coef0=1):
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    K = (gamma * np.dot(X, Y.T) + coef0) ** degree
    return K

def get_activations(images, model, batch_size=50, dims=2048, device='cpu'):
    model.eval()
    pred_arr = np.empty((len(images), dims))

    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        batch = torch.stack(batch).to(device)
        with torch.no_grad():
            pred = model(batch)

        if pred.dim() == 2:
            pred = pred.unsqueeze(-1).unsqueeze(-1)

        pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
        pred_arr[i:i + batch_size] = pred.cpu().data.numpy().reshape(pred.size(0), -1)

    return pred_arr

def calculate_kid(real_activations, fake_activations, num_subsets=100, subset_size=1000):
    m = min(len(real_activations), len(fake_activations))
    m = min(m, subset_size)

    kid_values = []
    for _ in range(num_subsets):
        idx_real = np.random.choice(len(real_activations), m, replace=False)
        idx_fake = np.random.choice(len(fake_activations), m, replace=False)

        X = real_activations[idx_real]
        Y = fake_activations[idx_fake]

        K_XX = polynomial_kernel(X, X)
        K_YY = polynomial_kernel(Y, Y)
        K_XY = polynomial_kernel(X, Y)

        kid_value = K_XX.mean() + K_YY.mean() - 2 * K_XY.mean()
        kid_values.append(kid_value)

    return np.mean(kid_values), np.std(kid_values)

def calculate_kid_score(real_images, fake_images, model, batch_size=50, device='cpu'):
    real_activations = get_activations(real_images, model, batch_size, device=device)
    fake_activations = get_activations(fake_images, model, batch_size, device=device)

    kid_mean, kid_std = calculate_kid(real_activations, fake_activations)
    return kid_mean, kid_std

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

kid_mean, kid_std = calculate_kid_score(real_images, fake_images, model, device=device)
print(f"KID mean: {kid_mean}, KID std: {kid_std}")

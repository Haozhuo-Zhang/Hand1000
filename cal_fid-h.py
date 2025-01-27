import cv2
import numpy as np
import torch
from scipy.linalg import sqrtm
from torchvision import models, transforms
from torch.nn.functional import adaptive_avg_pool2d
from PIL import Image
import os
import mediapipe as mp
import argparse

parser = argparse.ArgumentParser(description='Calculate FID-H')
parser.add_argument('--real_images', type=str, required=True, help='Path to the folder containing real images')
parser.add_argument('--fake_images', type=str, required=True, help='Path to the folder containing fake images')
args = parser.parse_args()

# Initialize Mediapipe hand detection model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

def detect_and_crop_hand(image):
    img_rgb = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if not results.multi_hand_landmarks:
        return None  # No hand detected

    x_min, y_min, x_max, y_max = 1, 1, 0, 0
    for hand_landmarks in results.multi_hand_landmarks:
        for landmark in hand_landmarks.landmark:
            x_min = min(x_min, landmark.x)
            y_min = min(y_min, landmark.y)
            x_max = max(x_max, landmark.x)
            y_max = max(y_max, landmark.y)
    
    h, w, _ = img_rgb.shape
    x_min = max(0, int(x_min * w))
    y_min = max(0, int(y_min * h))
    x_max = min(w, int(x_max * w))
    y_max = min(h, int(y_max * h))
    
    cropped_img = img_rgb[y_min:y_max, x_min:x_max]
    return Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR))

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

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    
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

transform = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_and_crop_images_from_folder(folder, transform):
    images = []
    for filename in os.listdir(folder):
        img = Image.open(os.path.join(folder, filename))
        cropped_img = detect_and_crop_hand(img)
        if cropped_img is not None:
            images.append(transform(cropped_img))
    return images

real_images_folder = args.real_images
fake_images_folder = args.fake_images

real_images = load_and_crop_images_from_folder(real_images_folder, transform)
fake_images = load_and_crop_images_from_folder(fake_images_folder, transform)

fid_value = calculate_fid(real_images, fake_images, model, device=device)
print(f"FID-H: {fid_value}")

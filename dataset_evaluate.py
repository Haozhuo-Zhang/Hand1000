import torch
import clip
from PIL import Image
import numpy as np
from tqdm import tqdm
import os
import argparse

parser = argparse.ArgumentParser(description='Dataset Evaluation')
parser.add_argument('--image_folder', type=str, required=True, help='Path to the folder containing images')
parser.add_argument('--image_filenames_path', type=str, required=True, help='Path to the file containing image filenames')
parser.add_argument('--captions_path', type=str, required=True, help='Path to the file containing captions')
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocess = clip.load("ViT-B/32", device=device)

image_folder = args.image_folder
image_filenames_path = args.image_filenames_path
captions_path = args.captions_path

with open(image_filenames_path, "r") as f:
    image_filenames = f.readlines()

with open(captions_path, "r") as f:
    captions = f.readlines()

image_filenames = [filename.strip() for filename in image_filenames[:5000]]
captions = [caption.strip() for caption in captions[:5000]]

image_features_list = []
for filename in tqdm(image_filenames, desc="Processing Images"):
    image_path = os.path.join(image_folder, filename)
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
    image_features_list.append(image_features)

image_features_tensor = torch.cat(image_features_list)

text_tokens = clip.tokenize(captions).to(device)
with torch.no_grad():
    text_features_tensor = model.encode_text(text_tokens)

# calculate the cosine similarity between the image and text features
cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
similarities = cosine_similarity(image_features_tensor, text_features_tensor)
similarity1 = similarities.mean().item()

# calculate the cosine similarity between the text features
similarity_matrix = cosine_similarity(text_features_tensor.unsqueeze(1), text_features_tensor.unsqueeze(0))
similarity2 = similarity_matrix.mean().item()

print("Similarity 1 (image-caption average similarity):", similarity1)
print("Similarity 2 (caption-caption average similarity):", similarity2)

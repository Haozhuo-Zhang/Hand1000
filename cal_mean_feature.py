import cv2
import mediapipe as mp
import numpy as np
from glob import glob
from tqdm import tqdm
import os
import argparse

parser = argparse.ArgumentParser(description='Calculate Mean Hand Feature')
parser.add_argument('--gesture', type=str, required=True, help='Path to the folder containing images')
args = parser.parse_args()

# init mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# function to extract hand features
def extract_features(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            features = []
            for landmark in hand_landmarks.landmark:
                features.extend([landmark.x, landmark.y, landmark.z])
            return features
    return None

# define the img folder
folder = f'./{args.gesture}'

# read all images in the folder
image_paths = glob(os.path.join(folder, '*.jpg'))

# extract features from all images
all_features = []
for image_path in tqdm(image_paths, desc=f"Processing {args.gesture}"):
    features = extract_features(image_path)
    if features:
        all_features.append(features)

# calculate the mean of all features
if all_features:
    all_features = np.array(all_features)
    mean_features = np.mean(all_features, axis=0)

    # save the mean features
    output_path = f'./{args.gesture}.npy'
    np.save(output_path, mean_features)
    print(f"Mean features saved to '{output_path}'")
else:
    print(f"No hand landmarks detected in the images of {args.gesture}.")

# release mediapipe hands
hands.close()

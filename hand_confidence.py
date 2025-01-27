import os
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='Compute Hand Confidence')
parser.add_argument('--image_folder', type=str, required=True, help='Path to the folder containing images')
args = parser.parse_args()

def load_images(image_folder):
    images = []
    for img_file in os.listdir(image_folder):
        img_path = os.path.join(image_folder, img_file)
        image = cv2.imread(img_path)
        if image is not None:
            images.append(image)
    return images

def detect_hands_and_compute_confidence(images):
    mp_hands = mp.solutions.hands
    confidences = []
    with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.0) as hands:
        for image in tqdm(images):
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if results.multi_hand_landmarks:
                for hand_landmarks, hand_landmark in zip(results.multi_hand_landmarks, results.multi_handedness):
                    confidence = hand_landmark.classification[0].score
                    confidences.append(confidence)
    return confidences

def main(image_folder):
    images = load_images(image_folder)
    confidences = detect_hands_and_compute_confidence(images)
    
    if len(confidences) == 0:
        print("No hands detected in the images.")
        return
    
    average_confidence = np.mean(confidences)
    print("Average confidence:", average_confidence)

if __name__ == "__main__":
    image_folder = args.image_folder
    main(image_folder)

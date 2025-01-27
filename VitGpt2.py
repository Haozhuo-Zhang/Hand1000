import argparse
from PIL import Image
import torch
import os
from tqdm import tqdm
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

parser = argparse.ArgumentParser(description='Image Captioning with VitGpt2')
parser.add_argument('--gesture', type=str, required=True, help='gesture name')
args = parser.parse_args()

# folder_path of images
folder_path = f"./{args.gesture}"
output_file = f"./{args.gesture}_VitGpt2.txt"
output_file_name = f"./{args.gesture}_VitGpt2_file_name.txt"
# get all image files in the folder
image_files_unsorted = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
image_files = sorted(image_files_unsorted, key=lambda x: os.path.basename(x))

# Save sorted file names to output_file_name
with open(output_file_name, 'w') as file_name_file:
    for image_file in image_files:
        file_name_file.write(os.path.basename(image_file) + "\n")

model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda:0")
model.to(device)

max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
def predict_step(url):
    images = []
    
    image = Image.open(url).convert('RGB') 
    images.append(image)

    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = model.generate(pixel_values, **gen_kwargs)

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds

def process_image(image_path):
    pred = predict_step(image_path)
    generated_text = pred[0]
    return generated_text

if __name__ == '__main__':
    with open(output_file, 'w') as file:
        for image_file in tqdm(image_files, desc='Processing images'):
            result = process_image(image_file)
            file.write(result + "\n")
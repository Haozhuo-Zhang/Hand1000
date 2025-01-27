import argparse
from transformers import AutoProcessor
from PIL import Image
import torch
import os
from multiprocessing import Pool, set_start_method
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch
from tqdm.contrib.concurrent import process_map 

parser = argparse.ArgumentParser(description='Image to text using BLIP2')
parser.add_argument('--gesture', type=str, required=True, help='gesture name')
args = parser.parse_args()

# folder_path of images
folder_path = f"./{args.gesture}"
output_file = f"./{args.gesture}_BLIP2.txt"
output_file_name = f"./{args.gesture}_BLIP2_file_name.txt"
# get all image files in the folder
image_files_unsorted = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
image_files = sorted(image_files_unsorted, key=lambda x: os.path.basename(x))

# Save sorted file names to output_file_name
with open(output_file_name, 'w') as file_name_file:
    for image_file in image_files:
        file_name_file.write(os.path.basename(image_file) + "\n")

# try to use spawn method for multiprocessing
try:
    set_start_method('spawn')
except RuntimeError:
    pass

processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
device = "cuda:0"
model.to(device)

def process_image(image_path):
    image = Image.open(image_path).convert('RGB')  
    image.resize((596, 437))
    inputs = processor(image, return_tensors="pt").to(device, torch.float16)
    generated_ids = model.generate(**inputs)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return generated_text

if __name__ == '__main__':
    with open(output_file, 'w') as file:
        results = process_map(process_image, image_files, max_workers=8, chunksize=1, desc='Processing images')
        for result in results:
            file.write(result + "\n")
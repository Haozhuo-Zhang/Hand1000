import argparse
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from PIL import Image
import torch
import os
from tqdm import tqdm
from multiprocessing import Pool, set_start_method

parser = argparse.ArgumentParser(description='Image Captioning with PaliGemma')
parser.add_argument('--gesture', type=str, required=True, help='gesture name')
args = parser.parse_args()

# try to use spawn method for multiprocessing
try:
    set_start_method('spawn')
except RuntimeError:
    pass

# model initialization
model_id = "google/paligemma-3b-mix-224"
device = "cuda:0"
dtype = torch.bfloat16
model = PaliGemmaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=dtype,
    device_map=device,
    revision="bfloat16",
).eval()
processor = AutoProcessor.from_pretrained(model_id)

# folder_path of images
folder_path = f"./{args.gesture}"
output_file = f"./{args.gesture}_PaliGemma.txt"
output_file_name = f"./{args.gesture}_PaliGemma_file_name.txt"

# get all image files in the folder
image_files_unsorted = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
image_files = sorted(image_files_unsorted, key=lambda x: os.path.basename(x))

# Save sorted file names to output_file_name
with open(output_file_name, 'w') as file_name_file:
    for image_file in image_files:
        file_name_file.write(os.path.basename(image_file) + "\n")

def process_image(image_path):
    image = Image.open(image_path)
    prompt = "caption"
    model_inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    input_len = model_inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        generation = model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
        generation = generation[0][input_len:]
        decoded = processor.decode(generation, skip_special_tokens=True)
    return decoded

if __name__ == '__main__':
    total_images = len(image_files)
    progress_interval = total_images
    progress_counter = 0
    with open(output_file, 'w') as file:
        with Pool(processes=10) as pool:
            for result in tqdm(pool.imap_unordered(process_image, image_files), total=total_images, desc='Processing images'):
                file.write(result + "\n")
                progress_counter += 1

import os
import librosa
import numpy as np
from PIL import Image, ImageDraw
from datasets import load_dataset
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool

def waveform_to_image(waveform, height, width):
    """
    Converts a 1D waveform into a 2D image array with a solid wave shape.
    """
    # Create an empty (white) image
    img = np.full((height, width), 255, dtype=np.uint8)
    
    # Find the center line of the image
    center_line = height // 2
    
    # Normalize and scale the waveform to fit the image height
    # We scale it based on the half-height to plot above and below the center line
    max_amp = np.max(np.abs(waveform))
    if max_amp == 0: max_amp = 1 # Avoid division by zero for silent audio
    scaled_waveform = (waveform / max_amp) * (height / 2 - 1)

    # Downsample the waveform to match the image width
    indices = np.linspace(0, len(waveform) - 1, width, dtype=int)
    sampled_waveform = scaled_waveform[indices]
    
    # Draw a vertical line for each point in the sampled waveform
    for x, amp in enumerate(sampled_waveform):
        amp_int = int(amp)
        # Determine the start and end of the line on the y-axis
        y_start = min(center_line, center_line - amp_int)
        y_end = max(center_line, center_line - amp_int)
        
        # Ensure the line is at least 1 pixel thick even for small amplitudes
        if y_start == y_end: y_end +=1
            
        # Set the pixels for the vertical line to black
        img[y_start:y_end, x] = 0
            
    return img

def process_and_save(item, root_dir, img_size):
    """
    Worker function for processing one audio file.
    """
    try:
        # 1. Get audio data and class name
        waveform = item['audio']['array']
        class_name = item['class']
        original_filename = os.path.splitext(item['slice_file_name'])[0]
        
        # 2. Create class directory
        class_dir = os.path.join(root_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        # 3. Generate waveform image array
        img_array = waveform_to_image(waveform, height=img_size[0], width=img_size[1])
        
        # 4. Save using Pillow (much faster than matplotlib)
        pil_img = Image.fromarray(img_array)
        image_path = os.path.join(class_dir, f"{original_filename}.png")
        pil_img.save(image_path)
    except Exception as e:
        # This helps to debug if one file fails without stopping the whole process
        print(f"Failed to process {item.get('slice_file_name', 'N/A')}: {e}")

if __name__ == '__main__':
    IMAGE_SAVE_DIRECTORY = "./data/urbansound_waveforms"
    IMAGE_SIZE = (128, 384) # Height, Width

    print("Loading UrbanSound8K dataset...")
    full_dataset_dict = load_dataset('danavery/urbansound8K')
    train_split = full_dataset_dict['train']
    dataset_list = [item for item in train_split]

    # Create a partial function to pass fixed arguments to the worker
    worker_func = partial(process_and_save, root_dir=IMAGE_SAVE_DIRECTORY, img_size=IMAGE_SIZE)
    
    # Get number of available CPUs
    num_processes = os.cpu_count()
    print(f"Starting pre-processing on {num_processes} cores...")

    # Create a multiprocessing pool and run the jobs
    with Pool(processes=num_processes) as pool:
        # Use tqdm to show a progress bar for the parallel processing
        list(tqdm(pool.imap(worker_func, dataset_list), total=len(dataset_list)))

    print(f"✅ Pre-processing complete. Images saved to '{IMAGE_SAVE_DIRECTORY}'")
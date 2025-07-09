import torch
from torch.utils.data import Dataset, DataLoader # Correct: Import Dataset from torch.utils.data
from torchvision import transforms
from datasets import load_dataset # Correct: Only import load_dataset here
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import io


# classifier = EncoderClassifier.from_hparams(
#     source="speechbrain/urbansound8k_ecapa",
#     savedir="pretrained_models/gurbansound8k_ecapa",
# )

# out_prob, score, index, text_lab = classifier.classify_file(
#     "speechbrain/urbansound8k_ecapa/dog_bark.wav"
# )

# print(out_prob, score, index, text_lab)


full_dataset = load_dataset('danavery/urbansound8K')
print(type(full_dataset))
print(full_dataset.shape) # returns {'train': (8732, 9)}
print(full_dataset)
print(full_dataset['train'].num_rows)
print(len(full_dataset['train']))
print(full_dataset['train'][0])
# load data
# preprocess data
# check shape
# divide into train and test
# set up dataloader

def save_waveform_image(y, sr, image_path):
    """
    Loads a WAV file, plots its waveform, and saves it as an image.

    Args:
        wav_path (str): The file path for the input WAV file.
        image_path (str): The file path to save the output PNG image.
    """
    # Load the audio file
    # y, sr = librosa.load(wav_path)

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 4))
    librosa.display.waveshow(y, sr=sr, ax=ax, color='blue')

    # Make the image clean by removing axes, labels, and extra space
    ax.axis('off')
    fig.tight_layout(pad=0)

    # Save the figure to the specified path
    fig.savefig(image_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig) # Close the plot to free up memory

class UrbanSoundWaveformDataset(Dataset):
    """
    Custom PyTorch Dataset for generating waveform images on the fly.
    """
    def __init__(self, hf_dataset, transform=None):
        """
        Args:
            hf_dataset: A Hugging Face dataset split (e.g., full_dataset['train']).
            transform (callable, optional): A torchvision transform pipeline.
        """
        print(f"DEBUG: The type of 'hf_dataset' passed to __init__ is: {type(hf_dataset)}")
        print(f"DEBUG: The value of 'hf_dataset' is: {hf_dataset}")
        
        
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        # 1. Get the raw audio data and label
        item = self.hf_dataset[idx]
        waveform = item['audio']['array']
        sample_rate = item['audio']['sampling_rate']
        label = item['classID']

        # 2. Generate the waveform plot IN MEMORY
        fig, ax = plt.subplots(figsize=(6, 2)) # Smaller figsize is faster
        librosa.display.waveshow(waveform, sr=sample_rate, ax=ax, color='blue')
        ax.axis('off')
        fig.tight_layout(pad=0)

        # 3. Convert the plot to a NumPy array (the "image")
        # Instead of saving, we draw to a buffer and read it as an array
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        image = plt.imread(buf) # Reads the PNG buffer into a NumPy array
        plt.close(fig) # IMPORTANT: close the figure to free memory

        # 4. Apply transformations to create the final tensor
        if self.transform:
            image_tensor = self.transform(image)
        else:
            # If no transform is provided, just convert to a tensor
            image_tensor = transforms.ToTensor()(image)

        return image_tensor, torch.tensor(label, dtype=torch.long)

def get_urbandsounds_dataloaders(batch_size=128):
    """Makes train and test dataloaders

    This function downloads urbansounds dataset,
    converts WAV files from urbansounds dataset into Pytorch tensors,
    loads the training and testing parts
    creates the two dataloaders and returns them.

    Parameters
    ----------
    Batch_size is set to 128, because otherwise dataloaders default to 1

    Returns
    ----------
    (Training Dataloader, Testing Dataloader)

    """
    #transform = transforms.Compose([
    #transforms.Resize((64, 64)),
    #transforms.ToTensor(),
    #transforms.Normalize((0.1307,), (0.3081,)),
    #])
    # transforms input images from PIL format to PyTorch tensors.
    # uses .Compose from the transforms module to create a transformation pipeline

    # Create datasets:

    full_dataset = load_dataset('danavery/urbansound8K')

    # Create dataloader:

    train_dataloader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)
 
    # return dataloaders:
    return train_dataloader



if __name__ == "__main__":
    # Define the image transformation pipeline for your CNN
    image_transforms = transforms.Compose([
        transforms.ToTensor(),
        # Resize to a standard size for your CNN
        transforms.Resize((128, 384), antialias=True),
        # Matplotlib creates an RGBA image (4 channels). We keep the first 3 (RGB).
        transforms.Lambda(lambda x: x[:3, :, :]),
        # Normalize with standard values for image models
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load the base dataset from Hugging Face
    full_dataset_dict = load_dataset('danavery/urbansound8K')
    train_split = full_dataset_dict['train']

    # Create an instance of your custom on-the-fly dataset
    train_dataset = UrbanSoundWaveformDataset(
        hf_dataset=train_split,
        transform=image_transforms
    )

    # Create the DataLoader. This will now generate images as needed.
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=4)

    print("✅ Dataloader is ready.")
    print("Fetching one batch to test the pipeline...")

    # Test the dataloader
    images, labels = next(iter(train_dataloader))

    print(f"\nShape of one batch of image tensors: {images.shape}")
    print(f"Shape of one batch of labels: {labels.shape}")
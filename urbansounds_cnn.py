from urbansounds_data import UrbanSoundWaveformDataset
from torchvision.models import resnet18
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.nn as nn
import os
from tqdm import tqdm 


def get_resnet18():
    
    ''' Get and set up ResNet model
    
    Function to load a resnet model and make adjustments to first 
    convolutional layer to reduce to 1 channel (grayscale) instead of RGB
    
    Returns: 
    adjusted model 
    
    '''
    
    model = resnet18(num_classes=10)  # 10 classes for MNIST
    # Change first conv layer to accept 1 channel (instead of 3 for RGB)
    #model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    return model

def train_model(model, train_loader, epochs=5, batch_size=32,device=None, model_path="models/resnet18_urbansounds.pth"):
    
    '''Training Function 
    
    Loops training data in batches of 128 images,
    applies transformations to each image,
    trains the model for specified number of epochs,
    calculates loss, back propogates and then optomises weights 
    saves the updated parameters   
    
    '''
    
    # image_transforms = transforms.Compose([
    #     transforms.ToTensor(),
    #     # Resize to a standard size for your CNN
    #     transforms.Resize((128, 384), antialias=True),
    #     # Matplotlib creates an RGBA image (4 channels). We keep the first 3 (RGB).
    #     transforms.Lambda(lambda x: x[:3, :, :]),
    #     # Normalize with standard values for image models
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # ])
    # transforms input images from PIL format to PyTorch tensors.
    # uses .Compose from the transforms module to create a transformation pipeline
    
    # Set device
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    # assign device if GPU is available use it, if not use cpu
    
    #train_loader = UrbanSoundWaveformDataset(batch_size)
    # get dataloaders with transformation applied 
    
    model = get_resnet18().to(device) 
    # initialise model and move to device 
    criterion = nn.CrossEntropyLoss()
    # set loss function 
    optimizer = torch.optim.Adam(model.parameters())
    # set optomiser, use adam 
    
    #start training loop 
    for epoch in range(epochs):
        
        # initialise model in training mode
        model.train()
        # initalise running_loss, correct, total all to 0
        running_loss, correct, total = 0.0, 0, 0
        
        # loop over images & labels from each batch
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        
        # loop over the training data
        for images, labels in loop:
            
            # move images and labels to device
            images, labels = images.to(device), labels.to(device)
            
            # reset gradients to zero before new backward pass
            optimizer.zero_grad()
            
            # peform a forward pass on current batch images
            outputs = model(images)
            
            # calculate loss between predicted and true labels
            loss = criterion(outputs, labels)
            
            # back propogate the loss - compute gradients for each parameter
            loss.backward()
            
            # update weights using Adam optimizer
            optimizer.step()
            
            # keep running total of accumlated loss, correct predictions, and total samples 
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            loop.set_postfix(loss=running_loss / (total / batch_size), acc=100. * correct / total)
            
            # print loss and accuracy
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {running_loss / len(train_loader):.4f} - Accuracy: {100 * correct / total:.2f}%")
       
    # save the model after 12 epochs
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    


if __name__ == "__main__":
    # This code will now ONLY run if you execute "python3 urbansounds_data.py"
    # It will NOT run when imported by another script.
    print("Testing the model loader function...")
    dummy_model = get_resnet18()
    print(type(dummy_model))
    
    image_transforms = transforms.Compose([
        transforms.ToTensor(),
        # Resize to a standard size for your CNN
        transforms.Resize((128, 384), antialias=True),
        # Matplotlib creates an RGBA image (4 channels). We keep the first 3 (RGB).
        transforms.Lambda(lambda x: x[:3, :, :]),
        # Normalize with standard values for image models
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    print("Preparing data...")
    # Define transforms
    # image_transforms = transforms.Compose([ ... ]) # Your transforms

    # Correctly create the dataset
    # full_dataset_dict = load_dataset('danavery/urbansound8K')
    # train_dataset = UrbanSoundWaveformDataset(
    #     hf_dataset=full_dataset_dict['train'],
    #     transform=image_transforms
    # )    
    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    train_dataset = datasets.ImageFolder(root="data/urbansound_waveforms", transform=image_transforms)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    model = get_resnet18()
    print("Starting training...")
    train_model(model=model, train_loader=train_loader, epochs=5)
    
    
    
    
    
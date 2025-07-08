MLX_WEEK5_AUDIO

Pre-requisities

- Python 3.10
- conda - install conda 25.5.1 via https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html

## Conda Set Up ##

1. $ conda update -n base conda
2. $ conda env create -f environment.yml
3. $ conda remove --name <myenv> --all
4. $ conda config --set auto_activate_base false

To add a dependency (package / libarary) : 

1. check if its contained in conda::forge
2. add to enviroment.yml in appropriate place
3. $ conda env update --file environment.yml --prune

TO DO 

spin up GPU
ssh into remote server 
set up kaggle json file on remote gpu  
write script for downloading and preparing data  
execute script 
write training script for CNN 
write training script for Transformer 
save both models to weights and biases 

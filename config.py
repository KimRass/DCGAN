from image_utils import get_image_dataset_mean_and_std

### Data
IMG_SIZE = 64
LATENT_DIM = 100
# mean, std = get_image_dataset_mean_and_std(args.data_dir)
MEAN = (0.506, 0.425, 0.383)
STD = (0.311, 0.291, 0.289)

### Optimizer
# "We used the Adam optimizer with tuned hyperparameters. We used 0.0002 for learning rate. We found
# reducing $\beta_{1}$ to 0.5 helped stabilize training."
LR = 0.0002
BETA1 = 0.5
BETA2 = 0.999

### Data
IMG_SIZE = 64
LATENT_DIM = 100

### Optimizer
# "We used the Adam optimizer with tuned hyperparameters. We used 0.0002 for learning rate. We found
# reducing $\beta_{1}$ to 0.5 helped stabilize training."
LR = 0.0002
BETA1 = 0.5
BETA2 = 0.999

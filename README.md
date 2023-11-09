# 'DCGAN' (Radford et al., 2016) implementation from scratch in PyTorch
- [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://github.com/KimRass/DCGAN/blob/main/unsupervised_representation_learning_with_deep_convolutional_generative_adversarial_networks.pdf)
## Pre-trained Models
- Trained on CelebA dataset for 30 epochs
    - [dcgan_celeba.pth](https://drive.google.com/file/d/1dgC2lhIN-Qf_JvN2rz77brHHru13DFNy/view?usp=sharing)
## Generated Image Samples (CelebA 64×64)
```bash
# e.g.,
    python3 generate_images.py\
    --ckpt_path="/Users/jongbeomkim/Documents/dcgan/dcgan_celeba.pth"\
    --batch_size=100\
    --n_images=10\
    --n_cpus=0 # Optional
```
- <img src="https://github.com/KimRass/DCGAN/assets/105417680/6d3f0276-0448-4330-a3a7-1b7489b0d21b" width="600">
- Using interpolation
    ```bash
    # e.g.,
        python3 interpolate.py\
        --ckpt_path="/Users/jongbeomkim/Documents/dcgan/dcgan_celeba.pth"\
        --batch_size=10\
        --n_images=10\
        --n_cpus=0 # Optional
    ```
    - <img src="https://github.com/KimRass/DCGAN/assets/105417680/88e6751f-58f5-4291-9201-b10a2df65f95" width="600">

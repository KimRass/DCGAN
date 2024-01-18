# 1. Pre-trained Model
- Trained on CelebA dataset for 30 epochs
    - [dcgan_celeba.pth](https://drive.google.com/file/d/1dgC2lhIN-Qf_JvN2rz77brHHru13DFNy/view?usp=sharing)

# 2. Sampling
```bash
# e.g.,
python3 sample.py\
    --ckpt_path="/.../dcgan/dcgan_celeba.pth"\
    --n_cells=100\
    --n_iters=10\
    --n_cpus=0 # Optional
```
- <img src="https://github.com/KimRass/DCGAN/assets/105417680/6d3f0276-0448-4330-a3a7-1b7489b0d21b" width="600">

# 3. Interpolation
```bash
# e.g.,
python3 interpolate.py\
    --ckpt_path="/.../dcgan/dcgan_celeba.pth"\
    --n_cells=10\
    --n_iters=10\
    --n_cpus=0 # Optional
```
- <img src="https://github.com/KimRass/DCGAN/assets/105417680/88e6751f-58f5-4291-9201-b10a2df65f95" width="600">

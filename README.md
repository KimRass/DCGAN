# 1. Pre-trained Model
- [dcgan_celeba.pth](https://drive.google.com/file/d/1dgC2lhIN-Qf_JvN2rz77brHHru13DFNy/view?usp=sharing)

# 2. Sampling
## 1) `"grid"` mode
```bash
# e.g.,
python3 sample.py\
    --mode="grid"\
    --ckpt_path="/.../dcgan/dcgan_celeba.pth"\
    --batch_size=100\
    --n_iters=10\
    --n_cpus=0 # Optional
```
- <img src="https://github.com/KimRass/DCGAN/assets/105417680/6d3f0276-0448-4330-a3a7-1b7489b0d21b" width="600">
## 2) `"interpolation"` mode
```bash
# e.g.,
python3 sample.py\
    --mode="interpolation"\
    --ckpt_path="/.../dcgan/dcgan_celeba.pth"\
    --n_cells=10\
    --n_iters=10\
    --n_cpus=0 # Optional
```
- <img src="https://github.com/KimRass/KimRass/assets/67457712/16cdcfea-9fca-41de-a4bd-147b9d5eae5f" width="600">

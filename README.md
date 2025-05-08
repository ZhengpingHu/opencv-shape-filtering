# CVAE Frame Collection and Training

This repository provides tools for collecting frames from a running environment, training an enhanced Convolutional Variational Autoencoder (CVAE) on the collected data, and visualizing reconstruction results.

## Table of Contents

1. [Prerequisites](#prerequisites)  
2. [Installation](#installation)  
3. [Frame Collection](#frame-collection)  
4. [Enhanced CVAE Training](#enhanced-cvae-training)  
5. [Reconstruction Test](#reconstruction-test)  
6. [Quantitative Evaluation (Optional)](#quantitative-evaluation-optional)  
7. [Git Ignore](#git-ignore)  

## Prerequisites

- Python 3.8+  
- PyTorch  
- torchvision  
- h5py  
- numpy  
- matplotlib  
- tqdm  
- pytorch-msssim  
- Optional: CUDA toolkit for GPU acceleration  

## Installation

```bash
git clone https://github.com/ZhengpingHu/opencv-shape-filtering.git
cd opencv-shape-filtering

# Create a Python virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch torchvision h5py numpy matplotlib tqdm pytorch-msssim
```

## Frame Collection

Capture frames from a running environment (e.g. a Pygame window) and save them to disk:

```bash
python3 frame_collector.py \
  --launch-env \
  --env-script env.py \
  --window "pygame window" \
  --out ./frames \
  --man ./manifest.txt \
  --max 2000000 \
  --fps 5.0
```

- `--launch-env` / `--env-script`: Start the environment script before capturing.  
- `--window`: Match window title.  
- `--out`: Directory to save frames.  
- `--man`: Manifest file listing saved frames.  
- `--max`: Maximum number of frames.  
- `--fps`: Capture frame rate.

## Enhanced CVAE Training

Train the improved CVAE with advanced techniques: deeper network, residual blocks, multi-loss (L1 + SSIM + perceptual + β-KLD), KL annealing, cosine LR scheduling, data augmentation, and train/validation split.

```bash
python3 train_improved.py \
  --h5 ./frames_gray.h5 \
  --save_dir ./models2 \
  --batch 32 \
  --epochs 100 \
  --lr 1e-3 \
  --max_beta 1.0 \
  --device cuda \
  --workers 4 \
  --fp16
```

- `--h5`: Path to the HDF5 dataset (e.g. `frames_gray.h5`).  
- `--save_dir`: Directory to save model checkpoints.  
- `--batch`, `--epochs`: Batch size and number of epochs.  
- `--lr`: Learning rate.  
- `--max_beta`: Final weight for the KLD term (β-VAE).  
- `--fp16`: Use mixed precision (float16) for training.

## Reconstruction Test

Visualize reconstruction results on a random frame:

```bash
python3 test_cvae.py \
  --h5     ./frames_gray.h5 \
  --enc    ./models2/enc_ep100.pth \
  --dec    ./models2/dec_ep100.pth \
  --in_ch  1 \
  --latent 64 \
  --device cuda \
  --outdir ./recon_plots
```

- `--in_ch`: Number of input channels (usually 1 for grayscale).  
- `--latent`: Latent dimensionality (must match training).  
- `--outdir`: Directory to save comparison images (original vs. reconstruction).

## Quantitative Evaluation (Optional)

Compute average MSE and PSNR on random samples:

```bash
python3 eval_metrics.py \
  --h5    ./frames_gray.h5 \
  --enc   ./models2/enc_ep100.pth \
  --dec   ./models2/dec_ep100.pth \
  --device cuda \
  --n      1000
```

## Git Ignore

Add the following to `.gitignore` to avoid committing large datasets and images:

```
*.h5
*.hdf5
*.pth
```

## Notice

Waiting to add the YOLO frame in the next step and a different branch.

## OBB

Model trained, saved at: ```runs\obb\train8\weights\best.pt```

Model prediction:

use command:

```bash
yolo obb predict model=runs/obb/train8/weights/best.pt source=MODIFY save=True save_txt=True

to save the extra angel information

## References:

[1] R. C. Gonzalez and R. E. Woods, _Digital Image Processing_, 4th ed. Pearson, 2018.

[2] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, “Image quality assessment: From error visibility to structural similarity,” _IEEE Trans. Image Process._, vol. 13, no. 4, pp. 600–612, Apr. 2004.

[3] Z. Wang, E. P. Simoncelli, and A. C. Bovik, “Multi-scale structural similarity for image quality assessment,” in _Proc. IEEE Asilomar Conf. Signals, Syst., Comput._, Pacific Grove, CA, Nov. 2003, vol. 2, pp. 1398–1402.

[4] J. Johnson, A. Alahi, and L. Fei-Fei, “Perceptual losses for real-time style transfer and super-resolution,” in _Proc. European Conf. Computer Vision (ECCV)_, 2016, pp. 694–711.

[5] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, “The unreasonable effectiveness of deep features as a perceptual metric,” in _Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)_, 2018, pp. 586–595.

[6] I. Higgins _et al._, “β-VAE: Learning basic visual concepts with a constrained variational framework,” in _Proc. International Conf. Learning Representations (ICLR)_, 2017.

[7] D. P. Kingma and M. Welling, “Auto-Encoding Variational Bayes,” in _Proc. International Conf. Learning Representations (ICLR)_, 2014.

[8] “pytorch-msssim: PyTorch implementations of SSIM and MS-SSIM,” GitHub, 2025. [Online]. Available: https://github.com/VainF/pytorch-msssim

[9] A. Q. Nguyen and M. K. Ng, “Deep learning based image quality assessment: A survey,” _IEEE Access_, vol. 9, pp. 148204–148232, 2021.

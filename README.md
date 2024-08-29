# Fast-GeCo: Noise-robust Speech Separation with Fast Generative Correction
[![Paper](https://img.shields.io/badge/arXiv-2406.07461-brightgreen.svg?style=flat-square)](https://arxiv.org/pdf/2406.07461.pdf)  [![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://arxiv.org/pdf/2406.07461.pdf) [![Demo page](https://img.shields.io/badge/Audio_Samples-blue?logo=Github&style=flat-square)](https://fastgeco.github.io/Fast-GeCo/)

In this paper, we propose a generative correction method to enhance the output of a discriminative separator. By leveraging a generative corrector based on a diffusion model, we refine the separation process for single-channel mixture speech by removing noises and perceptually unnatural distortions. Furthermore, we optimize the generative model using a predictive loss to streamline the diffusion model’s reverse process into a single step and rectify any associated errors by the reverse process. Our method achieves state-of-the-art performance on the in-domain Libri2Mix noisy dataset, and out-of-domain WSJ with a variety of noises, improving SI-SNR by 22-35% relative to SepFormer, demonstrating robustness and strong generalization capabilities.

## Environment setup

```
conda create -n geco python=3.8.19
conda activate geco
pip install -r requirments.txt
```


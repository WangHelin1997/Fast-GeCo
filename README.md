# Fast-GeCo: Noise-robust Speech Separation with Fast Generative Correction
[![Paper](https://img.shields.io/badge/arXiv-2406.07461-brightgreen.svg?style=flat-square)](https://arxiv.org/pdf/2406.07461.pdf)  [![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://arxiv.org/pdf/2406.07461.pdf) [![Demo page](https://img.shields.io/badge/Audio_Samples-blue?logo=Github&style=flat-square)](https://fastgeco.github.io/Fast-GeCo/)

In this paper, we propose a generative correction method to enhance the output of a discriminative separator. By leveraging a generative corrector based on a diffusion model, we refine the separation process for single-channel mixture speech by removing noises and perceptually unnatural distortions. Furthermore, we optimize the generative model using a predictive loss to streamline the diffusion model’s reverse process into a single step and rectify any associated errors by the reverse process. Our method achieves state-of-the-art performance on the in-domain Libri2Mix noisy dataset, and out-of-domain WSJ with a variety of noises, improving SI-SNR by 22-35% relative to SepFormer, demonstrating robustness and strong generalization capabilities.

<div style="text-align: center;">
    <img src="geco.webp" alt="Fast-GeCo Image" width="400"/>
</div>

## NEWS & TODO
Huggingface demos and pretrained models will be released soon!

## Environment setup

```
conda create -n geco python=3.8.19
conda activate geco
pip install -r requirements.txt
```

## Data Preparation

To train GeCo or Fast-GeCo, you should prepare a data folder in the following way:

```
libri2mix-train100/
    -1_mix.wav
    -1_source1.wav
    -1_source1hatP.wav
    -2_mix.wav
    -2_source1.wav
    -2_source1hatP.wav
    ....
```

Here,  `*_mix.wav` is the mixture audio, `*_source1.wav` is the grouth truth audio, and `*_source1hatP.wav` is the estimated audio by a speech separation model like SepFormer.


## Train GeCo
with 1 GPU, run:

```
CUDA_VISIBLE_DEVICES=0 python train_geco.py --gpus 1 --batch_size 16
```

## Train Fast-GeCo
with 1 GPU, run:

```
CUDA_VISIBLE_DEVICES=0 python train_fastgeco.py --gpus 1 --batch_size 32
```

## Evaluate GeCo

```
CUDA_VISIBLE_DEVICES=0 python eval-geco.py
```


## Evaluate Fast-GeCo

```
CUDA_VISIBLE_DEVICES=0 python eval-fastgeco.py
```


## Run baseline SepFormer

We also provide codes to train and evaluate the SepFormer model, the same as in our paper.

See [speechbrain](https://github.com/speechbrain) for more details of training and test.

## Citations & References
We kindly ask you to cite our paper in your publication when using any of our research or code:

```
misc{wang2024noiserobustspeechseparationfast,
      title={Noise-robust Speech Separation with Fast Generative Correction}, 
      author={Helin Wang and Jesus Villalba and Laureano Moro-Velazquez and Jiarui Hai and Thomas Thebaud and Najim Dehak},
      year={2024},
      eprint={2406.07461},
      archivePrefix={arXiv},
      primaryClass={eess.AS},
      url={https://arxiv.org/abs/2406.07461}, 
}
```


## Acknowledgement

[1] [speechbrain](https://github.com/speechbrain)

[2] [Conv-TasNet](https://github.com/JusperLee/Conv-TasNet)

[3] [sgmse-bbed](https://github.com/sp-uhh/sgmse-bbed)

[4] [sgmse-crp](https://github.com/sp-uhh/sgmse_crp)
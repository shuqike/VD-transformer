# VD-transformer

## Overview

This repository contains the PyTorch implementation of for VD-transformer motor control.

## Vision components

There are two alternative ways to learn visual representations: masked autoencoders (MAE) or vision transformers.

### Pre-trained vision encoders



### Pre-trained vision transformer

Cordonnier et al. (2020) theoretically proved that self-attention can learn to behave similarly to convolution. Empirically, patches were taken from images as inputs, but the small patch size makes the model only applicable to image data with low resolutions. Without specific constraints on patch size, vision Transformers (ViTs) extract patches from images and feed them into a Transformer encoder to obtain a global representation, which will finally be transformed for classification (Dosovitskiy et al., 2021). Notably, Transformers show better scalability than CNNs: when training larger models on larger datasets, vision Transformers outperform ResNets by a significant margin.

## Decision

Given the visual encoder, we train controllers on top with reinforcement learning or other control algorithms. We keep the visual representations frozen and do not perform any taskspecific fine-tuning of the encoder; all motor control tasks use the same visual representations.

## Adapter

## Reference

1. https://d2l.ai/chapter_attention-mechanisms-and-transformers/vision-transformer.html
2. Cordonnier, J.-B., Loukas, A., & Jaggi, M. (2020). On the relationship between self-attention and convolutional layers. International Conference on Learning Representations.
3. Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., … others. (2021). An image is worth 16x16 words: transformers for image recognition at scale. International Conference on Learning Representations.
4. https://github.com/alohays/awesome-visual-representation-learning-with-transformers
5. Xiao, T., Radosavovic, I., Darrell, T., Malik, J. (2022). Masked Visual Pre-training for Motor Control. https://arxiv.org/abs/2203.06173.
6. Radosavovic, T., Xiao, T., James, S., Abbeel, P., Malik, J., Darrell. T. (2022). Real-World Robot Learning with Masked Visual Pre-training. https://arxiv.org/abs/2210.03109.
7. Chen, Z., Duan, Y., Wang, W., He, J., Lu, T., Dai, J., Qiao, Y. (). Vision Transformer Adapter for Dense Predictions. https://arxiv.org/abs/2205.08534.

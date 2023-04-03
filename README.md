# VD-transformer

## Overview

This repository contains the PyTorch implementation of VD-transformer for motor control (MC) tasks.

## Benchmark suite

PixMC https://github.com/ir413/mvp/tree/master/pixmc

## Vision components

There are two alternative ways to learn general visual representations: masked autoencoders (MAE) or vision transformer adapters (ViT-Adapter).

### Pre-trained vision encoders (MAE)

MAEs mask-out random patches of the input image and reconstruct the missing pixels with a Vision Transformer (ViT).

### Pre-trained vision transformer adapter (ViT-Adapter)

This is an improvement of the MAE component. The plain ViT suffers inferior performance on dense predictions due to weak prior assumption. To address this issue, we use the ViT-Adapter. The backbone in its framework is a plain ViT that can learn powerful representations from large-scale multi-modal data. When transferring to downstream tasks, a pretraining-free adapter is used to introduce the image-related inductive biases into the model, making it suitable for these task.

## Decision

### MVP

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

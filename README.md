# VD-transformer

## Overview

This repository contains the PyTorch implementation of VD-transformer for motor control (MC) tasks.

Note: this README file is generated with the help of GPT-3.5.

## Benchmark suite

PixMC https://github.com/ir413/mvp/tree/master/pixmc

PixMC features a combination of two robotic arms and grippers:

1. Franka: A 7-DoF arm equipped with a 2-DoF gripper, commonly used for research, manufactured by Franka Emika.
2. Kuka with Allegro: A Kuka LBR iiwa arm that has 7 DoFs and a 4-finger Allegro hand with 16 DoFs (4 DoFs per finger), resulting in a total of 23 DoFs.

## Vision components

There are two alternative ways to learn general visual representations: masked autoencoders (MAE) or vision transformer adapters (ViT-Adapter).

### Pre-trained vision encoders (MAE)

MAEs leverage Vision Transformers (ViT) to reconstruct missing pixels in input images after masking out random patches. Their simplicity and minimal dependence on dataset-specific augmentation techniques make them an attractive choice.

### Pre-trained vision transformer adapter (ViT-Adapter)

The ViT-Adapter could provide an improvement over the MAE component. While the plain ViT can learn powerful representations from large-scale multi-modal data, it can suffer from weak prior assumptions, leading to inferior performance on dense predictions. To address this, we introduce a pretraining-free adapter into the plain ViT framework, allowing it to acquire image-related inductive biases necessary for downstream tasks.

## Decision components

Using a visual encoder as a starting point, we employ reinforcement learning or other control algorithms to train controllers. In doing so, we maintain the visual representations as they are, without any task-specific fine-tuning of the encoder. This approach provides two primary advantages. Firstly, it avoids the encoder from overfitting to the current setting, thereby preserving general visual representations that can facilitate learning new tasks. Secondly, it considerably reduces memory usage and run time since there is no need to perform back-propagation through the encoder.

### MVP

Using a model-free reinforcement learning approach called Proximal Policy Optimization (PPO), we train task-specific motor control policies on top of this embedding. PPO is a state-of-the-art policy gradient method that has demonstrated impressive results in tackling complex motor control tasks, and has been successfully applied to transfer learning in real-world hardware scenarios.

Our policy is implemented as a compact multi-layer perceptron (MLP) network. In addition, we train a critic that uses the same representations and has an identical architecture as the policy, but without sharing weights between them.

## Adapter

## Reference

1. https://d2l.ai/chapter_attention-mechanisms-and-transformers/vision-transformer.html
2. Cordonnier, J.-B., Loukas, A., & Jaggi, M. (2020). On the relationship between self-attention and convolutional layers. International Conference on Learning Representations.
3. Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., … others. (2021). An image is worth 16x16 words: transformers for image recognition at scale. International Conference on Learning Representations.
4. https://github.com/alohays/awesome-visual-representation-learning-with-transformers
5. Xiao, T., Radosavovic, I., Darrell, T., Malik, J. (2022). Masked Visual Pre-training for Motor Control. https://arxiv.org/abs/2203.06173.
6. Radosavovic, T., Xiao, T., James, S., Abbeel, P., Malik, J., Darrell. T. (2022). Real-World Robot Learning with Masked Visual Pre-training. https://arxiv.org/abs/2210.03109.
7. Chen, Z., Duan, Y., Wang, W., He, J., Lu, T., Dai, J., Qiao, Y. (). Vision Transformer Adapter for Dense Predictions. https://arxiv.org/abs/2205.08534.

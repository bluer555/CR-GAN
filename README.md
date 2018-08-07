# CR-GAN: Learning Complete Representations for Multi-view Generation

Training code for the paper 
**[CR-GAN: Learning Complete Representations for Multi-view Generation](https://arxiv.org/abs/1806.11191.pdf)**, IJCAI 2018

## Overview
"Encoder-Generator-Discriminator" framework is widely used to address Computer Vision applications. Training data is first mapped to a subspace via encoder, then the generator is only trained on this subspace. It lacks the ability to deal with new data, as an "unseen" data may be mapped out of the subspace, and the generator for this case is undefined.

We propose a two-pathway framework to address this problem. Generation path is introduced to let generator generates from whole space. Reconstruction path is used to reconstruct every training data.
<p align="center"><img src="intro.png" alt="Two pathway framework" width="400"></p>

### Prerequisites

This package has the following requirements:

* `Python 2.7`
* `Pytorch 0.3.1`

# CR-GAN: Learning Complete Representations for Multi-view Generation

Training code for the paper 
**[CR-GAN: Learning Complete Representations for Multi-view Generation](https://arxiv.org/abs/1806.11191.pdf)**, IJCAI 2018

## Overview
Prior works use "Encoder-Generator-Discriminator" framework to generate multi-view images for a single view input. Where training data is first mapped to a subspace via encoder, then the generator learns multi-view generation within this subspace. It lacks the ability to deal with new data, as an "unseen" data may be mapped out of the subspace, and the generator behavior for this case is undefined.

We propose a two-pathway framework to address this problem. Generation path is introduced to let generator generates in whole space. Reconstruction path is used to reconstruct all training data.
<p align="center"><img src="intro.png" alt="Two pathway framework" width="400"></p>

### Prerequisites

This package has the following requirements:

* `Python 2.7`
* `Pytorch 0.3.1`

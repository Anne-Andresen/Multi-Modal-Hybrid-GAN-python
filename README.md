# Python implementation of the Hybrid 3D GAN C and C++ Implementation

## Overview

This repository contains a python implementation of a 3D Hybrid GAN, which integrates cross-attention, self-attention, and convolutional blocks in the generator with UNet architecture of a GAN. The self-attention, convolutional layers, and GAN structure, including a UNet architecture within the generator, are implemented in and Python.

## Features

- **Self-Attention:** Integrated into the generator and implemented in Python.
- **Cross-Attention Mechanism:** Designed for 3D tensors using PyTorch, applicable as input to CNN layers. This mechanism merges two separate input tensors, providing an output of the same size, which allows for multiple input images or the introduction of new data thorughout the network. Implemented in Python, can be foundin C and C++ in another repository
- **Convolutional Blocks:** Essential convolution operations for the GAN, implemented in Python.
- **GAN Structure:** The overall GAN architecture, including the use of a UNet within the generator, is implemented in C and Python.




## Getting Started

### Prerequisites


- Python (for the Python implementation and cross-attention mechanism)
- PyTorch (for the cross-attention mechanism in Python)


### Building the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/Anne-Andresen/Hybrid-GAN-python.git
   cd Hybrid-GAN-python
   ```



## Usage


- Python Implementation: Execute the Python script:
``` bash
python3 Hybrid-GAN-python/train.py


```

## Contributing


Contributions are welcome. Please feel free to submit pull requests or open issues with suggestions and improvements.

## License


This project is licensed under the MIT License - see the LICENSE file for details.

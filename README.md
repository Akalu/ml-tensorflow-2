About
======

Building neural networks and data processing tools with the help of TensorFlow 2.0, part2

Topics have been covered so far:

* basics of TensorFlow (NumPy, Pandas, Keras)
* implementation of Convolutional Neural Network for digits recognition trained on 
  MNIST dataset (https://en.wikipedia.org/wiki/MNIST_database) - DNN, CNN

Installation
=============

Create new conda environment and make it active:

```
c:\ProgramData\Anaconda3\Scripts\conda.exe env create -f .\scripts\test-tf-2.yml

c:\ProgramData\Anaconda3\Scripts\conda.exe env list

c:\ProgramData\Anaconda3\Scripts\activate test-tf-2
```

Install additional requirements into active environment:

```
(test-tf-2) pip install -r requirements.txt
```


Overview
=========

Jupyter notebook was added for experiments and as a PoC, but the whole code is written in Python 

## Neural Network for digits recognition

Total 3 models were implemented

### Dense Neural Network (DNN)

Dense network is a network in which the number of links of each node is close to the maximal number of nodes. 
Each node is linked to almost all other nodes. The total connected case in which exactly each node is linked to each 
other node is called a completely connected network.

### Convolutional Neural Network (CNN)

[Convolutional Neural Network](https://en.wikipedia.org/wiki/Convolutional_neural_network)

A Convolutional Neural Network (ConvNet/CNN) is a Deep Learning algorithm which can take in an input image, 
assign importance (learnable weights and biases) to various aspects/objects in the image and be able to 
differentiate one from the other. The pre-processing required in a ConvNet is much lower 
as compared to other classification algorithms.

### Hyper parameter tuning

https://en.wikipedia.org/wiki/Hyperparameter_optimization

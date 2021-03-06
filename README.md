# MNIST-VAE
Variational Autoencoder using the MNIST dataset. Also included, is an ANN and CNN for MNIST as well.


## MNIST: ANN ##
* dataset is downloaded from PyTorch
  * no data prep is needed
* 3 linear layers with the ReLU activation function. 
* regulizer to prevent overfitting
  * weight decay is adjusted first
  * then number of epochs is adjusted
* back propagation is utilized 


## MNIST: CNN ##
* dataset is downloaded from PyTorch
  * the images are normalized from (0, 1) to (-1, 1)
* 2 convolutional layers that are pooled
* Kernal : 5 x 5
* Pooling : 2 x 2
* 3 linear layers
* ReLU activation function for all layers
* need to implement regulizer

## MNIST: VAE ##
* Includes a pretrained model with 500 epochs
* encoder: 
  * 3 convolutional layers
  * (batch_size, 1, 28, 28) -> (batch_size, 64, 1, 1)
  * ReLU activation
  * Compress to (batch_size, 64)
 
* sampling:
  * mean: linear layer (64, 64)
  * log variance: linear layer (64, 64)

* decoder:
  * 3 convolutional layers (transpose)
  * decompress to (batch_size, 64, 1, 1)
  * ReLU and Sigmoid activation
  * (batch_size, 64, 1, 1) -> (batch_size, 1, 28, 28) 

* mse loss function with reduction being sum

* example generation:
  * Sampling:  
    * mean: tensor of zeros (batch_size, 64)
    * log variance: tensor of ones (batch_size, 64)

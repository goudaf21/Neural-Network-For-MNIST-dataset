
To Download the Mnist training and testing sets here: https://www.kaggle.com/oddrationale/mnist-in-csv
### Introduction
In this project, I implement a 2 layer feed forward neural network to help me classify images or data from the MNIST dataset. I start
by building a network with only one hidden layer, and then add a second one after, while varying the learning rate and number of hidden nodes
and noticing the accuracy ofthe models. I also make my own images using the GIMP program and test the model on those images as well. 

### Elements of learning:
Target Function: A function that perfectly maps any image of a number to an output that correctly classify it.

Training Examples: The training examples were extracted by splitting the images of the MNIST data set into a 70/30 split (70 percent for 
training and 30% for testing. The features were in the form of 28 x 28 matrices that represents the pixels of the grey scale input images,
ranging from 0 to 255. I normalize the features to have a range between 0.01 and 0.99 and change the input matrices to one dimensional vectors
of 1 x 784 matrices. 

Hypothesis set:
Is the set of all function that the model can choose from to map inputs into outputs, in a trial to reach or approximate the target function.
During training, the model tries to choose the most fitting hypothesis from the set. In other ways, it is all the mapping functions that the model 
tries out to classify the input images into outputs (0 to 9). In doing so it uses the learning algorithm.

Learning Algorithm: 
In this model, it is batch gradient decent, which uses batches of 100 examples and uses them to calculate the gradient of the cost function of the model.
The gradient is then used to update the weights that connect the nodes among layers of the nn. This algorithm is better than normal gradient decent
as it has a less change of getting stuck in one of the local minimas of the cost function and never reaching the global minimum. 

Final Hypothesis:
The final hypothesis is chosen after training the model for 50 epochs. The model performs very well on the training data but not good for untested data.



### Description of Code:
In the code, I use a neural network class to build my neural network. When initialized, the network takes in the number of nodes of each layer,
the learning rate, the number of training epochs and the minibatch size for batch gradient decent. The model also uses a one hot encoding function which is used
to encode the target values into an array that represents multi class classification. I also use a sigmoid function to by my activation function for each of the hidden layers
and the output layer as well. I use the sigmoid function because it is applied to z_h,z_h2 and z_out, which are just dot products between the weight matrices and the outputs of each
layer. The sigmoid activation function helps keep all the values these dot products into a small range between 0 and 1, and also has an easy to use derivative. 
In the forward function, it apply the forward propagration step as it goes forward across layers, first it takes in the input, and then it calculates the values of
 z_layer and the activated functions as well a_layer. It does so by doing the dot product between the matrix of the layers and the weight matrix, then apply the sigmoid function to them.
The model also uses a compute cost function, which just computes the mean squared error between the predicted and target lists. I am not really sure that mean squared error makes a lot of sense here
since the difference between the number classified and the target number is not really of importance to us, but it is rather either a correct classification or an incorrect one. I think cross validation
would have been better in this case. 
The train function of the model first initializes the weights of the model between the layers, and the initial weights are taken from a normal distribution with a mean of 0 and a standard deviation that is scaled
number of hidden nodes to the power of -0.05. The training function keeps track of its evaluation of the model as it goes, by keeping track of training accuracy per epoch, test accuracy and the cost. Weights are then updated
and the model propogates forward once again, then computes the cost, keeps track of and then calculate the accuracy. This process is repeated for every epoch.
The model uses batch gradient decent with batch size of 100 to calculate the graient that changes the weights. When using a batch size of 1, running the model takes a much more time, so I used 100 examples per batch and 50 epochs.


### Results:
When using only one hidden layer, the accuracy on the training data reached 98 % when using 200 hidden nodes, learning rate of 0.01 and 50 epochs. When adding the second layer to the network, the accuracy reached a maxmimum of 99% when tested 
on the training data, using the same parameters. 
During my testing of the model, I tried using different learning rates for my model, the best learning rate I chose was 0.01, and the higher it increased the lower of the performance became, suggesting that when the learning rate is higher than that,
it gets stuck alternating by taking too big steps. The relationship between learning rate and performance of model is shown in the graph.
I was not able to produce a graph of epochs vs performance because of how long it takes with longer epochs, but the longer it took to train the better the model performed across different other hyper parameters.
When varying the number of hidden nodes per hidden layers, the model performed higher in general when increasing the number of nodes.

The model performed very highly when it was tested on the training data but performed very poorly with testing data (13 to 15 %), and when used on the Gimp images it also couldn't predict the numbers correctly, which could suggest an overfitting on the model.
Something I would like to change is to go back to the one hidden layer and see if adding the second layer overfit the model, in which case removing the layer would be benificial. Something I also noticed when testing the gimp images is that 3 was the most commonly predicted number, which could suggest,
that the data is not equally distributed and might have a lot more drawings of 3s than other numbers.
 






resources:
https://www.pluralsight.com/guides/importing-image-data-into-numpy-arrays
https://pillow.readthedocs.io/en/stable/reference/Image.html
https://www.kaggle.com/mikalaichaly/diy-mnist-dataset
https://ljvmiranda921.github.io/notebook/2017/02/17/artificial-neural-networks/#loss
Python Machine learning Third Edition 
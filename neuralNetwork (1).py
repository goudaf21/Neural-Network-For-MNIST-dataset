import numpy
import scipy.special
import matplotlib.pyplot as plt
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
import numpy as np
import os 
import struct
from PIL import Image,ImageOps



(x_train,y_train),(x_test,y_test)=mnist.load_data()

#Divide all values by 255 for to normalize it. Get the values between 0.01 and 0.99
x_train, x_test= (x_train/255.0)*0.98+0.01, (x_test/255.0) *0.98+0.01

#Change the array shape to a 1D array with 784 columns and one row (28*28)
x_train=x_train.reshape(x_train.shape[0],784)
x_test=x_test.reshape(x_test.shape[0],784)

images=['0.jpeg','1.jpeg','2.jpeg','3.jpeg','4.jpeg','5.jpeg','6.jpeg','7.jpeg','8.jpeg','9.jpeg']
pixels=[]
for i in images:
    image=Image.open(i)
    gray=ImageOps.grayscale(image)
    data=np.asarray(gray)
    data=data.reshape(1,784)
    pixels.append(data)








class neuralNetwork:
      
    # initialise the neural network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate,epochs,minibatch_size=100 ,l2=0):
        self.inputnodes=inputnodes
        self.hiddennodes=hiddennodes
        self.outputnodes=outputnodes
        self.lr=learningrate
        self.epochs=epochs
        self.l2=l2
        self.minibatch_size=minibatch_size
        
        
    
    def _onehot(self,y,n_classes):
        onehot=np.zeros((n_classes,y.shape[0]))
        for idx,val in enumerate(y.astype(int)):
            onehot[val,idx]=1
        return onehot.T

    
    # train the neural network
    def train(self, inputs_list, targets_list,input_test,targets_test):
        n_output = np.unique(targets_list).shape[0]
        n_features = inputs_list.shape[1]
        self.b_h = np.zeros(self.hiddennodes)
        self.w_h = numpy.random.normal(loc=0.0, scale=self.hiddennodes**(-0.05), size=(n_features, self.hiddennodes))
        self.b_h2=np.zeros(self.hiddennodes)
        self.w_h2=np.random.normal(loc=0.0, scale=self.hiddennodes**(-0.05),size=(self.hiddennodes,self.hiddennodes))
        self.b_out = np.zeros(n_output)
        self.w_out=numpy.random.normal(loc=0.0,scale=self.outputnodes**(-0.05),size=(self.hiddennodes,n_output))

        self.eval_ = {'cost': [], 'train_acc': [],'test_acc':[]}
        y_train_enc = self._onehot(targets_list, n_output)
        for i in range (self.epochs):
            indices=np.arange(inputs_list.shape[0])
            for start_idx in range(0, indices.shape[0] - self.minibatch_size +
                                   1, self.minibatch_size):
                batch_idx = indices[start_idx:start_idx + self.minibatch_size]
                z_h, a_h,z_h2,a_h2, z_out, a_out = self._forward(inputs_list[batch_idx])
                delta_out = a_out - y_train_enc[batch_idx]
                sigmoid_derivative_h = a_h * (1. - a_h)
                sigmoid_derivative_h2=a_h2*(1.-a_h2)
                
                delta_h = (np.dot(delta_out, self.w_out.T) *
                           sigmoid_derivative_h)
                delta_h2=(np.dot(delta_out,self.w_out.T)*sigmoid_derivative_h2)
                grad_w_h = np.dot(inputs_list[batch_idx].T, delta_h)
                grad_b_h = np.sum(delta_h, axis=0)
                grad_w_h2=np.dot(a_h.T,delta_h2)
                grad_b_h2=np.sum(delta_h,axis=0)
                
                grad_w_out = np.dot(a_h2.T, delta_out)
                grad_b_out = np.sum(delta_out, axis=0)
                delta_w_h = (grad_w_h)
                delta_b_h = grad_b_h # bias is not regularized
                delta_w_h2=(grad_w_h2)
                delta_b_h2=grad_b_h2
                self.w_h -= self.lr * delta_w_h
                self.b_h -= self.lr * delta_b_h
                self.w_h2-=self.lr*delta_w_h2
                self.b_h2-=self.lr*delta_b_h2
                delta_w_out = (grad_w_out + self.l2*self.w_out)
                delta_b_out = grad_b_out  # bias is not regularized
                self.w_out -= self.lr * delta_w_out
                self.b_out -= self.lr * delta_b_out
            z_h, a_h,z_h2,a_h2, z_out, a_out = self._forward(inputs_list)
            cost = self._compute_cost(y_enc=y_train_enc,
                                      output=a_out)
            y_train_pred = self.query(inputs_list)
            y_test_pred=self.query(input_test)
            train_acc = ((np.sum(y_train == y_train_pred)).astype(np.float) /
                         inputs_list.shape[0])
            test_acc=((np.sum(y_test==y_test_pred)).astype(np.float)/inputs_list.shape[0])
            
            print(train_acc)

            self.eval_['cost'].append(cost)
            self.eval_['train_acc'].append(train_acc)
            self.eval_['test_acc'].append(test_acc)
            
        return self

                

            

    
    # query the neural network
    def query(self, inputs_list):
        z_h, a_h,z_h2,a_h2, z_out, a_out = self._forward(inputs_list)
        y_pred = np.argmax(z_out, axis=1)
        return y_pred
        

    
    def _sigmoid(self, z):
        """Compute logistic function (sigmoid)"""
        return 1. / (1. + np.exp(-z))
    
    def  _forward(self,X):
        z_h = np.dot(X, self.w_h) + self.b_h
        a_h = self._sigmoid(z_h)
        z_h2=np.dot(z_h,self.w_h2)+self.b_h2
        a_h2=(self._sigmoid(z_h2))
        z_out = np.dot(a_h2, self.w_out) + self.b_out
        a_out = self._sigmoid(z_out)
        return z_h,a_h,z_h2,a_h2,z_out,a_out
    
    
    def _compute_cost(self,y_enc,output):
        cost=np.sum(np.square(y_enc-output))/len(output)

        return (cost)
    
    
    
    
nn=neuralNetwork(inputnodes=784,hiddennodes=100, outputnodes=10,learningrate=0.01,epochs=50,minibatch_size=100)
nn.train(x_train,y_train,x_test,y_test)
for i in pixels:
    pred=nn.query(i)
    print(pred)
plt.plot(range(nn.epochs), nn.eval_['cost'])
plt.ylabel('Cost')
plt.xlabel('Epochs')
plt.show()

lr=[0.001,0.01,0.1,0.2,0.4]
networks=[]
accuracy=[]
for i in lr:
    networks.append(neuralNetwork(inputnodes=784,hiddennodes=20, outputnodes=10,learningrate=i,epochs=50,minibatch_size=100))
    
for network in networks:
    network.train(x_train,y_train,x_test,y_test)
    accuracy.append(network.eval_['train_acc'][49])
print(accuracy)

plt.plot(lr,accuracy)
plt.ylabel('Performance')
plt.xlabel=('learning Rate')
plt.show()    

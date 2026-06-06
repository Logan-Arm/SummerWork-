import pandas as pd
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn.model_selection import train_test_split
import math as m
import argparse

"""Recall our setup, we have an input layer X, and an output layer Y, the data we are creating is going to be Y =f(x) where f is some function.
Lets start simple at first and have f(x) = sinh(x)
"""

"""Want to include around 4 periods of data from the sin function, so from -4pi -> 4pi, want to avoid 'riding' the periodicity of the sampled points
Want around 20 points to include, 8pi/20 = 1.2566 ~1.25"""
X = torch.arange(-4*m.pi,4*m.pi,1.25).view(-1,1).type(torch.FloatTensor)



Y = torch.sin(X.squeeze()) #Can just use torch math functions for the function

X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size= 0.2, random_state=42) #Splits the data into 80% training 20% test


#Can now apply previous architecture



class MultiLayerNet(nn.Module): #nn.module is the base class for all neural networks in pytorch


    def __init__(self, input_size, num_layers,width, output_size, std):
        super().__init__()#This runs the __init__ from the parent class, i.e., nn.module which is necessary to initialize correctly


        """Since we have a variable number of hidden layers, it is best to create a list in itialisation"""

        self.hidden_layers = nn.ModuleList()
        #Note nn.modulelist is the go to architecture which is dependent on the inputs


        #Create the initial layer
        self.hidden_layers.append(nn.Linear(input_size,width))

        #Change from perceptronVariableWidthNetwork, just sticking with 10 width throughout, so no need to gradually narrow
        for i in range(num_layers-1):
            #width_new = int(width*(1-width_mult)) 
            self.hidden_layers.append(nn.Linear(width,width))
            #width = width_new


        self.output_layer = nn.Linear(width, output_size)

        """What above does
        Maps initial values onto the first hidden layer Init--> Hidden

        Creates all other hidden layers

        Maps hidden to output hidden --> Output
        
        """

        for layer in self.hidden_layers:
            nn.init.normal_(layer.weight, mean = 0, std = std)
            nn.init.zeros_(layer.bias)#Confirm that we want zero for the biases
        """IMPORTANT Note TO SELF:  the _ at the end of each nn.init.shape creates an IN PLACE change, so we are actually editing the layers"""
    

        ###Output layer is not included in self.hidden_layers so need to handle that one externally
        nn.init.normal_(self.output_layer.weight,mean = 0, std = std)
        nn.init.zeros_(self.output_layer.bias)
    def forward(self,x):
        #pass input through the hidden layer applying sigmoid activation
        for layer in self.hidden_layers:
            x= torch.tanh(layer(x))
        y_pred = torch.tanh(self.output_layer(x))
        #Note reintroduced the tanh function here



        """What above line is doing
        
        Calling the layer as a function and passing x through it

        X is fed into hidden layer
        Hidden layer configuration applies the corresponding weights and biases ... (X * weights)+ bias
        Tanh activation is applied
        We are then using a tanh activation function to modify this final data
        """

        return y_pred
        


"""Now defining a loss function"""

def criterion(y_pred, y_true):
    #mean squared loss is simply (predicted-actual)^2
    return torch.mean((y_pred-y_true)**2)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--InputSize', type = int, help='Defines the size of the input node, almost always 1', default = 1)
    parser.add_argument('--OutputSize', type = int, help='Defines the size of the output node, almost always 1', default = 1)
    parser.add_argument('--HiddenLayerWidth', type = int, help='Defines how wide we want the hidden layers to be, i.e., how many nodes is ' \
    'the initial data mapped onto when going from initial --> HiddenLayer1', default = 10)
    parser.add_argument('--HiddenLayerDepth', type = int, help='Defines how many hidden layers we want', default = 3)
    parser.add_argument('--lr', type = float, help='Determines the learning rate for the model', default = 0.001)
    #Later add a seed function to change what f(x) is for each case
    parser.add_argument('--WidthModifyer', type = float, help='Modifyer of the width of each hidden layer, for example' \
    'say our initial width is 10, and we set our widthmodifyer = 0.1, then the width of each hidden layer will decrease by 10% ', default=0.1)
    parser.add_argument('--Epochs',type=int, help='Determines the number of training epochs', default = 200)
    parser.add_argument('--STD',type=float, help='Determines the standard deviation (width) of the normal distribution for the hidden layers weights', default = 1.0)
    parser.add_argument('--EnsembleNum', type = int, help= ' Determines the number of models to create for the purposes of ensemble averages', default= 10)
    args = parser.parse_args()

    
    
    #Need to create a loop here to iterate over and create ensemble data
    num_layers = args.HiddenLayerDepth+1
    ensemble_derivs_sq = {i:[] for i in range(num_layers)}
    for j in range(args.EnsembleNum):
            
        model = MultiLayerNet(args.InputSize,args.HiddenLayerDepth,args.HiddenLayerWidth,args.OutputSize,args.STD)

        activation_history = {i: [] for i in range(len(model.hidden_layers)+1)}
        """Creates a dictionary of lists corresponding to each layer, where we will store the activation history to calculate finite differences"""
        def make_hook(layer_id):
            """This NEEDS to be a nested function to safely pass the layer_id value to the hook function, which only ever takes the module, input and output as function inputs"""
            def hook(module, input, output):
                activation_history[layer_id].append(output.detach().abs().mean().item())
                #Appends the absolute value of the mean output of the layer to the corresponding activation history list
            return hook

        hooks =[] 



        """Note, while the hooks below are registered forward and not register_forward_pre_hook, the model is set up to calculate the layers weighting and biases upon the data
        which is then passed to the tanh function.  Since this is done separately, (i.e., no nn.sequential()), this is technically a pre-activation value, as it is done
        prior to tanh being called.  
        
        However it is good to be aware of why register_forward_ is being used in tbis case
        
        """

        for i, layer in enumerate(model.hidden_layers): #Note this does not include the output layer
            hooks.append(layer.register_forward_hook(make_hook(i)))
        
        #now need to add one for the output layer
        hooks.append(model.output_layer.register_forward_hook(make_hook(len(model.hidden_layers)))) #This will be the last output hook
        


        

        """Training the model"""
        optimiser  = torch.optim.SGD(model.parameters(), lr = args.lr)
        """Note on gradient descent... since we are using the full batch of data, this is actually a regular gradient descent not stoichastic 
        i.e.,  (no random sampling)"""

        #optimiser = torch.optim.Adam(model.parameters(),lr=args.lr)

        epochs = args.Epochs
        inst_loss = [] #This will represent the total loss of the model
        for epoch in range(epochs):
            model.train()
            #Sets the model to training mode

            y_pred = model(X_train)
            #Plugs in our X_train through the network to generate predictions
            loss = criterion(y_pred.squeeze(),Y_train)
            #This is batch loss, not necessarily ensemble loss, maybe ask about this
            


            loss.backward()
            #This is the backwards pass, it calculates the change in loss for each variable, i.e., dloss/dx for each x
            #This is a baked in method to pytorch to calculate the gradient
            
            optimiser.step()
            #Updates the weights and biases of the optimiser (i.e., where the learning occurs)
            
            optimiser.zero_grad()
            #clears the gradients
            inst_loss.append(loss.item())

            if epoch %100 ==0:
                print(f'Finished epoch {epoch}')
        derivatives_sq = {}
        for i in range(len(model.hidden_layers)+1):
            derivatives_sq[i] = abs(np.diff(activation_history[i],1))**2

        for i in range(num_layers):
            ensemble_derivs_sq[i].append(derivatives_sq[i])
        
        """Above should append the total list of derivatives at each epoch to the ensemble derivs array"""
    ensemble_means = {i:np.mean(ensemble_derivs_sq[i],axis = 0) for i in range(num_layers)}
    ensemble_uncertainty = {i:(np.std(ensemble_derivs_sq[i], axis = 0)/np.sqrt(args.EnsembleNum)) for i in range(num_layers)}
    #Above is just standard error calculation... sigma/sqrt(N)
    fig, ax = plt.subplots(figsize=(10, 6))

    for k in range(len(ensemble_means)):
        epochs_axis = range(1,args.Epochs) #Note, we start at 1 since using a np.diff finite difference schema
        mean = ensemble_means[k]
        std = ensemble_uncertainty[k]
        
        # Single line with shaded uncertainty band
        ax.plot(epochs_axis, mean, label=f'Layer {k}')
        ax.fill_between(epochs_axis, mean - std, mean + std, alpha=0.2)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Squared Activation Derivative')
    ax.set_title('Finite Difference of Layer Activations')
    ax.legend()
    plt.tight_layout()
    plt.show()
        





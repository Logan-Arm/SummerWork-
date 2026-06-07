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
X = torch.linspace(-4*m.pi,4*m.pi,20).view(-1,1).type(torch.FloatTensor)
#Used torch.linspace instead of previous torch.arrange as that created 21 points for some reason


Y = torch.sin(X.squeeze()) #Can just use torch math functions for the function

X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size= 0.2, random_state=42) #Splits the data into 80% training 20% test
sorted_indices = X_test.squeeze().argsort()
X_test_sorted = X_test[sorted_indices]
Y_test_sorted = Y_test[sorted_indices]

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
        y_pred = self.output_layer(x)



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
    parser.add_argument('--Performances', type = int, help='Determines the number of printouts of model performance desired', default=4)
    args = parser.parse_args()

    #Need to create a loop here to iterate over and create ensemble data
    print(X_test_sorted)
    num_layers = args.HiddenLayerDepth+1
    ensemble_derivs_sq = {i:[] for i in range(num_layers)}
    
    """Attempting to use more numpy arrays as opposed to dictionaries, hope to make it more clear
    """
    loss_array = np.zeros((args.Epochs,args.EnsembleNum)) #Creates an array of size [Epochs, Ensemble] to store all loss information
    epochs = args.Epochs

    performance_array = np.zeros((args.Performances, args.EnsembleNum, len(X_test_sorted)))
    #[Selected_Epoch_For_Printout, Model#, Value]

    step=int(args.Epochs/args.Performances)

    for j in range(args.EnsembleNum):

        if j%100 ==0:
            print(f'On model # {j}')

        model = MultiLayerNet(args.InputSize,args.HiddenLayerDepth,args.HiddenLayerWidth,args.OutputSize,args.STD)
        if j==0:
            print(model)
        activation_history = {i: [] for i in range(len(model.hidden_layers)+1)}
        """Creates a dictionary of lists corresponding to each layer, where we will store the activation history to calculate finite differences"""
        def make_hook(layer_id):
            """This NEEDS to be a nested function to safely pass the layer_id value to the hook function, which only ever takes the module, input and output as function inputs"""
            def hook(module, input, output):
                activation_history[layer_id].append(output.detach())#Removed the absolute value from this
                #Appends the value of the output of the layer to the corresponding activation history list
            return hook

        hooks =[] 

        """Note, while the hooks below are registered forward and not register_forward_pre_hook, the model is set up to calculate the layers weighting and biases upon the data
        which is then passed to the tanh function.  Since this is done separately, (i.e., no nn.sequential()), this is technically a pre-activation value, as it is done
        prior to tanh being called.      
        However it is good to be aware of why register_forward_ is being used in this case
        """
        for i, layer in enumerate(model.hidden_layers): #Note this does not include the output layer
            hooks.append(layer.register_forward_hook(make_hook(i)))

        #now need to add one for the output layer
        hooks.append(model.output_layer.register_forward_hook(make_hook(len(model.hidden_layers)))) #This will be the last output hook

        """Training the model"""
        optimiser  = torch.optim.SGD(model.parameters(), lr = args.lr)
        """Note on gradient descent... since we are using the full batch of data, this is actually a regular gradient descent not stoichastic 
        i.e.,  (no random sampling)"""

        for epoch in range(epochs):
            model.train()

            y_pred = model(X_train)

            loss = criterion(y_pred.squeeze(),Y_train)

            loss.backward()
            
            optimiser.step()
            
            optimiser.zero_grad()
            loss_array[epoch, j]=loss.item()

            if epoch!=0 and epoch%step ==0:
                """This will record the model performance at selected epochs
                Note that since the values calculated here are separate from the derivatives, we must remove the hooks, and we must also put the model back into training mode
                """
                for hook in hooks:
                    hook.remove()

                epoch_count = int(epoch/step)
                model.eval()
                with torch.no_grad(): 
                    Y_test_pred = model(X_test_sorted)
                for i in range(len(Y_test_pred)):
                    performance_array[epoch_count,j,i] = Y_test_pred[i]
                
#------------------------------------------------------------------------------------#
                """Now to reattach hooks and return the model to training mode"""
                for i, layer in enumerate(model.hidden_layers): 
                    hooks.append(layer.register_forward_hook(make_hook(i)))
                hooks.append(model.output_layer.register_forward_hook(make_hook(len(model.hidden_layers))))
                model.train()
#-------------------------------------------------------------------------------------#
        
        derivatives_sq = {}
        for i in range(len(model.hidden_layers)+1):
            stacked = torch.stack(activation_history[i]).numpy()
            time_deriv = np.diff(stacked,1,0) #First derivative of stacked along the 0th(epoch) axis
            derivatives_sq[i] = (time_deriv.mean(axis=(1,2)))**2
        for i in range(num_layers):
            ensemble_derivs_sq[i].append(derivatives_sq[i])
        

    ensemble_means = {i:np.mean(ensemble_derivs_sq[i],axis = 0) for i in range(num_layers)}
    ensemble_uncertainty = {i:(np.std(ensemble_derivs_sq[i], axis = 0)/np.sqrt(args.EnsembleNum)) for i in range(num_layers)}
    #Above is just standard error calculation... sigma/sqrt(N)
    fig, ax = plt.subplots(figsize=(10, 6))

    for k in range(len(ensemble_means)-1): #Added the -1 to exclude the output layer which has high activity and therefore skews graph
        epochs_axis = range(1,args.Epochs) #Note, we start at 1 since using a np.diff finite difference schema
        mean = ensemble_means[k]
        std = ensemble_uncertainty[k]
        
        # Single line with shaded uncertainty band
        ax.plot(epochs_axis, mean, label=f'Layer {k}')
        ax.fill_between(epochs_axis, mean - std, mean + std, alpha=0.3)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Squared Pre-Activation Derivative')
    ax.set_title('Finite Difference of Layer Pre-Activations')
    ax.legend()
    plt.tight_layout()
    plt.show()


    """Plotting Losses"""
    ensemble_loss = np.mean(loss_array, axis=1)
    ensemble_loss_uncert = np.std(loss_array, axis = 1)/np.sqrt(args.EnsembleNum)
    plt.plot(range(args.Epochs),ensemble_loss)
    plt.fill_between(range(args.Epochs),ensemble_loss+ensemble_loss_uncert,ensemble_loss-ensemble_loss_uncert, alpha = 0.3)
    plt.xlabel(f'Epoch')
    plt.ylabel(f"Average ensemble loss value")
    plt.title(f'Ensemble loss vs epoch')
    plt.show()

    """Plotting Performance"""
    mean_ensemble_val = np.mean(performance_array,axis=1)
    ensemble_uncert = np.std(performance_array,axis = 1)/np.sqrt(args.EnsembleNum)
    x_vals = X_test_sorted.squeeze().numpy()
    for k in range(performance_array.shape[0]):
        epoch_value = (k+1)*step

        plt.plot(x_vals, mean_ensemble_val[k,:], label = f'Predicted Values')
        plt.plot(x_vals,Y_test_sorted, label = f'True Values')
        plt.fill_between(x_vals, mean_ensemble_val[k,:]+ensemble_uncert[k,:],mean_ensemble_val[k,:]-ensemble_uncert[k,:], alpha = 0.3)
        #plt.fill_between is finicky, needs inputs to be explicitly 1 dimensional, hence the X_test_sorted.squeeze().numpy()
        plt.xlabel(f'X')
        plt.ylabel(f'Y')
        plt.title(f'Performance of the model at epoch {epoch_value}')
        plt.legend()
        plt.show()
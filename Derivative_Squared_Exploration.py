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
    parser.add_argument('--DataTrial', type = int, help='Determines what data we are looking for, 0 corresponds to overall layer analysis, ' \
    '1 corresponds to individual layer neuron analysis ' \
    '2 corresponds to both network wide analysis and layer by layer')
    parser.add_argument('--Layer', type= int, help='Determines the layer of interest for layer-specific tasks', default = 2)
    args = parser.parse_args()

    if args.DataTrial==0:
        #Need to create a loop here to iterate over and create ensemble data
        num_layers = args.HiddenLayerDepth+1
        ensemble_derivs_sq = {i:[] for i in range(num_layers)}
        for j in range(args.EnsembleNum):

            model = MultiLayerNet(args.InputSize,args.HiddenLayerDepth,args.HiddenLayerWidth,args.OutputSize,args.STD)
            if j==0:
                print(model)
            activation_history = {i: [] for i in range(len(model.hidden_layers)+1)}
            """Creates a dictionary of lists corresponding to each layer, where we will store the activation history to calculate finite differences"""
            def make_hook(layer_id):
                """This NEEDS to be a nested function to safely pass the layer_id value to the hook function, which only ever takes the module, input and output as function inputs"""
                def hook(module, input, output):
                    activation_history[layer_id].append(output.detach().abs())
                    #Appends the absolute value of the mean output of the layer to the corresponding activation history list
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
                stacked = torch.stack(activation_history[i]).numpy()
                time_deriv_sq = np.diff(stacked,1,0)**2 #First derivative of stacked along the 0th(epoch) axis
                derivatives_sq[i]=time_deriv_sq.mean(axis=(1,2))
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
            ax.fill_between(epochs_axis, mean - std, mean + std, alpha=0.3)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Squared Activation Derivative')
        ax.set_title('Finite Difference of Layer Activations')
        ax.legend()
        plt.tight_layout()
        plt.show()
    if args.DataTrial ==1:
        """#Note, as it stands this only works for hidden layers (i.e., ones with a width of 10, it does not yet work for the input or output layer)
        ^Upon testing it seems to work, not quite sure why that is

        It works for layer 0 since the output is 10 wide, it fails for layer 3 since the output is 1 wide

        """
        



        num_layers = args.HiddenLayerDepth+1
        ensemble_derivs_sq = {i:[] for i in range(args.HiddenLayerWidth)} #now focused on each neuron as opposed to each layer
        for j in range(args.EnsembleNum):
        
            model = MultiLayerNet(args.InputSize,args.HiddenLayerDepth,args.HiddenLayerWidth,args.OutputSize,args.STD)
            if j==0:
                print(model)
            activation_history = {i: [] for i in range(len(model.hidden_layers)+1)}
            def make_hook(layer_id):
                def hook(module, input, output):
                    activation_history[layer_id].append(output.detach().abs())
                return hook
            hooks =[] 

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
            neuron_derivatives_sq = {}
            stacked = torch.stack(activation_history[args.Layer]).numpy() #Only care about the specified layer
            time_deriv_sq = np.diff(stacked,1,0)**2 
            for i in range(args.HiddenLayerWidth):
                neuron_derivatives_sq[i] = (time_deriv_sq.mean(axis=1))[:,i] #Only care about mean along Input data axis, not neuron axis
                #Still want all epochs, hence why need the :,i
        

            for i in range(args.HiddenLayerWidth):
                ensemble_derivs_sq[i].append(neuron_derivatives_sq[i])

            
            """Above should append the total list of derivatives at each epoch to the ensemble derivs array"""
        ensemble_means = {i:np.mean(ensemble_derivs_sq[i],axis = 0) for i in range(args.HiddenLayerWidth)}
        ensemble_uncertainty = {i:(np.std(ensemble_derivs_sq[i], axis = 0)/np.sqrt(args.EnsembleNum)) for i in range(args.HiddenLayerWidth)}
        #Above is just standard error calculation... sigma/sqrt(N)
        fig, ax = plt.subplots(figsize=(10, 6))

        for k in range(len(ensemble_means)):
            epochs_axis = range(1,args.Epochs) #Note, we start at 1 since using a np.diff finite difference schema
            mean = ensemble_means[k]
            std = ensemble_uncertainty[k]
            
            # Single line with shaded uncertainty band
            ax.plot(epochs_axis, mean, label=f'Neuron {k}')
            ax.fill_between(epochs_axis, mean - std, mean + std, alpha=0.3)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Squared Activation Derivative')
        ax.set_title(f'Finite Difference of Neuron Activations At Layer {args.Layer}')
        ax.legend()
        plt.tight_layout()
        plt.show()

    if args.DataTrial ==2:
        num_layers = args.HiddenLayerDepth+1
        ensemble_derivs_sq = {i:[] for i in range(num_layers)}
        layer_ensemble_derivs_sq = {i: {j: [] for j in range(args.HiddenLayerWidth)} 
                            for i in range(num_layers)}
        """Need the layer ensemble derivatives to have 2 indices, one for the corresponding layer (i), and one for the neuron within that layer(j)"""



        for j in range(args.EnsembleNum):
        
            model = MultiLayerNet(args.InputSize,args.HiddenLayerDepth,args.HiddenLayerWidth,args.OutputSize,args.STD)
            if j==0:
                print(model)
            if j %10 ==0: 
                print(f'On model # {j}')
            activation_history = {i: [] for i in range(len(model.hidden_layers)+1)}
            def make_hook(layer_id):
                """This NEEDS to be a nested function to safely pass the layer_id value to the hook function, which only ever takes the module, input and output as function inputs"""
                def hook(module, input, output):
                    activation_history[layer_id].append(output.detach().abs())
                    #Appends the absolute value of the mean output of the layer to the corresponding activation history list
                return hook

            hooks =[] 
            for i, layer in enumerate(model.hidden_layers): #Note this does not include the output layer
                hooks.append(layer.register_forward_hook(make_hook(i)))
            
            #now need to add one for the output layer
            hooks.append(model.output_layer.register_forward_hook(make_hook(len(model.hidden_layers)))) #This will be the last output hook
            
            """Training the model"""
            optimiser  = torch.optim.SGD(model.parameters(), lr = args.lr)

            epochs = args.Epochs
            inst_loss = [] #This will represent the total loss of the model
            for epoch in range(epochs):
                model.train()
                #Sets the model to training mode

                y_pred = model(X_train)
                #Plugs in our X_train through the network to generate predictions
                loss = criterion(y_pred.squeeze(),Y_train)
                loss.backward()
                optimiser.step()
                optimiser.zero_grad()
                inst_loss.append(loss.item())

            derivatives_sq = {}
            for i in range(len(model.hidden_layers)+1):
                stacked = torch.stack(activation_history[i]).numpy()
                time_deriv_sq = np.diff(stacked,1,0)**2 #First derivative of stacked along the 0th(epoch) axis squared
                derivatives_sq[i]=time_deriv_sq.mean(axis=(1,2)) #Taking the average across all data points across all neurons for this model, only have layer activity at epoch
                
                #Now need to handle the layer information
                width = args.HiddenLayerWidth if i< len(model.hidden_layers) else 1
                """Come back to this, this is not a particularly nice way of dealing with output neuron"""
                #print(width)
                for j in range(width):
                    layer_ensemble_derivs_sq[i][j].append(time_deriv_sq.mean(axis=1)[:,j])
                    """Reads: for layer i append all derivatives pertaining to neuron j, mean along axis 1 removes specific test data dimensionality"""



            for i in range(num_layers):
                ensemble_derivs_sq[i].append(derivatives_sq[i])
#                layer_ensemble_derivs_sq[i].append(layer_derivs_sq[i])
            
            """Above should append the total list of derivatives at each epoch to the ensemble derivs array"""


        """Plotting network wide analysis"""
        epochs_axis = range(1,args.Epochs) #Note, we start at 1 since using a np.diff finite difference schema

        ensemble_means = {i:np.mean(ensemble_derivs_sq[i],axis = 0) for i in range(num_layers)}
        ensemble_uncertainty = {i:(np.std(ensemble_derivs_sq[i], axis = 0)/np.sqrt(args.EnsembleNum)) for i in range(num_layers)}
        #Above is just standard error calculation... sigma/sqrt(N)
        fig, ax = plt.subplots(figsize=(10, 6))

        for k in range(len(ensemble_means)):
            mean = ensemble_means[k]
            std = ensemble_uncertainty[k]
            
            # Single line with shaded uncertainty band
            ax.plot(epochs_axis, mean, label=f'Layer {k}')
            ax.fill_between(epochs_axis, mean - std, mean + std, alpha=0.3)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Squared Activation Derivative')
        ax.set_title('Finite Difference of Layer Activations')
        ax.legend()
        plt.tight_layout()
        plt.show()

        """Plotting layer by layer analysis"""

        for z in range(num_layers):
            if z< args.HiddenLayerDepth:
                #All these have width described in the argsparser
                ensemble_means = {i:np.mean(layer_ensemble_derivs_sq[z][i], axis = 0) for i in range(args.HiddenLayerWidth)}
                ensemble_uncertainty = {i:(np.std(layer_ensemble_derivs_sq[z][i], axis = 0)/np.sqrt(args.EnsembleNum)) for i in range(args.HiddenLayerWidth)}

                fig, ax = plt.subplots(figsize=(10, 6))

                for k in range(len(ensemble_means)):
                    mean = ensemble_means[k]
                    std = ensemble_uncertainty[k]
                    
                    # Single line with shaded uncertainty band
                    ax.plot(epochs_axis, mean, label=f'Neuron {k}')
                    #ax.fill_between(epochs_axis, mean - std, mean + std, alpha=0.3)

                ax.set_xlabel('Epoch')
                ax.set_ylabel('Squared Activation Derivative')
                ax.set_title(f'Finite Difference of Neuron Activations At Layer {z}')
                ax.legend()
                plt.tight_layout()
                plt.show()


                # heatmap_array = np.array([ensemble_means[i] for i in range(args.HiddenLayerWidth)])
                # fig, ax = plt.subplots(figsize=(10,6))
                # im = ax.imshow(heatmap_array,aspect = 'auto', cmap = 'coolwarm')
                # ax.set_ylabel(f'Neuron')
                # ax.set_xlabel(f'Epoch')
                # ax.set_yticks(range(args.HiddenLayerWidth))
                # plt.colorbar(im, ax=ax, label='Squared Activation Derivative')
                # ax.set_title(f'Neuron Activity Heatmap at Layer {z}')
                # plt.show()


            """Below section of code is not working, come back to at a later date, also not entirely necessary as there is only one neuron, layer wide analysis 
            captures entire activity"""
            # else:
            #     index = args.HiddenLayerDepth+1
            #     ensemble_mean = np.mean(layer_ensemble_derivs_sq[z][0], axis = 0)
            #     ensemble_uncertainty = np.std(layer_ensemble_derivs_sq[z][0], axis = 0)/np.sqrt(args.EnsembleNum)
            #     #Since only one neuron do not need the k loop

            #     ax.plot(epochs_axis, ensemble_mean, label='Output Neuron')
            #     ax.fill_between(epochs_axis, ensemble_mean - ensemble_uncertainty, ensemble_mean + ensemble_uncertainty, alpha=0.3)
            #     ax.set_xlabel('Epoch')
            #     ax.set_ylabel('Squared Activation Derivative')
            #     ax.set_title(f'Finite Difference of Neuron Activations Of Output Neuron')
            #     plt.tight_layout()
            #     plt.show()


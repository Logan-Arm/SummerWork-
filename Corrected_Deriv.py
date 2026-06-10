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

#Re-attempting using sinh
#Y = torch.sinh(X.squeeze())




X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size= 0.2, random_state=42) #Splits the data into 80% training 20% test
sorted_indices = X_test.squeeze().argsort()
X_test_sorted = X_test[sorted_indices]
Y_test_sorted = Y_test[sorted_indices]
print(f'X_train = {X_train.squeeze()}')

# print(f'Initial is{Y_test_sorted }' )
# Y_test_sorted=torch.sin(X_test_sorted.squeeze())
# print(f'Final is{Y_test_sorted }' )



#Testing continuous line instead of 4 individual points
X_eval = torch.linspace(-4*m.pi, 4*m.pi, 200).view(-1,1).type(torch.FloatTensor)
Y_eval = torch.sin(X_eval.squeeze())



#Re-attempting using sinh
#Y_eval = torch.sinh(X_eval.squeeze())


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
            #nn.init.zeros_(layer.bias)#Confirm that we want zero for the biases
            nn.init.normal_(layer.bias,mean = 0, std= std)
        """IMPORTANT Note TO SELF:  the _ at the end of each nn.init.shape creates an IN PLACE change, so we are actually editing the layers"""
    

        ###Output layer is not included in self.hidden_layers so need to handle that one externally
        nn.init.normal_(self.output_layer.weight,mean = 0, std = std)
        #nn.init.zeros_(self.output_layer.bias)

        nn.init.normal_(self.output_layer.bias, mean = 0, std= std)

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
    #Note, in the NN_EFT notes there is a 1/2 which is in front of this term, this is not necessary here as the torch autograd factors this in automatically




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--InputSize', type = int, help='Defines the size of the input node, almost always 1', default = 1)
    parser.add_argument('--OutputSize', type = int, help='Defines the size of the output node, almost always 1', default = 1)
    parser.add_argument('--HiddenLayerWidth', type = int, help='Defines how wide we want the hidden layers to be, i.e., how many nodes is ' \
    'the initial data mapped onto when going from initial --> HiddenLayer1', default = 10)
    parser.add_argument('--HiddenLayerDepth', type = int, help='Defines how many hidden layers we want', default = 3)
    parser.add_argument('--lr', type = float, help='Determines the learning rate for the model', default = 0.05)
    #Later add a seed function to change what f(x) is for each case
    parser.add_argument('--WidthModifyer', type = float, help='Modifyer of the width of each hidden layer, for example' \
    'say our initial width is 10, and we set our widthmodifyer = 0.1, then the width of each hidden layer will decrease by 10% ', default=0.3)
    parser.add_argument('--Epochs',type=int, help='Determines the number of training epochs', default = 100000)
    parser.add_argument('--STD',type=float, help='Determines the standard deviation (width) of the normal distribution for the hidden layers weights', default = 0.3)
    parser.add_argument('--EnsembleNum', type = int, help= ' Determines the number of models to create for the purposes of ensemble averages', default= 10)
    parser.add_argument('--Performances', type = int, help='Determines the number of printouts of model performance desired', default=4)
    parser.add_argument('--Bootstraps', type= int, help='Determines the number of bootstraps to calculate for error propagation', default=100)
    args = parser.parse_args()



    def Bootstrap_Analysis(list):
        
        resampled_means = []
        for i in range(args.Bootstraps):
            resampled_vals = np.random.choice(list,size = len(list), replace = True) #Generates a resampled list of same size as inputted list
            inst_resampled_mean = np.mean(resampled_vals)
            resampled_means.append(inst_resampled_mean)
        new_mean = np.mean(resampled_means)
        new_mean_of_sq = np.mean(np.array(resampled_means)**2) #need to convert to an np array before squaring
        
        #Floating point errors, leading to runtime problems, need to implement a maximum value instead
        #error = np.sqrt(new_mean_of_sq-new_mean**2) #Sigma = sqrt[<X^2>-<X>^2]
    
        error = np.sqrt(max(0, new_mean_of_sq - new_mean**2))
    
        return error





    #Need to create a loop here to iterate over and create ensemble data
    print(X_test_sorted)
    num_layers = args.HiddenLayerDepth+1
    ensemble_derivs = {i:[] for i in range(num_layers)} #Removed the _sq, no longer working with the square
    
    """Attempting to use more numpy arrays as opposed to dictionaries, hope to make it more clear
    """
    loss_array = np.zeros((args.Epochs,args.EnsembleNum)) #Creates an array of size [Epochs, Ensemble] to store all loss information
    epochs = args.Epochs

    performance_array = np.zeros((args.Performances, args.EnsembleNum, len(X_eval)))  # was len(X_test_sorted)
    #[Selected_Epoch_For_Printout, Model#, Value]

    step=int(args.Epochs/args.Performances)

    for j in range(args.EnsembleNum):

        print(f'On model # {j}')

        model = MultiLayerNet(args.InputSize,args.HiddenLayerDepth,args.HiddenLayerWidth,args.OutputSize,args.STD)
        if j==0:
            print(model)
        activation_history = {i: [] for i in range(len(model.hidden_layers)+1)}
        """Creates a dictionary of lists corresponding to each layer, where we will store the activation history to calculate finite differences"""
        def make_hook(layer_id):
            """This NEEDS to be a nested function to safely pass the layer_id value to the hook function, which only ever takes the module, input and output as function inputs"""
            def hook(module, input, output):

                activation_history[layer_id].append(output.detach()) #Confirm that there should be NO ABSOLUTE VALUE HERE
                
                #Appends the absolute value of the output of the layer to the corresponding activation history list
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

            if (epoch+1)%step ==0:
                """This will record the model performance at selected epochs
                Note that since the values calculated here are separate from the derivatives, we must remove the hooks, and we must also put the model back into training mode
                """
                for hook in hooks:
                    hook.remove()

                epoch_count = int((epoch+1)/step) - 1
                model.eval()
                with torch.no_grad(): 
                    Y_test_pred = model(X_eval)
                for i in range(len(Y_test_pred)):
                    performance_array[epoch_count,j,i] = Y_test_pred[i]
                
#------------------------------------------------------------------------------------#
                """Now to reattach hooks and return the model to training mode"""
                for i, layer in enumerate(model.hidden_layers): 
                    hooks.append(layer.register_forward_hook(make_hook(i)))
                hooks.append(model.output_layer.register_forward_hook(make_hook(len(model.hidden_layers))))
                model.train()
#-------------------------------------------------------------------------------------#
        

        """Function should be ensemble mean for each neuron, square it, then compute mean for layer"""

        derivatives_sq = {}
        for i in range(len(model.hidden_layers)+1):

            stacked = torch.stack(activation_history[i]).numpy() #Converts it to a numpy array
            #above has the following dimensionality: [epochs, batch (or data point), neuron]
            #axis 0 = epochs
            #axis 1 = batch/data
            #axis 2 = neuron

            time_deriv = np.abs(np.diff(stacked, 1, 0)) #Takes first derivative along the epoch axis

            print(f'time_deriv shape: {time_deriv.shape}') #Sanity check
            
            ensemble_derivs[i].append(time_deriv)
            #appends the layer information to the corresponding layer list in the ensemble_derivs dictionary
            #New dimensionality would be the following:
            #[layer] [ensemble_num, epoch-1 (since taken derivative), batch, neuron]

            
            
            #According to new formula, as derived and described in meeting on June 9th, this should be done AFTER ensemble averaging
            #ensemble_derivs_sq[i].append(time_deriv.mean(axis=1)) #Takes mean across the batch axis, i.e., the 16 training points



########################################################################################################################
    """End of the ensemble loop"""
########################################################################################################################


    ensemble_means = {i:[] for i in range(num_layers)}
    ensemble_uncertainty = {i:[] for i in range(num_layers)}


    #Sanity checker
    for k in range(num_layers):
            print(f'Layer {k}: ensemble_derivs[{k}] has {len(ensemble_derivs[k])} entries, first entry shape: {ensemble_derivs[k][0].shape}')
        

    for i in range(num_layers):


        print(f"On ensemble calculations for layer {i}")

        stacked_ensemble = np.stack(ensemble_derivs[i], axis = 0) #Shape is now [ensemble, epochs-1, neurons]
        """What is going on in above line:
        We are taking the FIRST ELEMENT of the ensemble_derivs, recall from line 248 that this corresponds to selecting a specific layer
        We then stack all these list elements into one numpy array along the 0th axis of the components, i.e., along the ensemble_num axis
        Therefore we have one numpy array for each layer, of the shape [ensemble, epoch -1, batch, neuron ]
        """

        #Take the mean value with respect to the ensembles
        ens_mean = np.mean(stacked_ensemble, axis = 0) #New dimensionality is [epochs-1,batch, neurons]
        ens_std = np.std(ens_mean, axis = 0)/args.EnsembleNum

        """COme back to the uncertainty, not totally sure how to calculate it"""

        #ens_mean_sq = ens_mean**2 #Values should (in theory) scale to 1/n, so to square it is just going to shrink it immensely. work with absolute vals instead

        #Now take the mean over the remaining axes, i.e., batches first, then calculate error, then take remaining mean
        ensemble_means_neurons = ens_mean.mean(axis = 1)
        #Now have something of shape [epochs -1, neurons]


        #Want to calculate the bootstrap uncertainty for this list
        for j in range(ensemble_means_neurons.shape[0]):
            ensemble_uncertainty[i].append(Bootstrap_Analysis(ensemble_means_neurons[j,:]))
        """Above is commented out initially to simply get a plot without errors to save computation time"""

        ensemble_means[i] = np.mean(ensemble_means_neurons,axis =1) #Now averaging over all the neurons
        ensemble_uncertainty[i] = np.array(ensemble_uncertainty[i]) #Need it to be a numpy array
        #What is remaining is a list of dimensionality [epochs-1, means], this is the appended to the ith component of the ensemble_means dictionary


        



    #Above is just standard error calculation... sigma/sqrt(N)
    fig, ax = plt.subplots(figsize=(10, 6))

    for k in range(len(ensemble_means)): 
        """Attempting to go performance by performance to mitigate the zoomout effect of training over all epochs"""

        epochs_axis = range(1,args.Epochs) #Note, we start at 1 since using a np.diff finite difference schema
        mean = ensemble_means[k]
        std = ensemble_uncertainty[k]
        
        # Single line with shaded uncertainty band
        ax.plot(epochs_axis, mean, label=f'Layer {k}')
        ax.fill_between(epochs_axis, mean - std, mean + std, alpha=0.3)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Pre-Activation Derivative')
    ax.set_title('Finite Difference of Layer Pre-Activations')
    ax.legend()
    plt.tight_layout()
    plt.show()
    plt.close()



    """Excluding the final layer"""

    fig, ax = plt.subplots(figsize=(10, 6))
    for k in range(len(ensemble_means)-1): 

        epochs_axis = range(1,args.Epochs) #Note, we start at 1 since using a np.diff finite difference schema
        mean = ensemble_means[k]
        std = ensemble_uncertainty[k]
        
        # Single line with shaded uncertainty band
        ax.plot(epochs_axis, mean, label=f'Layer {k}')
        ax.fill_between(epochs_axis, mean - std, mean + std, alpha=0.3)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Pre-Activation Derivative')
    ax.set_title('Finite Difference of Layer Pre-Activations')
    ax.legend()
    plt.tight_layout()
    plt.show()
    plt.close()


    """Plotting Losses"""
    ensemble_loss = np.mean(loss_array, axis=1)
    ensemble_loss_uncert = np.std(loss_array, axis = 1)/np.sqrt(args.EnsembleNum)
    plt.plot(range(args.Epochs),ensemble_loss)
    plt.fill_between(range(args.Epochs),ensemble_loss+ensemble_loss_uncert,ensemble_loss-ensemble_loss_uncert, alpha = 0.3)
    plt.xlabel(f'Epoch')
    plt.ylabel(f"Average ensemble loss value")
    plt.title(f'Ensemble loss vs epoch')
    plt.show()
    plt.close

    """Plotting Performance"""
    mean_ensemble_val = np.mean(performance_array,axis=1)
    ensemble_uncert = np.std(performance_array,axis = 1)/np.sqrt(args.EnsembleNum)
    x_vals = X_eval.squeeze().numpy()
    for k in range(performance_array.shape[0]):
        
        
        epoch_value = (k+1)*step

        plt.plot(x_vals, mean_ensemble_val[k,:], label = f'Predicted Values')
        plt.plot(x_vals,Y_eval.numpy(), label = f'True Values')
        plt.fill_between(x_vals, mean_ensemble_val[k,:]+ensemble_uncert[k,:],mean_ensemble_val[k,:]-ensemble_uncert[k,:], alpha = 0.3)
        #plt.fill_between is finicky, needs inputs to be explicitly 1 dimensional, hence the X_test_sorted.squeeze().numpy()
        plt.xlabel(f'X')
        plt.ylabel(f'Y')
        plt.title(f'Performance of the model at epoch {epoch_value}')
        plt.legend()
        plt.show()
        plt.close() #Prevents data from piling up
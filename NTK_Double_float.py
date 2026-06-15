import pandas as pd
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn.model_selection import train_test_split
import math as m
import argparse
from joblib import Parallel, delayed
from torch.func import jacrev, vmap, functional_call
#above will be used for NTK calcs
"""Recall our setup, we have an input layer X, and an output layer Y, the data we are creating is going to be Y =f(x) where f is some function.
Lets start simple at first and have f(x) = sinh(x)
"""

"""Want to include around 4 periods of data from the sin function, so from -4pi -> 4pi, want to avoid 'riding' the periodicity of the sampled points
Want around 20 points to include, 8pi/20 = 1.2566 ~1.25"""
X = torch.linspace(-4*m.pi,4*m.pi,20).view(-1,1).type(torch.DoubleTensor)
#Used torch.linspace instead of previous torch.arrange as that created 21 points for some reason


Y = torch.sin(X.squeeze()) #Can just use torch math functions for the function

#Re-attempting using sinh
#Y = torch.sinh(X.squeeze())




X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size= 0.2, random_state=42) #Splits the data into 80% training 20% test
sorted_indices = X_test.squeeze().argsort()
X_test_sorted = X_test[sorted_indices]
Y_test_sorted = Y_test[sorted_indices]
print(f'X_train = {X_train.squeeze()}')


Test_sorted_indices = X_train.squeeze().argsort()
X_train_sorted = X_train[Test_sorted_indices]
Y_train_sorted = Y_train[Test_sorted_indices]
# print(f'Initial is{Y_test_sorted }' )
# Y_test_sorted=torch.sin(X_test_sorted.squeeze())
# print(f'Final is{Y_test_sorted }' )



#Testing continuous line instead of 4 individual points
X_eval = torch.linspace(-4*m.pi, 4*m.pi, 200).view(-1,1).type(torch.DoubleTensor)
Y_eval = torch.sin(X_eval.squeeze())



#Re-attempting using sinh
#Y_eval = torch.sinh(X_eval.squeeze())


#Can now apply previous architecture

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

def NTK_calc(model, X):
    params = {k: v.detach() for k,v in model.named_parameters()}
    #model.named_parameters() returns all the learnable tensors within the model, i.e., each layers weight and bias tensor
    #model.named... returns a tuple of shape (name, tensor), k is then attached to the name, and v gets attached to the actual tensor
    #v.detach ensures that the v (tensor) we store is not changed by any autograds, in essence we are decoupling it from the learning procedure

    def fnet_single(params, x):
        """
    Above lets us work in a functionally "pure" space, see june 12th notes in word document for more details
    Input model describes the architecture and forward pass procedure of our network, however, parameter space values are overwritten with the values provided
    Input params, gives the functional call all the information known about the parameter tensors of each layer of the network
    Functional call requires the data be provided in a tuple form, hence the x.unsqueeze(0), which just adds a batch dimension 
    .squeeze(0) removes this instant added dimensionality so that the data can be passed as a scalar    
        """
        return functional_call(model, params, (x.unsqueeze(0),)).squeeze(0)
    J = vmap(jacrev(fnet_single),(None,0))(params,X)
   
    """Vmap vectorises/parallelises the function/calculation being called, in this case jacrev of fnet_single
    jacrev calls the jacobian with respect to the inputs first argument, which in the case of fnet_single is params
    The (None,0) tells vmap which dimensionality to parallelise along, as seen by the function inputs i.e., the (params,X) at the end of line 92 
    None describes parallelisation of the first input of jacrev(fnet_single), i.e., the parameters
    Since we want the parameters to stay the same (as is the goal of calculating the jacobian) we set this to none
    We do however want to vectorise the X calculations, so we say we vectorise along the 0th dimension of the X inputs, i.e., row by row
    (params,X) give us the inputs to feed into the fnet_single

    The output of this, for a model with N data points is the following
    layer_0_weight = [N,10,1]
    layer_0_bias = [N,10]
    layer_1_weight = [N,10,10]
    layer_1_bias = [N,10]
    """

    #We now need to flatten these arrays
    J_flat = torch.cat([j.flatten(1) for j in J.values()], dim=1)
    """
    J.values() simply grabs one of the rows described previously (end of previous docstring)

    j.flatten(1) collapses all dimensionality beyond 1, --> everything to 2 dimensionals
    layer_0_weight --> [N,10] as 10X1 =10
    layer_1_weight --> [N,100] as 10X10 =100

    concatanating joins all the different gradients along the parameter space, i.e., for layer 1,2, etc

    
    """
    return J_flat @ J_flat.T #Final matrix multiplication to provide the NTK
    #Explicitly provides grad_theta_f (xi) dot grad_theta_f (xj), which is the definition of the NTK 





class MultiLayerNet(nn.Module): #nn.module is the base class for all neural networks in pytorch


    def __init__(self, input_size, num_layers,width, output_size, std):
        super().__init__()#This runs the __init__ from the parent class, i.e., nn.module which is necessary to initialize correctly


        """Since we have a variable number of hidden layers, it is best to create a list in itialisation"""


        self.hidden_layers = nn.ModuleList()

        #Create the initial layer
        self.hidden_layers.append(nn.Linear(input_size,width))

        for i in range(num_layers-1): 
            self.hidden_layers.append(nn.Linear(width,width))

        self.output_layer = nn.Linear(width, output_size)
        """What above does
        Maps initial values onto the first hidden layer Init--> Hidden
        Creates all other hidden layers
        Maps hidden to output hidden --> Output
        """
        self.layer_widths = []
        for layer in self.hidden_layers:
            nn.init.normal_(layer.weight, mean = 0, std = std)
            nn.init.normal_(layer.bias,mean = 0, std= std)
            self.layer_widths.append(layer.out_features)#Appending widths of the layer
        """IMPORTANT Note TO SELF:  the _ at the end of each nn.init.shape creates an IN PLACE change, so we are actually editing the layers"""
    

        ###Output layer is not included in self.hidden_layers so need to handle that one externally
        nn.init.normal_(self.output_layer.weight,mean = 0, std = std)
        nn.init.normal_(self.output_layer.bias, mean = 0, std= std)
        self.layer_widths.append(self.output_layer.out_features) #appending the last layer to the widths array

    def forward(self,x):
        #pass input through the hidden layer applying tanh activation
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

"""Creating a function to remove all hooks from the model"""
def remove_hooks(hooks):
    for hook in hooks:
        hook.remove()
    hooks.clear

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
    parser.add_argument('--AlignmentInterval', type = int, help='Determines how frequently to calculate the NTK alignment', default=100)
    args = parser.parse_args()


    #Need to create a loop here to iterate over and create ensemble data
    print(X_test_sorted)
    num_layers = args.HiddenLayerDepth+1
    ensemble_derivs = {i:[] for i in range(num_layers)} #Removed the _sq, no longer working with the square
    
    NTK_Zero_Matrix = []
    """Creates a 1 dimensional numpy array of length =  ensemblenumber, will append the initial NTK value for each ensemble
    member here, then average it to create the average NTK at time = 0
    """

    Y_vector = Y_train.numpy()
    normalising_const = np.dot(Y_vector,Y_vector.T)#Normalising constant to be used in alignment calculations

    #Alignment array needs to have following dimensionality [ensemblenum, epoch]
    alignment_array = np.zeros((args.EnsembleNum,args.Epochs//args.AlignmentInterval))


    loss_array = np.zeros((args.Epochs,args.EnsembleNum)) #Creates an array of size [Epochs, Ensemble] to store all loss information
    epochs = args.Epochs

    performance_array = np.zeros((args.Performances, args.EnsembleNum, len(X_eval)))  # was len(X_test_sorted)
    #[Selected_Epoch_For_Printout, Model#, Value]

    step=int(args.Epochs/args.Performances)

    for j in range(args.EnsembleNum):
        print(f'On model # {j}')
        model = MultiLayerNet(args.InputSize,args.HiddenLayerDepth,args.HiddenLayerWidth,args.OutputSize,args.STD).double()
        if j==0:
            print(model)
        activation_history = {i: [] for i in range(len(model.hidden_layers)+1)}
        """Creates a dictionary of lists corresponding to each layer, where we will store the activation history to calculate finite differences"""
        def make_hook(layer_id):
            """This NEEDS to be a nested function to safely pass the layer_id value to the hook function, which only ever takes the module, input and output as function inputs"""
            def hook(module, input, output):
                activation_history[layer_id].append(output.detach())                
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

            
            if epoch==0: #appends the first matrix
                remove_hooks(hooks)
                ntk_matrix = NTK_calc(model,X_train).numpy()
                NTK_Zero_Matrix.append(ntk_matrix)
                for i, layer in enumerate(model.hidden_layers):
                    hooks.append(layer.register_forward_hook(make_hook(i)))
                hooks.append(model.output_layer.register_forward_hook(make_hook(len(model.hidden_layers))))


            if epoch%100==0: #Only include NTK calculations every 100 epochs
                #print(f"Currently on epoch {epoch}")

                #Removing hooks

                remove_hooks(hooks)
                ntk_matrix = NTK_calc(model,X_train).numpy()
                
                alignment_item = np.dot(Y_vector,ntk_matrix)
                alignment_array[j,epoch//args.AlignmentInterval] = np.dot(alignment_item,Y_vector.T)/normalising_const
                
                #Re-adding hooks
                for i, layer in enumerate(model.hidden_layers): #Note this does not include the output layer
                    hooks.append(layer.register_forward_hook(make_hook(i)))
                hooks.append(model.output_layer.register_forward_hook(make_hook(len(model.hidden_layers)))) #This will be the last output hook

            #Start of training loop
            model.train()
            y_pred = model(X_train)
            loss = criterion(y_pred.squeeze(),Y_train)
            loss.backward()
            optimiser.step()
            optimiser.zero_grad()
            loss_array[epoch, j]=loss.item()
            #End of training loop

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

                """Now to reattach hooks and return the model to training mode"""
                for i, layer in enumerate(model.hidden_layers): 
                    hooks.append(layer.register_forward_hook(make_hook(i)))
                hooks.append(model.output_layer.register_forward_hook(make_hook(len(model.hidden_layers))))
                model.train()

        """Function should be ensemble mean for each neuron, square it, then compute mean for layer"""

        derivatives_sq = {}
        for i in range(len(model.hidden_layers)+1):

            stacked = torch.stack(activation_history[i]).numpy() #Converts it to a numpy array
            #above has the following dimensionality: [epochs, batch (or data point), neuron]
            #axis 0 = epochs
            #axis 1 = batch/data
            #axis 2 = neuron

            time_deriv = np.diff(stacked, 1, 0) #Takes first derivative along the epoch axis

            print(f'time_deriv shape: {time_deriv.shape}') #Sanity check
            
            ensemble_derivs[i].append(time_deriv)
            #appends the layer information to the corresponding layer list in the ensemble_derivs dictionary
            #New dimensionality would be the following:
            #[layer] [ensemble_num, epoch-1 (since taken derivative), batch, neuron]

            
            
            #According to new formula, as derived and described in meeting on June 9th, this should be done AFTER ensemble averaging
            #ensemble_derivs_sq[i].append(time_deriv.mean(axis=1)) #Takes mean across the batch axis, i.e., the 16 training points



########################################################################################################################
    """End of the ensemble loop"""




#Averging net matrix
########################################################################################################################
    
    NTK_stacked = np.stack(NTK_Zero_Matrix, axis = 0) #has dimensionality [Ensemblenum, N,N]
    ens_zero_NTK = np.mean(NTK_stacked, axis = 0)
    #Stack each ensemble's ntk matrix along the 0th dimension, then average across that 0th dimension
    eigvals, eigvecs = np.linalg.eigh(ens_zero_NTK) #Calculates the average NTK's eigenvalues and eigenvectors
    
    #now calculate the variance along each element of the NTK matrix for different ensemble members
    element_var = np.var(NTK_stacked, axis = 0) / len(NTK_Zero_Matrix) 
    #This is an array of shape [N,N], where each element is the uncertainty on the mean NTK's (ens_zero_NTK) value at corresponding point
    
    eigval_uncert = np.zeros(len(eigvals)) #Creates a zero array of length of the eigenvalue array, one uncertainty per eigenvalue
    for k in range(len(eigvals)):
        v = eigvecs[:,k] #Creates a list of all the corresponding eigenvectors for each ensemble
        grad_matrix = np.outer(v,v) #Calculates the outerproduct of the vector
        #Above is the uncertainty on the individual eigenvector to perturbations of the NTK matrix
        eigval_uncert[k] = np.sqrt(np.sum(grad_matrix**2 * element_var)) #Error propagation

    mean_eigenvals = np.sort(eigvals)#You use eigvalsh for symmetric matrices
    



########################################################################################################################




#Trying on eigenvalue by eigenvalue basis
    # eigenvalue_array = np.zeros((len(NTK_Zero_Matrix),len(X_train))) #Creates array of size [ensemble, test_data]
    # for i in range(len(NTK_Zero_Matrix)):
    #     eigenvals = np.sort(np.linalg.eigvals(NTK_Zero_Matrix[i]))
    #     eigenvalue_array[i,:] =eigenvals
    # eigval_uncert = np.std(eigenvalue_array, axis = 0)/np.sqrt(eigenvalue_array.shape[0]) #standard error on mean calculation
    # mean_eigenvals = np.mean(eigenvalue_array, axis = 0)
########################################################################################################################

    if np.any(mean_eigenvals< 0):
        print(f"Negative eigenvalue in list")
    ens_zero_eigenvalues = mean_eigenvals[mean_eigenvals>0]
    #ens_zero_eigenvalues = np.linalg.eigvals(ens_zero_NTK)
    print(f'The eigenvalues of the ensemble average NTK matrix are {ens_zero_eigenvalues} ')


    NTK_points = 1/ens_zero_eigenvalues #Points of interest are described by the inverse values of the eigenvalues
    print(f"The specific points of interest from the initial NTK axes is {NTK_points}")

    """Need to remove any points which are outside the maximum epoch range"""
    
    NTK_points = NTK_points[NTK_points <= args.Epochs].real #Only care about real eigenvalues, filter out any points which are beyond our training range
    ensemble_means = {i:[] for i in range(num_layers)}
    ensemble_uncertainty = {i:[] for i in range(num_layers)}

    alignment_uncert = np.std(alignment_array, axis = 0)/np.sqrt(args.EnsembleNum)
    alignment_mean = np.mean(alignment_array, axis = 0)

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
        ens_mean = np.abs(np.mean(stacked_ensemble, axis = 0)) #New dimensionality is [epochs-1,batch, neurons]

        """COme back to the uncertainty, not totally sure how to calculate it"""

        #ens_mean_sq = ens_mean**2 #Values should (in theory) scale to 1/n, so to square it is just going to shrink it immensely. work with absolute vals instead

        #Now take the mean over the remaining axes, i.e., batches first, then calculate error, then take remaining mean
        ensemble_means_neurons = ens_mean.mean(axis = 1)
        #Now have something of shape [epochs -1, neurons]


        # #Want to calculate the bootstrap uncertainty for this list
        # for j in range(ensemble_means_neurons.shape[0]):
        #     ensemble_uncertainty[i].append(Bootstrap_Analysis(ensemble_means_neurons[j,:]))
        """Above is commented out initially to simply get a plot without errors to save computation time"""

        ensemble_means[i] = np.mean(ensemble_means_neurons,axis =1) #Now averaging over all the neurons
        # results = Parallel(n_jobs= -1)(delayed(Bootstrap_Analysis)(ensemble_means_neurons[j,:]) for j in range(ensemble_means_neurons.shape[0])
        #                                )
        # ensemble_uncertainty[i] = np.array(results)
       
        #ensemble_uncertainty[i] = np.array(ensemble_uncertainty[i]) #Need it to be a numpy array
        #What is remaining is a list of dimensionality [epochs-1, means], this is the appended to the ith component of the ensemble_means dictionary


    """Printing out the NTK matrix"""
    train_vals = X_train.squeeze().numpy().round(2)
    NTK_df = pd.DataFrame(ens_zero_NTK, index=train_vals, columns=train_vals)
    print(NTK_df.to_string(float_format=lambda x: f'{x:.2f}'))

    # NTK_df.to_csv('NTK_matrix.csv', float_format='%.2f')
    #Saving the csv^^

    """End of calculations"""
#####################################################################################################################################

    """Starting plots"""
    fig, ax = plt.subplots(figsize=(10, 6)) 
    for k in range(len(ensemble_means)): 
        epochs_axis = range(1,args.Epochs) #Note, we start at 1 since using a np.diff finite difference schema
        mean = ensemble_means[k]
        #std = ensemble_uncertainty[k]
        ax.plot(epochs_axis, mean, label=f'Layer {k+1}') #Convention is to use layer 0 as input, so need to shift everything up by 1
        #ax.fill_between(epochs_axis, mean - std, mean + std, alpha=0.3)

    #Adding important regions to plot
    for j in range(len(NTK_points)):
        ax.axvspan(NTK_points[j] - eigval_uncert[j], NTK_points[j] + eigval_uncert[j], color='purple', alpha=0.3)
        #Axvspan expects a single point, cannot use an array, hence need the for loop    
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
        #std = ensemble_uncertainty[k]
        
        # Single line with shaded uncertainty band
        ax.plot(epochs_axis, mean, label=f'Layer {k+1}')
        #ax.fill_between(epochs_axis, mean - std, mean + std, alpha=0.3)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Pre-Activation Derivative')
    ax.set_title('Finite Difference of Layer Pre-Activations')
    for j in range(len(NTK_points)):
        ax.axvspan(NTK_points[j] - eigval_uncert[j], NTK_points[j] + eigval_uncert[j], color='purple', alpha=0.3)
        #Axvspan expects a single point, cannot use an array, hence need the for loop
    ax.legend()
    plt.tight_layout()
    plt.show()
    plt.close()


    """NTK alignment value"""
    print(f'The dimensionality of the alignment_mean array is {np.shape(alignment_mean)}')
    """Plotting alignment values"""
    plt.figure(figsize=(8,6))
    plt.plot(np.arange(0,args.Epochs,100),alignment_mean)
    plt.fill_between(np.arange(0,args.Epochs,100), alignment_mean +alignment_uncert, alignment_mean-alignment_uncert, alpha = 0.3)
    plt.xlabel(f'Epoch')
    plt.ylabel(f'Alignment')
    plt.title(f'Alignment of the NTK vs Epoch')
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
    #Need to include training points here
    mean_ensemble_val = np.mean(performance_array,axis=1)
    ensemble_uncert = np.std(performance_array,axis = 1)/np.sqrt(args.EnsembleNum)
    x_vals = X_eval.squeeze().numpy()
    for k in range(performance_array.shape[0]):
        epoch_value = (k+1)*step
        plt.figure(figsize=(8,6))
        plt.plot(x_vals, mean_ensemble_val[k,:], label = f'Predicted Values')
        plt.plot(x_vals,Y_eval.numpy(), label = f'True Values')
        plt.scatter(X_train_sorted,Y_train_sorted,label = "Training points")
        plt.fill_between(x_vals, mean_ensemble_val[k,:]+ensemble_uncert[k,:],mean_ensemble_val[k,:]-ensemble_uncert[k,:],color = 'blue', alpha = 0.3)
        #plt.fill_between is finicky, needs inputs to be explicitly 1 dimensional, hence the X_test_sorted.squeeze().numpy()
        plt.xlabel(f'X')
        plt.ylabel(f'Y')
        plt.title(f'Performance of the model at epoch {epoch_value}')
        plt.legend()
        plt.show()
        plt.close() 
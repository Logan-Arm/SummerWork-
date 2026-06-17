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




# X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size= 0.2, random_state=42) #Splits the data into 80% training 20% test
# sorted_indices = X_test.squeeze().argsort()
# X_test_sorted = X_test[sorted_indices]
# Y_test_sorted = Y_test[sorted_indices]

X_train = torch.linspace(-0.5*m.pi,0.5*m.pi,16).view(-1,1).type(torch.DoubleTensor)
Y_train = torch.sin(X_train).squeeze()
verification_array = X_train.numpy()
mean_sq = np.mean(verification_array**2)
print(f'The expectation value squared of the training data is {mean_sq}')
#print(f'X_train = {X_train.squeeze()}')


Test_sorted_indices = X_train.squeeze().argsort()
X_train_sorted = X_train[Test_sorted_indices]
Y_train_sorted = Y_train[Test_sorted_indices]
# print(f'Initial is{Y_test_sorted }' )
# Y_test_sorted=torch.sin(X_test_sorted.squeeze())
# print(f'Final is{Y_test_sorted }' )



#Testing continuous line instead of 4 individual points
X_eval = torch.linspace(-0.5*m.pi, 0.5*m.pi, 200).view(-1,1).type(torch.DoubleTensor)
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
        self.convergence = 0 #To be used to tell when loss plateaus
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
        for layer in self.hidden_layers:
            x= (layer(x)) #Linear activation
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
    parser.add_argument('--lr', type = float, help='Determines the learning rate for the model', default = 0.001)
    parser.add_argument('--WidthModifyer', type = float, help='Modifyer of the width of each hidden layer, for example' \
    'say our initial width is 10, and we set our widthmodifyer = 0.1, then the width of each hidden layer will decrease by 10% ', default=0.3)
    parser.add_argument('--Epochs',type=int, help='Determines the number of training epochs', default = 3000)
    parser.add_argument('--STD',type=float, help='Determines the standard deviation (width) of the normal distribution for the hidden layers weights', default = 0.3)
    parser.add_argument('--EnsembleNum', type = int, help= ' Determines the number of models to create for the purposes of ensemble averages', default= 20)
    parser.add_argument('--Performances', type = int, help='Determines the number of printouts of model performance desired', default=4)
    parser.add_argument('--Bootstraps', type= int, help='Determines the number of bootstraps to calculate for error propagation', default=100)
    parser.add_argument('--AlignmentInterval', type = int, help='Determines how frequently to calculate the NTK alignment', default=100)
    parser.add_argument('--SaveFig', action='store_true', help='If set, saves figures to args.Filename')
    parser.add_argument('--Filename', type = str, help='Determines the file to save data to', default= 'SummerWork')
    args = parser.parse_args()

    plateau_array = []
    #Need to create a loop here to iterate over and create ensemble data
    #print(X_test_sorted)
    num_layers = args.HiddenLayerDepth+1
    ensemble_derivs = {i:[] for i in range(num_layers)} #Removed the _sq, no longer working with the square
    
    """Creating an eigenvector alignment array to store data of e^T y where y is our training data
    Must have corresponding dimensionality [Training_time, ensemble member, eigenvector alignment]
    """
    
    evec_alignment = np.zeros((args.Epochs//args.AlignmentInterval, args.EnsembleNum, 2)) #Only ever 2 "features" to learn in linear model
    # print(f"Evec_alignment has dimensionality {evec_alignment.ndim}")
    eval_array = np.zeros((args.Epochs//args.AlignmentInterval, args.EnsembleNum, 2))
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

                remove_hooks(hooks)
                ntk_matrix = NTK_calc(model,X_train).numpy()
                ntk_norm = np.linalg.norm(ntk_matrix, 'fro')
                evals, evecs = np.linalg.eigh(ntk_matrix) #calculates eigenvectors and eigenvales
                selected_indices = np.argsort(evals)[-2:] #Takes last 2 values of the sorted eigenvalues
                selected_evecs = evecs[:,selected_indices]
                selected_evals = evals[selected_indices]
                for z in range(2):
                    evec_alignment[epoch//args.AlignmentInterval,j,z] = np.dot(selected_evecs[:,z].T,Y_vector)/(np.linalg.norm(Y_vector)) #Normalises
                    eval_array[epoch//args.AlignmentInterval,j,z] =selected_evals[z]
                alignment_item = np.dot(Y_vector,ntk_matrix)
                alignment_array[j,epoch//args.AlignmentInterval] = np.dot(alignment_item,Y_vector.T)/(normalising_const * ntk_norm) #ntk norm is frobenius
                
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
            if epoch >0 and model.convergence == 0 and abs((loss_array[epoch,j]-loss_array[epoch-1,j]))/loss_array[epoch,j] < 1e-8: #Relative change in loss
                print(f"Loss has plateaued at epoch {epoch}")
                model.convergence = 1
                plateau_array.append(epoch)
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



    mean_plateau = np.mean(plateau_array)
    print(f"The mean plateau value is {mean_plateau}")
#Averging net matrix
########################################################################################################################
    
    NTK_stacked = np.stack(NTK_Zero_Matrix, axis = 0) #has dimensionality [Ensemblenum, N,N]
    # ens_zero_NTK = np.mean(NTK_stacked, axis = 0)
    # ens_zero_NTK = (ens_zero_NTK + ens_zero_NTK.T) / 2 #Symmetrise the matrix
    # #Stack each ensemble's ntk matrix along the 0th dimension, then average across that 0th dimension
    # eigvals, eigvecs = np.linalg.eigh(ens_zero_NTK) #Calculates the average NTK's eigenvalues and eigenvectors
    
    

    # #now calculate the variance along each element of the NTK matrix for different ensemble members
    # element_var = np.var(NTK_stacked, axis = 0) / len(NTK_Zero_Matrix) 
    # #This is an array of shape [N,N], where each element is the uncertainty on the mean NTK's (ens_zero_NTK) value at corresponding point
    
    # eigval_uncert = np.zeros(len(eigvals)) #Creates a zero array of length of the eigenvalue array, one uncertainty per eigenvalue
    # for k in range(len(eigvals)):
    #     v = eigvecs[:,k] #Creates a list of all the corresponding eigenvectors for each ensemble
    #     grad_matrix = np.outer(v,v) #Calculates the outerproduct of the vector
    #     #Above is the uncertainty on the individual eigenvector to perturbations of the NTK matrix
    #     eigval_uncert[k] = np.sqrt(np.sum(grad_matrix**2 * element_var)) #Error propagation

    # mean_eigenvals = np.sort(eigvals)#You use eigvalsh for symmetric matrices

    """Bootstrapping the eigenvalues and the uncertainty for the eigenvalues"""
    bootstrapped_eigvals = np.zeros((args.Bootstraps, 2)) #only care about the 2 nonzero eigenvalues
    bootstrapped_eigvecs = np.zeros((args.Bootstraps, 2, len(X_train)))#only care about the 2 eigenvectors corresponding the 2 nonzero eigenvalues
    for b in range(args.Bootstraps):
        #Recall NTK_stacked zero axis is our replica axis
        boots_indices = np.random.choice(NTK_stacked.shape[0], size = NTK_stacked.shape[0], replace = True)
        #Randomly resampling (with replacement) which replicas to include
        boot_mean_NTK = np.mean(NTK_stacked[boots_indices], axis = 0) 
        #Taking mean of this resampled
        boot_mean_NTK = (boot_mean_NTK + boot_mean_NTK.T)/2 #Making explicitly symmetric, recall that the NTK is supposed to be symmetric
        boot_eigvals, boot_eigvecs = np.linalg.eigh(boot_mean_NTK)
        boot_sorted_indices = np.argsort(boot_eigvals)[-2:] #Grabs last 2 points, i.e., the nonzero eigenvalue
        boot_eigvals = boot_eigvals[boot_sorted_indices]
        boot_eigvecs = boot_eigvecs[:,boot_sorted_indices] #Need slice to actually take the vector
        bootstrapped_eigvals[b] = boot_eigvals
        # for k in range(len(boot_eigvals)):
        #     bootstrapped_eigvecs[b,k,:] = boot_eigvecs[:,k] 
    
    mean_eigenvals = np.mean(bootstrapped_eigvals, axis = 0)
    mean_eigenvals_std = np.std(bootstrapped_eigvals, axis =0)
    print(f"The eigenvalues of interest are {mean_eigenvals}")
    NTK_points = 1/mean_eigenvals
    NTK_point_uncert = mean_eigenvals_std/(mean_eigenvals**2)
    
    # if np.any(mean_eigenvals< 0):
    #     print(f"Negative eigenvalue in list")
    # ens_zero_eigenvalues = mean_eigenvals[mean_eigenvals>0]
    # #ens_zero_eigenvalues = np.linalg.eigvals(ens_zero_NTK)
    # print(f'The eigenvalues of the ensemble average NTK matrix are {ens_zero_eigenvalues} ')


    # NTK_points = 1/ens_zero_eigenvalues #Points of interest are described by the inverse values of the eigenvalues
    # print(f"The specific points of interest from the initial NTK axes is {NTK_points}")

    # """Need to remove any points which are outside the maximum epoch range"""
    
    # NTK_points = NTK_points[NTK_points <= args.Epochs].real #Only care about real eigenvalues, filter out any points which are beyond our training range
    ensemble_means = {i:[] for i in range(num_layers)}
    ensemble_uncertainty = {i:[] for i in range(num_layers)}

    alignment_uncert = np.std(alignment_array, axis = 0)/np.sqrt(args.EnsembleNum)
    alignment_mean = np.mean(alignment_array, axis = 0)
    
    """Now calculating the alignment of the eigenvectors
    recall shape of evec_alignment array is of shape [training_time, ensemble, eigenvector alignment]
    """
    evec_alignment_uncert = np.std(evec_alignment, axis = 1)/np.sqrt(args.EnsembleNum)
    evec_alignment_mean = np.mean(evec_alignment, axis = 1)

    """Calculating the eigenvalues"""
    eval_2_uncert = np.std(eval_array, axis =1)/np.sqrt(args.EnsembleNum)
    eval_2_mean = np.mean(eval_array,axis =1)


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


    # """Printing out the NTK matrix"""
    # train_vals = X_train.squeeze().numpy().round(2)
    # NTK_df = pd.DataFrame(ens_zero_NTK, index=train_vals, columns=train_vals)
    # print(NTK_df.to_string(float_format=lambda x: f'{x:.2f}'))

    # NTK_df.to_csv('NTK_matrix.csv', float_format='%.2f')
    #Saving the csv^^

    """End of calculations"""
#####################################################################################################################################

    """Starting plots"""
    fig, ax = plt.subplots(figsize=(10, 6)) 
    for k in range(len(ensemble_means)): 
        epochs_axis = range(1,args.Epochs) #Note, we start at 1 since using a np.diff finite difference schema
        train_time_axis = np.array(epochs_axis) * args.lr
        mean = ensemble_means[k]
        #std = ensemble_uncertainty[k]
        ax.plot(train_time_axis, mean, label=f'Layer {k+1}') #Convention is to use layer 0 as input, so need to shift everything up by 1
        #ax.fill_between(epochs_axis, mean - std, mean + std, alpha=0.3)

    #Adding important regions to plot
    for j in range(len(NTK_points)):
        ax.axvspan(NTK_points[j] - NTK_point_uncert[j], NTK_points[j] + NTK_point_uncert[j], color='purple', alpha=0.3)
        #Axvspan expects a single point, cannot use an array, hence need the for loop    
    ax.set_xlabel('Training Time')
    ax.set_ylabel('Pre-Activation Derivative')
    ax.set_title('Finite Difference of Layer Pre-Activations')
    ax.legend()
    plt.tight_layout()
    if args.SaveFig:
        plt.savefig(fr'C:\Users\Logan\Downloads\SummerWork\{args.Filename}\Activity')
    plt.show()
    plt.close()

    """Excluding the final layer"""
    fig, ax = plt.subplots(figsize=(10, 6))
    for k in range(len(ensemble_means)-1): 
        epochs_axis = range(1,args.Epochs) #Note, we start at 1 since using a np.diff finite difference schema
        train_time_axis = np.array(epochs_axis) * args.lr
        mean = ensemble_means[k]
        #std = ensemble_uncertainty[k]
        ax.plot(train_time_axis, mean, label=f'Layer {k+1}')
    ax.set_xlabel('Training Time')
    ax.set_ylabel('Pre-Activation Derivative')
    ax.set_title('Finite Difference of Layer Pre-Activations')
    for j in range(len(NTK_points)):
        ax.axvspan(NTK_points[j] - NTK_point_uncert[j], NTK_points[j] + NTK_point_uncert[j], color='purple', alpha=0.3)
        #Axvspan expects a single point, cannot use an array, hence need the for loop
    ax.legend()
    plt.tight_layout()
    if args.SaveFig:
        plt.savefig(fr'C:\Users\Logan\Downloads\SummerWork\{args.Filename}\ActivityNoOuter')
    plt.show()
    plt.close()


    """NTK alignment value"""
    print(f'The dimensionality of the alignment_mean array is {np.shape(alignment_mean)}')
    """Plotting alignment values"""
    plt.figure(figsize=(8,6))
    train_axis_alignment = args.lr * np.arange(0,args.Epochs,100)
    plt.plot(train_axis_alignment,alignment_mean)
    plt.fill_between(train_axis_alignment, alignment_mean +alignment_uncert, alignment_mean-alignment_uncert, alpha = 0.3) #Error gargantuan, not sure why
    #plt.fill_between(train_axis_alignment, alignment_mean +.5, alignment_mean-.5, alpha = 0.3)
    plt.xlabel(f'Training time')
    plt.ylabel(f'Alignment')
    plt.title(f'Alignment of the NTK vs Training Time')
    if args.SaveFig:
        plt.savefig(fr'C:\Users\Logan\Downloads\SummerWork\{args.Filename}\NTKAlignment')
    plt.show()
    plt.close()

    """Plotting the eigenvector alignment value"""
    plt.figure(figsize=(8,6))
    for k in range(evec_alignment_mean.shape[1]):
        plt.plot(train_axis_alignment,evec_alignment_mean[:,k], label = f"Eigenvector {k+1}")
        plt.fill_between(train_axis_alignment, evec_alignment_mean[:,k]+evec_alignment_uncert[:,k],evec_alignment_mean[:,k]-evec_alignment_uncert[:,k], alpha = 0.3 )
    plt.xlabel(f"Training time")
    plt.ylabel(f"Normalised eigenvector alignment value")
    plt.title(f"2 Nonzero Eigenvector Alignment Values vs Training Time")
    plt.legend()
    if args.SaveFig:
        plt.savefig(fr'C:\Users\Logan\Downloads\SummerWork\{args.Filename}\EigenvectorAlignment')
    plt.show()
    plt.close()

    """Plotting the eigenvalues"""
    plt.figure(figsize=(8,6))
    for k in range(evec_alignment_mean.shape[1]):
        plt.plot(train_axis_alignment,eval_2_mean[:,k], label = f"Eigenvalue {k+1}")
        plt.fill_between(train_axis_alignment, eval_2_mean[:,k]+eval_2_uncert[:,k], eval_2_mean[:,k]-eval_2_uncert[:,k], alpha = 0.3 )
    plt.xlabel(f"Training time")
    plt.ylabel(f"Eigenvalue")
    plt.title(f"2 Nonzero Eigenvalues vs Training Time")
    plt.legend()
    if args.SaveFig:
        plt.savefig(fr'C:\Users\Logan\Downloads\SummerWork\{args.Filename}\Eigenvalues')
    plt.show()
    plt.close()

    """Plotting Losses"""
    training_axis_loss = np.arange(args.Epochs) * args.lr
    ensemble_loss = np.mean(loss_array, axis=1)
    ensemble_loss_uncert = np.std(loss_array, axis = 1)/np.sqrt(args.EnsembleNum)
    plt.plot(training_axis_loss,ensemble_loss)
    plt.fill_between(training_axis_loss,ensemble_loss+ensemble_loss_uncert,ensemble_loss-ensemble_loss_uncert, alpha = 0.3)
    plt.xlabel(f'Training time')
    plt.ylabel(f"Average ensemble loss value")
    plt.title(f'Ensemble loss vs training time')
    if args.SaveFig:
        plt.savefig(fr'C:\Users\Logan\Downloads\SummerWork\{args.Filename}\Loss')
    plt.show()
    plt.close()

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
        plt.title(f'Performance of the model at training time {epoch_value * args.lr}')
        plt.legend()
        if args.SaveFig:
            plt.savefig(fr'C:\Users\Logan\Downloads\SummerWork\{args.Filename}\Performance{k+1}')
        plt.show()
        plt.close() 
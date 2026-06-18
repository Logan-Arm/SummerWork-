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

"""Creating a function to remove all hooks from the model"""
def remove_hooks(hooks):
    for hook in hooks:
        hook.remove()
    hooks.clear()

"""Now defining a loss function"""

def criterion(y_pred, y_true):
    #mean squared loss is simply (predicted-actual)^2
    return torch.mean((y_pred-y_true)**2)
    #Note, in the NN_EFT notes there is a 1/2 which is in front of this term, this is not necessary here as the torch autograd factors this in automatically


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
        

    
class Trial():
    def __init__(self, input,output,width,depth,lr,epochs,STD,ensemblenum,performances,bootstraps,alignmentint,X_train,Y_train,x_test,y_test,filename, SaveFig):
        self.input = input
        self.output = output
        self.width = width
        self.depth = depth
        self.lr = lr
        self.epochs = epochs
        self.std = STD
        self.ensemble = ensemblenum
        self.performances = performances
        self.bootstraps = bootstraps
        self.alignmentint = alignmentint
        self.x_train = X_train
        self.y_train = Y_train
        self.x_test = x_test
        self.y_test = y_test
        self.Filename = filename
        self.SaveFig = SaveFig
        self.numlayer = self.depth+1

        self.step = int(self.epochs/self.performances)

        self.ensemble_derivs = {i:[] for i in range(self.numlayer)}
        
        self.Y_vector = self.y_train.numpy()
        self.NTK_zero = [] #Class object to represent the initial NTK
        self.norm_const = np.dot(self.Y_vector.T, self.Y_vector)

        self.evec_alignment = np.zeros((self.epochs//self.alignmentint, self.ensemble, 5)) #Class object to hold all the eigenvector alignment values
        self.evalues = np.zeros((self.epochs//self.alignmentint, self.ensemble, 5))

        self.NTK_alignment = np.zeros((self.ensemble,self.epochs//self.alignmentint)) #Array to store the NTK alignment values
        self.loss_array = np.zeros((self.epochs,self.ensemble))
        self.performance_array = np.zeros((self.performances,self.ensemble,len(self.x_test)))

        self.train_time_rate = np.array(range(1,self.epochs)) *self.lr #To be used for any derivative calculations, i.e., excluding T=0

        self.train_time_total = np.arange(self.epochs) *self.lr #To be used for any plot which requires the total training time

        self.train_time_alignment = self.lr * np.arange(0,self.epochs,100) #To be used for any alignment calculations



        
    def train_model(self,j):

        """Note, that since this is being stored as an instant variable, if we wanted to run anything in parallel, we would have to change this to be model, 
        i.e., not a class variable
        """
        self.model = MultiLayerNet(self.input,self.depth,self.width,self.output,self.std).double()
        activation_history = {i: [] for i in range(len(self.model.hidden_layers)+1)}
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
        for i, layer in enumerate(self.model.hidden_layers): #Note this does not include the output layer
            hooks.append(layer.register_forward_hook(make_hook(i)))

        hooks.append(self.model.output_layer.register_forward_hook(make_hook(len(self.model.hidden_layers)))) #This will be the last output hook

        """Training the model"""
        optimiser  = torch.optim.SGD(self.model.parameters(), lr = self.lr)
        """Note on gradient descent... since we are using the full batch of data, this is actually a regular gradient descent not stoichastic 
        i.e.,  (no random sampling)"""

        for epoch in range(self.epochs):
            if epoch==0: #appends the first matrix
                remove_hooks(hooks)
                ntk_matrix = NTK_calc(self.model,self.x_train).numpy()
                self.NTK_zero.append(ntk_matrix)
                for i, layer in enumerate(self.model.hidden_layers):
                    hooks.append(layer.register_forward_hook(make_hook(i)))
                hooks.append(self.model.output_layer.register_forward_hook(make_hook(len(self.model.hidden_layers))))


            if epoch%100==0 and epoch!=0: #Only include NTK calculations every 100 epochs
                remove_hooks(hooks)
                ntk_matrix = NTK_calc(self.model,self.x_train).numpy()
                
                """Calculating the eigenvalues, eigenvectors, and alignment"""
                evals, evecs = np.linalg.eigh(ntk_matrix) #calculates eigenvectors and eigenvales
                selected_indices = np.argsort(evals)[-5:] #Takes last 5 values of the sorted eigenvalues
                selected_evecs = evecs[:,selected_indices]
                selected_evals = evals[selected_indices]
                for z in range(selected_evecs.shape[1]):
                    self.evec_alignment[epoch//self.alignmentint, j, z] = np.dot(selected_evecs[:,z].T,self.Y_vector)/np.linalg.norm(self.Y_vector) #Calculates the eigenvalue alignment i.e., e^T Y
                    self.evalues[epoch//self.alignmentint, j, z] = selected_evals[z] #Appends the eigenvalue
                alignment_item = np.dot(self.Y_vector,ntk_matrix)
                ntk_norm = np.linalg.norm(ntk_matrix,'fro') #Calculates the frobenius norm of the matrix

                """Re-adding the hooks"""
                self.NTK_alignment[j,epoch//self.alignmentint] = np.dot(alignment_item,self.Y_vector.T)/(self.norm_const * ntk_norm) 
                for i, layer in enumerate(self.model.hidden_layers):
                    hooks.append(layer.register_forward_hook(make_hook(i)))
                hooks.append(self.model.output_layer.register_forward_hook(make_hook(len(self.model.hidden_layers))))
            
            """Start of training loop"""
            self.model.train()
            y_pred = self.model(self.x_train)
            loss = criterion(y_pred.squeeze(),self.y_train)
            loss.backward()
            optimiser.step()
            optimiser.zero_grad()
            self.loss_array[epoch, j]=loss.item()
            """End of training loop"""
            
            
            
            if (epoch+1)%self.step ==0:
                """This will record the model performance at selected epochs
                Note that since the values calculated here are separate from the derivatives, we must remove the hooks, and we must also put the model back into training mode
                """
                remove_hooks(hooks)
                epoch_count = int((epoch+1)/self.step) - 1
                self.model.eval()
                with torch.no_grad(): 
                    Y_test_pred = self.model(self.x_test)
                for i in range(len(Y_test_pred)):
                    self.performance_array[epoch_count,j,i] = Y_test_pred[i]

                """Now to reattach hooks and return the model to training mode"""
                for i, layer in enumerate(self.model.hidden_layers):
                    hooks.append(layer.register_forward_hook(make_hook(i)))
                hooks.append(self.model.output_layer.register_forward_hook(make_hook(len(self.model.hidden_layers))))
        for i in range(self.numlayer):
            
            stacked = torch.stack(activation_history[i]).numpy() #Converts it to a numpy array
            #above has the following dimensionality: [epochs, batch (or data point), neuron]
            #axis 0 = epochs
            #axis 1 = batch/data
            #axis 2 = neuron
            time_deriv = np.diff(stacked, 1, 0) #Takes first derivative along the epoch axis
            print(f'time_deriv shape: {time_deriv.shape}') #Sanity check
            self.ensemble_derivs[i].append(time_deriv)
            #appends the layer information to the corresponding layer list in the ensemble_derivs dictionary
            #New dimensionality would be the following:
            #[layer] [ensemble_num, epoch-1 (since taken derivative), batch, neuron]
    
    def compute_data(self):

        
        NTK_stacked = np.stack(self.NTK_zero, axis = 0) #has dimensionality [Ensemblenum, N,N]
        ens_zero_NTK = np.mean(NTK_stacked, axis = 0)

        
        bootstrapped_eigvals = np.zeros((self.bootstraps, len(self.x_train))) #only care about the 2 nonzero eigenvalues
        bootstrapped_eigvecs = np.zeros((self.bootstraps, len(self.x_train), len(self.x_train)))#only care about the 2 eigenvectors corresponding the 2 nonzero eigenvalues
        """Bootstrapping loop"""
        for b in range(self.bootstraps):
            #Recall NTK_stacked zero axis is our replica axis
            boots_indices = np.random.choice(NTK_stacked.shape[0], size = NTK_stacked.shape[0], replace = True)
            #Randomly resampling (with replacement) which replicas to include
            boot_mean_NTK = np.mean(NTK_stacked[boots_indices], axis = 0) 
            #Taking mean of this resampled
            boot_mean_NTK = (boot_mean_NTK + boot_mean_NTK.T)/2 #Making explicitly symmetric, recall that the NTK is supposed to be symmetric
            boot_eigvals, boot_eigvecs = np.linalg.eigh(boot_mean_NTK)
            boot_sorted_indices = np.argsort(boot_eigvals) #Grabs last 2 points, i.e., the nonzero eigenvalue
            boot_eigvals = boot_eigvals[boot_sorted_indices]
            boot_eigvecs = boot_eigvecs[:,boot_sorted_indices] #Need slice to actually take the vector
            bootstrapped_eigvals[b] = boot_eigvals
            # for k in range(len(boot_eigvals)):
            #     bootstrapped_eigvecs[b,k,:] = boot_eigvecs[:,k] 
        
        
        mean_eigenvals = np.mean(bootstrapped_eigvals, axis = 0)
        mean_eigenvals_std = np.std(bootstrapped_eigvals, axis =0)
        print(f"The eigenvalues of interest are {mean_eigenvals}")
        self.NTK_points = 1/mean_eigenvals
        self.NTK_point_uncert = mean_eigenvals_std/(mean_eigenvals**2)

        """Filtering out any eigenvalues which are negative, this is almost always due to floating point instability"""
        if np.any(mean_eigenvals< 0):
            print(f"Negative eigenvalue in list")
        print(f'The eigenvalues of the ensemble average NTK matrix are {mean_eigenvals} ')
        print(f"The specific points of interest from the initial NTK axes is {self.NTK_points}")

        """Need to remove any points which are outside the maximum epoch range"""
        max_time = self.epochs * self.lr #Adjust it to be in terms of learning rate
        self.NTK_points = self.NTK_points[self.NTK_points <= max_time].real #Only care about real eigenvalues, filter out any points which are beyond our training range
        self.ensemble_means = {i:[] for i in range(self.numlayer)}
        self.ensemble_uncertainty = {i:[] for i in range(self.numlayer)}

        self.alignment_uncert = np.std(self.NTK_alignment, axis = 0)/np.sqrt(self.ensemble)
        self.alignment_mean = np.mean(self.NTK_alignment, axis = 0)

        """Sanity checker"""
        for k in range(self.numlayer):
                    print(f'Layer {k}: ensemble_derivs[{k}] has {len(self.ensemble_derivs[k])} entries, first entry shape: {self.ensemble_derivs[k][0].shape}')

        for i in range(self.numlayer):
            print(f"On ensemble calculations for layer {i}")

            stacked_ensemble = np.stack(self.ensemble_derivs[i], axis = 0) #Shape is now [ensemble, epochs-1, neurons]
            """What is going on in above line:
            We are taking the FIRST ELEMENT of the ensemble_derivs, recall from line 248 that this corresponds to selecting a specific layer
            We then stack all these list elements into one numpy array along the 0th axis of the components, i.e., along the ensemble_num axis
            Therefore we have one numpy array for each layer, of the shape [ensemble, epoch -1, batch, neuron ]
            """

            #Take the mean value with respect to the ensembles
            ens_mean = np.abs(np.mean(stacked_ensemble, axis = 0)) #New dimensionality is [epochs-1,batch, neurons]

            """Come back to the uncertainty, not totally sure how to calculate it"""

            #ens_mean_sq = ens_mean**2 #Values should (in theory) scale to 1/n, so to square it is just going to shrink it immensely. work with absolute vals instead

            #Now take the mean over the remaining axes, i.e., batches first, then calculate error, then take remaining mean
            ensemble_means_neurons = ens_mean.mean(axis = 1)
            #Now have something of shape [epochs -1, neurons]


            # #Want to calculate the bootstrap uncertainty for this list
            # for j in range(ensemble_means_neurons.shape[0]):
            #     ensemble_uncertainty[i].append(Bootstrap_Analysis(ensemble_means_neurons[j,:]))
            """Above is commented out initially to simply get a plot without errors to save computation time"""

            self.ensemble_means[i] = np.mean(ensemble_means_neurons,axis =1) #Now averaging over all the neurons
            # results = Parallel(n_jobs= -1)(delayed(Bootstrap_Analysis)(ensemble_means_neurons[j,:]) for j in range(ensemble_means_neurons.shape[0])
            #                                )
            # ensemble_uncertainty[i] = np.array(results)
        
            #ensemble_uncertainty[i] = np.array(ensemble_uncertainty[i]) #Need it to be a numpy array
            #What is remaining is a list of dimensionality [epochs-1, means], this is the appended to the ith component of the ensemble_means dictionary

        """Calculating ensemble values of eigenvector alignment
        Recall evec_alignment has shape [time, ensemble, evec]
        """
        self.evec_alignment_uncert = np.std(self.evec_alignment, axis = 1)/np.sqrt(self.ensemble)
        self.evec_alignment_mean = np.mean(self.evec_alignment, axis = 1)            
        """Calculating the 5 largest eigenvalues"""
        self.eval_5_uncert = np.std(self.evalues,axis = 1)/np.sqrt(self.ensemble)
        self.eval_5_array = np.mean(self.evalues, axis =1)

    def make_plots(self):

        fig, ax = plt.subplots(figsize=(10, 6)) 
        for k in range(len(self.ensemble_means)): 
            mean = self.ensemble_means[k]
            #std = ensemble_uncertainty[k]
            ax.plot(self.train_time_rate, mean, label=f'Layer {k+1}') #Convention is to use layer 0 as input, so need to shift everything up by 1
            #ax.fill_between(epochs_axis, mean - std, mean + std, alpha=0.3)

        #Adding important regions to plot
        for j in range(len(self.NTK_points)):
            #ax.axvspan(self.NTK_points[j] - self.NTK_point_uncert[j], self.NTK_points[j] + self.NTK_point_uncert[j], color='purple', alpha=0.3)
            ax.axvline(self.NTK_points[j], color = 'purple', alpha = 0.3)
            #Axvspan expects a single point, cannot use an array, hence need the for loop    
        ax.set_xlabel('Training Time')
        ax.set_ylabel('Pre-Activation Derivative')
        ax.set_title('Finite Difference of Layer Pre-Activations')
        ax.legend()
        plt.tight_layout()
        if self.SaveFig:
            plt.savefig(fr'C:\Users\Logan\Downloads\SummerWork\{self.Filename}\Activity')
        plt.show()
        plt.close()

        """Excluding the final layer"""
        fig, ax = plt.subplots(figsize=(10, 6))
        for k in range(len(self.ensemble_means)-1): 
            mean = self.ensemble_means[k]
            #std = ensemble_uncertainty[k]
            ax.plot(self.train_time_rate, mean, label=f'Layer {k+1}')
        ax.set_xlabel('Training Time')
        ax.set_ylabel('Pre-Activation Derivative')
        ax.set_title('Finite Difference of Layer Pre-Activations')
        for j in range(len(self.NTK_points)):
            # ax.axvspan(self.NTK_points[j] - self.NTK_point_uncert[j], self.NTK_points[j] + self.NTK_point_uncert[j], color='purple', alpha=0.3)
            ax.axvline(self.NTK_points[j], color = 'purple', alpha = 0.3)
            #Axvspan expects a single point, cannot use an array, hence need the for loop
        ax.legend()
        plt.tight_layout()
        if self.SaveFig:
            plt.savefig(fr'C:\Users\Logan\Downloads\SummerWork\{self.Filename}\ActivityNoOuter')
        plt.show()
        plt.close()

        """NTK alignment value"""
        print(f'The dimensionality of the alignment_mean array is {np.shape(self.alignment_mean)}')
        """Plotting alignment values"""
        plt.figure(figsize=(8,6))
        plt.plot(self.train_time_alignment,self.alignment_mean)
        plt.fill_between(self.train_time_alignment, self.alignment_mean +self.alignment_uncert, self.alignment_mean-self.alignment_uncert, alpha = 0.3)
        plt.xlabel(f'Training time')
        plt.ylabel(f'Alignment')
        plt.title(f'Alignment of the NTK vs Training Time')
        if self.SaveFig:
            plt.savefig(fr'C:\Users\Logan\Downloads\SummerWork\{self.Filename}\NTKAlignment')
        plt.show()
        plt.close()

        """Plotting the eigenvector alignment value"""
        plt.figure(figsize=(8,6))
        for k in range(self.evec_alignment_mean.shape[1]):
            plt.plot(self.train_time_alignment,self.evec_alignment_mean[:,k], label = f"Eigenvector {k+1}")
            plt.fill_between(self.train_time_alignment, self.evec_alignment_mean[:,k]+self.evec_alignment_uncert[:,k],self.evec_alignment_mean[:,k]-self.evec_alignment_uncert[:,k], alpha = 0.3 )
        plt.xlabel(f"Training time")
        plt.ylabel(f"Normalised eigenvector alignment value")
        plt.title(f"5 Largest Eigenvector Alignment Values vs Training Time")
        plt.legend()
        if self.SaveFig:
            plt.savefig(fr'C:\Users\Logan\Downloads\SummerWork\{self.Filename}\EigenvectorAlignment')
        plt.show()
        plt.close()


        """Plotting the top 5 eigenvalues"""
        plt.figure(figsize=(8,6))
        for k in range(self.evec_alignment_mean.shape[1]):
            plt.plot(self.train_time_alignment,self.eval_5_array[:,k], label = f"Eigenvalue {k+1}")
            plt.fill_between(self.train_time_alignment, self.eval_5_array[:,k] + self.eval_5_uncert[:,k],self.eval_5_array[:,k] - self.eval_5_uncert[:,k], alpha = 0.3 )
        plt.xlabel(f"Training time")
        plt.ylabel(f" Eigenvalue")
        plt.title(f"5 Largest Eigenvalues vs Training Time")
        plt.legend()
        if self.SaveFig:
            plt.savefig(fr'C:\Users\Logan\Downloads\SummerWork\{self.Filename}\Top5Evals')
        plt.show()
        plt.close()


        """Plotting Losses"""
        ensemble_loss = np.mean(self.loss_array, axis=1)
        ensemble_loss_uncert = np.std(self.loss_array, axis = 1)/np.sqrt(self.ensemble)
        plt.plot(self.train_time_total,ensemble_loss)
        plt.fill_between(self.train_time_total,ensemble_loss+ensemble_loss_uncert,ensemble_loss-ensemble_loss_uncert, alpha = 0.3)
        for j in range(len(self.NTK_points)):
            #plt.axvspan(self.NTK_points[j] - self.NTK_point_uncert[j], self.NTK_points[j] + self.NTK_point_uncert[j], color='purple', alpha=0.3)
            plt.axvline(self.NTK_points[j], color = 'purple', alpha = 0.3)
        plt.xlabel(f'Training time')
        plt.ylabel(f"Average ensemble loss value")
        plt.title(f'Ensemble loss vs training time')
        if self.SaveFig:
            plt.savefig(fr'C:\Users\Logan\Downloads\SummerWork\{self.Filename}\Loss')
        plt.show()
        plt.close()

        """Plotting Performance"""
        #Need to include training points here
        mean_ensemble_val = np.mean(self.performance_array,axis=1)
        ensemble_uncert = np.std(self.performance_array,axis = 1)/np.sqrt(self.ensemble)
        x_vals = self.x_test.squeeze().numpy()
        for k in range(self.performance_array.shape[0]):
            epoch_value = (k+1)*self.step
            plt.figure(figsize=(8,6))
            plt.plot(x_vals, mean_ensemble_val[k,:], label = f'Predicted Values')
            plt.plot(x_vals,self.y_test.numpy(), label = f'True Values')
            plt.scatter(self.x_train,self.y_train,label = "Training points")
            plt.fill_between(x_vals, mean_ensemble_val[k,:]+ensemble_uncert[k,:],mean_ensemble_val[k,:]-ensemble_uncert[k,:],color = 'blue', alpha = 0.3)
            #plt.fill_between is finicky, needs inputs to be explicitly 1 dimensional, hence the X_test_sorted.squeeze().numpy()
            plt.xlabel(f'X')
            plt.ylabel(f'Y')
            plt.title(f'Performance of the model at training time {epoch_value * self.lr}')
            plt.legend()
            if self.SaveFig:
                plt.savefig(fr'C:\Users\Logan\Downloads\SummerWork\{self.Filename}\Performance{k+1}')
            plt.show()
            plt.close() 



        
    
    def run_no_plot(self):
        for j in range(self.ensemble):
            self.train_model(j)
        self.compute_data()

    def run_plot(self):
        for j in range(self.ensemble):
            self.train_model(j)
        self.compute_data()
        self.make_plots()
    
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
    parser.add_argument('--SaveFig', action='store_true', help='If set, saves figures to args.Filename')
    parser.add_argument('--Filename', type = str, help='Determines the file to save data to', default= 'Unsorted')
    args = parser.parse_args()

    trial = Trial(args.InputSize, args.OutputSize,args.HiddenLayerWidth,args.HiddenLayerDepth,args.lr,args.Epochs,args.STD,args.EnsembleNum,args.Performances,
                  args.Bootstraps,args.AlignmentInterval,X_train_sorted,Y_train_sorted,X_eval,Y_eval,args.Filename,args.SaveFig)
    trial.run_plot()
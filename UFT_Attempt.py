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


class MultiLayerNet_Linear(nn.Module): #nn.module is the base class for all neural networks in pytorch


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
            x= (layer(x))
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
    def __init__(self, input,output,width,depth,lr,epochs,STD,ensemblenum,performances,bootstraps,
                 alignmentint,X_train,Y_train,x_test,y_test,filename, SaveFig, regions, linear, eval_amount,
                 performance_times):
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
        self.regions = regions
        self.linear = linear
        self.e_amount = eval_amount

        # self.onnx_filename = onnx_filename

        if performance_times is None:
            # self.snapshot_array = np.geomspace(1,self.epochs,self.performances).astype(int)
            self.snapshot_array = np.unique(np.geomspace(1, self.epochs - 1, self.performances).astype(int))
            self.step = int(self.epochs/self.performances)
            self.performances = len(self.snapshot_array) #Just adjusts the value in case the .astype(int) messes with the values
        else:
            self.snapshot_array = performance_times
            self.performances = len(performance_times)

        self.ensemble_derivs = {i:[] for i in range(self.numlayer)}

        self.phi = {i:[] for i in range(self.numlayer)}
        
        self.Y_vector = self.y_train.numpy()
        self.NTK_zero = [] #Class object to represent the initial NTK
        self.norm_const = np.dot(self.Y_vector.T, self.Y_vector)

        self.evec_alignment = np.zeros((self.epochs//self.alignmentint, self.ensemble, self.e_amount)) #Class object to hold all the eigenvector alignment values
        self.evalues = np.zeros((self.epochs//self.alignmentint, self.ensemble, self.e_amount))

        self.NTK_alignment = np.zeros((self.ensemble,self.epochs//self.alignmentint)) #Array to store the NTK alignment values
        self.loss_array = np.zeros((self.epochs,self.ensemble))
        self.performance_array = np.zeros((self.performances,self.ensemble,len(self.x_test)))

        self.train_time_rate = np.array(range(1,self.epochs)) *self.lr #To be used for any derivative calculations, i.e., excluding T=0

        self.train_time_total = np.arange(self.epochs) *self.lr #To be used for any plot which requires the total training time

        self.train_time_alignment = self.lr * np.arange(0,self.epochs,self.alignmentint) #To be used for any alignment calculations

        self.eval2_active_times = []
        self.eval3_active_times = []


        self.eval1_rateofchange = np.zeros((self.ensemble, self.epochs//self.alignmentint))
        self.eval2_rateofchange = np.zeros((self.ensemble, self.epochs//self.alignmentint))
        self.eval3_rateofchange = np.zeros((self.ensemble, self.epochs//self.alignmentint))

        self.eval1_deriv = np.zeros((self.ensemble, self.epochs//self.alignmentint))
        self.eval2_deriv = np.zeros((self.ensemble, self.epochs//self.alignmentint))
        self.eval3_deriv = np.zeros((self.ensemble, self.epochs//self.alignmentint))

    def calc_NTK_data(self,epoch,j):
        """Calculates the NTk for ntk alignment data or time = 0 NTK calculations
        Calculates the appropriate number of eigenvalues and eigenvectors from self.e_amount
        j corresponds to the ensemblenum
        """
        remove_hooks(self.hooks)# Note this was originally just hooks, chaning it to self.hooks
        self.ntk_matrix = NTK_calc(self.model, self.x_train).numpy()
        
        ntk_norm = np.linalg.norm(self.ntk_matrix, 'fro')

        evals, evecs = np.linalg.eigh(self.ntk_matrix) #calculates eigenvectors and eigenvalues
        selected_indices = np.argsort(evals)[-self.e_amount:][::-1] #Takes last 2 values of the sorted eigenvalues
        #[::-1] puts in descending order
        self.selected_evecs = evecs[:, selected_indices]
        self.selected_evals = evals[selected_indices]
        for z in range(self.e_amount):
            self.evec_alignment[epoch//self.alignmentint, j, z] = np.dot(self.selected_evecs[:, z].T, self.Y_vector)/(np.linalg.norm(self.Y_vector)) #Normalises
            self.evalues[epoch//self.alignmentint, j, z] = self.selected_evals[z]
        alignment_item = np.dot(self.Y_vector, self.ntk_matrix)
        self.NTK_alignment[j, epoch//self.alignmentint] = np.dot(alignment_item, self.Y_vector.T)/(self.norm_const * ntk_norm) #ntk norm is frobenius

    def time_adjust_evals(self):
        self.eval1_prev = self.selected_evals[0]
        self.eval2_prev = self.selected_evals[1]
        self.eval3_prev = self.selected_evals[2]
    

    def eval_data_acquire(self,epoch,j):
        self.eval1_deriv[j,epoch//self.alignmentint] = np.abs(self.selected_evals[0] - self.eval1_prev)/(self.alignmentint*self.lr)
        self.eval2_deriv[j,epoch//self.alignmentint] = np.abs(self.selected_evals[1] - self.eval2_prev)/(self.alignmentint*self.lr)
        self.eval3_deriv[j,epoch//self.alignmentint] = np.abs(self.selected_evals[2] - self.eval3_prev)/(self.alignmentint*self.lr)

        #Absolute relative rate of change
        self.eval1_rateofchange[j,epoch//self.alignmentint] = np.abs(self.selected_evals[0] - self.eval1_prev)/(self.eval1_prev)
        self.eval2_rateofchange[j,epoch//self.alignmentint] = np.abs(self.selected_evals[1] - self.eval2_prev)/(self.eval2_prev)
        self.eval3_rateofchange[j,epoch//self.alignmentint] = np.abs(self.selected_evals[2] - self.eval3_prev)/(self.eval3_prev)


    def train_model(self,j):
        self.epoch_index = 0 #To be used for the performance plots

        # self.eval2_activity_j = []
        # self.eval3_activity_j = []

        """Note, that since this is being stored as an instant variable, if we wanted to run anything in parallel, we would have to change this to be model, 
        i.e., not a class variable
        """
        if self.linear == False:
            self.model = MultiLayerNet(self.input,self.depth,self.width,self.output,self.std).double()
        else:
            self.model = MultiLayerNet_Linear(self.input,self.depth,self.width,self.output,self.std).double()
        activation_history = {i: [] for i in range(len(self.model.hidden_layers)+1)}
        """Creates a dictionary of lists corresponding to each layer, where we will store the activation history to calculate finite differences"""



        def make_hook(layer_id):
            """This NEEDS to be a nested function to safely pass the layer_id value to the hook function, which only ever takes the module, input and output as function inputs"""
            def hook(module, input, output):
                activation_history[layer_id].append(output.detach())     
                """Appends the output of the layer (before tanh activation) to the corresponding list in the activation history
                .detach() is needed as the output is technically a tensor (which the dictionary will not accept/work well with)
                """           
            return hook
        
        #hooks =[] #Making self.hooks instead
        self.hooks = []
        
        """Note, while the hooks below are registered forward and not register_forward_pre_hook, the model is set up to calculate the layers weighting and biases upon the data
        which is then passed to the tanh function.  Since this is done separately, (i.e., no nn.sequential()), this is technically a pre-activation value, as it is done
        prior to tanh being called.      
        However it is good to be aware of why register_forward_ is being used in this case
        """
        for i, layer in enumerate(self.model.hidden_layers): #Note this does not include the output layer
            self.hooks.append(layer.register_forward_hook(make_hook(i)))

        self.hooks.append(self.model.output_layer.register_forward_hook(make_hook(len(self.model.hidden_layers)))) #This will be the last output hook

        """Training the model"""
        optimiser  = torch.optim.SGD(self.model.parameters(), lr = self.lr)
        """Note on gradient descent... since we are using the full batch of data, this is actually a regular gradient descent not stoichastic 
        i.e., no random sampling"""

        for epoch in range(self.epochs):
            if epoch==0: #appends the first matrix
                self.calc_NTK_data(epoch, j)
                self.time_adjust_evals()
                print(f"The 0th value of eval3 is {self.eval3_prev}")
                self.NTK_zero.append(self.ntk_matrix)
                
                #Re-adding the hooks
                for i, layer in enumerate(self.model.hidden_layers):
                    self.hooks.append(layer.register_forward_hook(make_hook(i)))
                self.hooks.append(self.model.output_layer.register_forward_hook(make_hook(len(self.model.hidden_layers))))

            

            if epoch%self.alignmentint==0 and epoch!=0: #Only include NTK calculations every alignmentint epochs
                self.calc_NTK_data(epoch, j)
                #Derivative and abs rel change calcs
                self.eval_data_acquire(epoch,j)
                #Adjust the definitions for future relative change algorithm
                self.time_adjust_evals()

                """Re-adding the hooks"""
                for i, layer in enumerate(self.model.hidden_layers):
                    self.hooks.append(layer.register_forward_hook(make_hook(i)))
                self.hooks.append(self.model.output_layer.register_forward_hook(make_hook(len(self.model.hidden_layers))))
            


            """Exporting the model to an onnx format
            Recall that doing so runs a tracer through the model (to properly assess the architecture)
            As such, we NEED to remove the hooks
            We do not however need to place in torch.no_grad() since we are only running a tracer there is no backwards call or updating of the loss landscape
            """
            # if j ==0 and epoch ==100:
            #     #Clear the hooks
            #     remove_hooks(self.hooks)
            #     self.export_onnx()
            #     for i, layer in enumerate(self.model.hidden_layers):
            #         self.hooks.append(layer.register_forward_hook(make_hook(i)))
            #     self.hooks.append(self.model.output_layer.register_forward_hook(make_hook(len(self.model.hidden_layers))))
            
            
            
            """Start of training loop"""
            self.model.train()
            y_pred = self.model(self.x_train)
            loss = criterion(y_pred.squeeze(),self.y_train)
            loss.backward()
            optimiser.step()
            optimiser.zero_grad()
            self.loss_array[epoch, j]=loss.item()
            """End of training loop"""
            
            
            
            if epoch in self.snapshot_array:
                """This will record the model performance at selected epochs
                Note that since the values calculated here are separate from the derivatives, we must remove the hooks, and we must also put the model back into training mode
                """
                remove_hooks(self.hooks)

                self.model.eval()
                with torch.no_grad():
                    Y_test_pred = self.model(self.x_test)
                for i in range(len(Y_test_pred)):
                    self.performance_array[self.epoch_index, j, i] = Y_test_pred[i]
                self.epoch_index+=1 #Increments the position in the array
                """Now to reattach hooks and return the model to training mode"""
                for i, layer in enumerate(self.model.hidden_layers):
                    self.hooks.append(layer.register_forward_hook(make_hook(i)))
                self.hooks.append(self.model.output_layer.register_forward_hook(make_hook(len(self.model.hidden_layers))))
                self.model.train()


        for i in range(self.numlayer):
            

            """Calculating the pre-activation derivative from the hooks' data"""
            self.stacked = torch.stack(activation_history[i]).numpy() #Converts it to a numpy array
            #above has the following dimensionality: [epochs, batch (or data point), neuron]
            #axis 0 = epochs
            #axis 1 = batch/data
            #axis 2 = neuron
            time_deriv = np.diff(self.stacked, 1, 0) #Takes first derivative along the epoch axis
            print(f'time_deriv shape: {time_deriv.shape}') #Sanity check
            self.ensemble_derivs[i].append(time_deriv)
            self.phi[i].append(self.stacked)
            #appends the layer information to the corresponding layer list in the ensemble_derivs dictionary
            #New dimensionality would be the following:
            #[layer] [ensemble_num, epoch-1 (since taken derivative), batch, neuron]

        # self.eval2_active_times.append(self.eval2_activity_j)
        # self.eval3_active_times.append(self.eval3_activity_j)
        # print(f"The number of times of activity for eval2 = {len(self.eval2_activity_j)}")
        # print(f"The number of times of activity for eval3 = {len(self.eval3_activity_j)}")

    def compute_eigvals_zero(self):
        NTK_stacked = np.stack(self.NTK_zero, axis = 0) #has dimensionality [Ensemblenum, N,N]
        ens_zero_NTK = np.mean(NTK_stacked, axis = 0)

        
        bootstrapped_eigvals = np.zeros((self.bootstraps, self.e_amount)) #only care about the top 5 eigenvals
        bootstrapped_eigvecs = np.zeros((self.bootstraps, len(self.x_train), self.e_amount))
        """Bootstrapping loop"""
        for b in range(self.bootstraps):
            #Recall NTK_stacked zero axis is our replica axis
            boots_indices = np.random.choice(NTK_stacked.shape[0], size = NTK_stacked.shape[0], replace = True)
            #Randomly resampling (with replacement) which replicas to include
            boot_mean_NTK = np.mean(NTK_stacked[boots_indices], axis = 0) 
            #Taking mean of this resampled
            boot_mean_NTK = (boot_mean_NTK + boot_mean_NTK.T)/2 #Making explicitly symmetric, recall that the NTK is supposed to be symmetric
            boot_eigvals, boot_eigvecs = np.linalg.eigh(boot_mean_NTK)
            boot_sorted_indices = np.argsort(boot_eigvals)[-self.e_amount:][::-1] #Grabs the last self.e points in descending order
            boot_eigvals = boot_eigvals[boot_sorted_indices]
            boot_eigvecs = boot_eigvecs[:,boot_sorted_indices] #Need slice to actually take the vector
            bootstrapped_eigvals[b] = boot_eigvals
            # for k in range(len(boot_eigvals)):
            #     bootstrapped_eigvecs[b,k,:] = boot_eigvecs[:,k] 
        
        
        self.mean_eigenvals = np.mean(bootstrapped_eigvals, axis = 0)
        self.mean_eigenvals_std = np.std(bootstrapped_eigvals, axis =0)
        self.mean_eigenvals_uncert = self.mean_eigenvals_std/np.sqrt(self.ensemble)
        print(f"The eigenvalues of interest are {self.mean_eigenvals}")

    def compute_phi_sq(self):
        """self.phi has the following dimensionality
        Self.phi is a dictionary with amount of lists = numlayers
        each list contains numpy arrays with shape [epochs,batch/data,neuron], corresponding to the activity in the selected layer for corresponding epochs etc

        We will stack these lists into one array, so the dimensionality will be [replica, epochs, batch/data, neuron], then we can simply calculate the mean and std
        """
        self.phi_sq = {i:[] for i in range(self.numlayer)}
        #self.phi_sq_uncert = {i:[] for i in range(self.numlayer)}
        for k in range(self.numlayer):
            stacked_ensemble = np.stack(self.phi[k], axis =0) #Dimensionality [replica, epoch, batch, neuron]
            stacked_ensemble = stacked_ensemble**2 #Recall we are interested in the square
            ensemble_mean = np.mean(stacked_ensemble, axis =0) #Dim is [epoch, batch, neuron]
            #now need to take mean over remaining dimensions
            self.phi_sq[k] = (np.mean(ensemble_mean, axis = (1,2)))

    def compute_NTK_pts(self):
        self.NTK_points = 1/self.mean_eigenvals
        self.NTK_point_uncert = self.mean_eigenvals_std/(self.mean_eigenvals**2)

        """Filtering out any eigenvalues which are negative, this is almost always due to floating point instability"""
        if np.any(self.mean_eigenvals< 0):
            print(f"Negative eigenvalue in list")
        print(f'The eigenvalues of the ensemble average NTK matrix are {self.mean_eigenvals} ')
        print(f"The specific points of interest from the initial NTK axes is {self.NTK_points}")

        """Need to remove any points which are outside the maximum epoch range"""
        max_time = self.epochs * self.lr #Adjust it to be in terms of learning rate
        self.NTK_points = self.NTK_points[self.NTK_points <= max_time].real #Only care about real eigenvalues, filter out any points which are beyond our training range
        self.NTK_points = self.NTK_points[self.NTK_points >0]

    def compute_preactivation_rate(self):
        self.ensemble_means = {i:[] for i in range(self.numlayer)}
        self.ensemble_uncertainty = {i:[] for i in range(self.numlayer)}
        """Sanity checker"""
        for k in range(self.numlayer):
                    print(f'Layer {k}: ensemble_derivs[{k}] has {len(self.ensemble_derivs[k])} entries, first entry shape: {self.ensemble_derivs[k][0].shape}')

        for i in range(self.numlayer):
            print(f"On ensemble calculations for layer {i}")

            stacked_ensemble = np.stack(self.ensemble_derivs[i], axis = 0) #Shape is now [ensemble, epochs-1,batch, neurons]
            """What is going on in above line:
            We are taking the FIRST ELEMENT of the ensemble_derivs, recall that this corresponds to selecting a specific layer
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

    def compute_chi(self):
        """Chi array will have one less value than ensemble, since layer 1 will not have a chi reading"""
        self.chi_array = {i:[] for i in range(1,self.numlayer)}
        """Calculating chi defined as ratio of pre-activation derivs"""
        for k in range(1,self.numlayer):
            ############################################################################
            if k ==self.numlayer:
                self.chi_array[k] = ((self.ensemble_means[k]/self.ensemble_means[k-1]))/10
                """Chi is dependent on number of neurons in layer, last output has 1/10th neurons so need to scale accordingly""" 
                """Should think of a better way to do this"""
            ############################################################################
            else:
                self.chi_array[k] = ((self.ensemble_means[k]/self.ensemble_means[k-1]))
            print(f"Length element is {len(self.ensemble_means[k]/self.ensemble_means[k-1])}") 

        print(f"The length of a chi list in the chi array is {len(self.chi_array[1])}")

    def compute_abs_rel_ROC(self):
        """Calculates the absolute relative rate of change of the top 3 eigenvalues"""
        self.eval1_rateofchange_mean = np.mean(self.eval1_rateofchange, axis = 0)
        self.eval1_rateofchange_uncert = np.std(self.eval1_rateofchange, axis = 0)/np.sqrt(self.ensemble)


        self.eval2_rateofchange_mean = np.mean(self.eval2_rateofchange, axis = 0)
        self.eval2_rateofchange_uncert = np.std(self.eval2_rateofchange, axis = 0)/np.sqrt(self.ensemble)

        self.eval3_rateofchange_mean = np.mean(self.eval3_rateofchange, axis = 0)
        self.eval3_rateofchange_uncert = np.std(self.eval3_rateofchange, axis = 0)/np.sqrt(self.ensemble)

    def compute_eval_deriv(self):
        """Computes the time derivative of the top 3 eigenvalues"""
        self.eval1_deriv_mean = np.mean(self.eval1_deriv, axis = 0)
        self.eval1_deriv_uncert = np.std(self.eval1_deriv, axis = 0)/np.sqrt(self.ensemble)

        self.eval2_deriv_mean = np.mean(self.eval2_deriv, axis = 0)
        self.eval2_deriv_uncert = np.std(self.eval2_deriv, axis = 0)/np.sqrt(self.ensemble)

        self.eval3_deriv_mean = np.mean(self.eval3_deriv, axis = 0)
        self.eval3_deriv_uncert = np.std(self.eval3_deriv, axis = 0)/np.sqrt(self.ensemble)

    def compute_alignments(self):
        
        """NTK alignment to training data"""
        self.alignment_uncert = np.std(self.NTK_alignment, axis = 0)/np.sqrt(self.ensemble)
        self.alignment_mean = np.mean(self.NTK_alignment, axis = 0)
        
        """Calculating ensemble values of eigenvector alignment
        Recall evec_alignment has shape [time, ensemble, evec]
        """
        self.evec_alignment_uncert = np.std(self.evec_alignment, axis = 1)/np.sqrt(self.ensemble)
        self.evec_alignment_mean = np.mean(self.evec_alignment, axis = 1)          

    def compute_top5Evals(self):
        """Computes the mean values of the top 5 largest eigenvalues"""
        self.eval_5_uncert = np.std(self.evalues,axis = 1)/np.sqrt(self.ensemble)
        self.eval_5_array = np.mean(self.evalues, axis =1)


    def compute_data(self):
        self.compute_eigvals_zero()
        self.compute_NTK_pts()
        self.compute_phi_sq()
        self.compute_preactivation_rate()
        self.compute_chi()
        self.compute_alignments()
        self.compute_abs_rel_ROC()
        self.compute_top5Evals()
        self.compute_eval_deriv()

        

    def save_data(self):
        params ={
            'InputSize': [self.input],
            'OutputSize': [self.output],
            'HiddenLayerWidth': [self.width],
            'HiddenLayerDepth': [self.depth],
            'LearningRate': [self.lr],
            'Epochs': [self.epochs],
            'STD': [self.std],
            'EnsembleNum': [self.ensemble],
            'Performances': [self.performances],
            'Bootstraps': [self.bootstraps],
            'AlignmentInterval': [self.alignmentint],
        }
        df = pd.DataFrame(params)
        df.to_csv(fr'C:\Users\Logan\Downloads\SummerWork\{self.Filename}\Params', index = False)
    
    def export_onnx(self):
        """Exports the model to an onnx file"""

        inst_input = torch.randn((1,1)).double() 
        """Since torch.export will run a tracer through the model, it does not actually matter what the input data is
        To make it as simple as possible, just using a 1X1 torch tensor
        """
        onnx_program = torch.onnx.export(self.model,inst_input)
        onnx_program.save(fr'C:\Users\Logan\Downloads\SummerWork\{self.Filename}\MLP_model.onnx')
        print(f"The model has been exported to an onnx file")


    
    def make_plots_regions(self):
        """Chi plots"""
        # for (t_start, t_end) in self.regions:
        #     mask = self.get_mask(self.train_time_rate, t_start, t_end)
        #     t = self.train_time_rate[mask]
        #     fig, ax = plt.subplots(figsize=(10, 6)) 
        #     for k in range(len(self.chi_array)): 
        #         mean = self.chi_array[k+1] #need k+1 since it starts at 1
        #         #std = ensemble_uncertainty[k]
                
                
                
        #         ###########################################################################################
        #         """Need to double check below is actually k+2"""

        #         ax.plot(t, mean[mask], label=f'Chi for Layer {k+2}') #Convention is to use layer 0 as input, so need to shift everything up by 1
        #         ###########################################################################################

        #         #ax.fill_between(epochs_axis, mean - std, mean + std, alpha=0.3)

        #     #Adding important regions to plot
        #     for j in range(len(self.NTK_points)):
        #         if t_start <= self.NTK_points[j] <= t_end:
        #             #ax.axvspan(self.NTK_points[j] - self.NTK_point_uncert[j], self.NTK_points[j] + self.NTK_point_uncert[j], color='purple', alpha=0.3)
        #             ax.axvline(self.NTK_points[j], color = 'purple', alpha = 0.3)
        #             #Axvspan expects a single point, cannot use an array, hence need the for loop    
        #     ax.set_xlabel('Training Time')
        #     ax.set_ylabel('Chi Value')
        #     ax.set_title('Chi vs Training Time')
        #     ax.legend()
        #     if self.SaveFig:
        #         plt.savefig(fr'C:\Users\Logan\Downloads\SummerWork\{self.Filename}\Chi_{t_start}_{t_end}')
        #     #plt.show()
        #     plt.close()
        


        """Absolute relative rate of change"""
        for (t_start, t_end) in self.regions:
            mask = self.get_mask(self.train_time_alignment, t_start, t_end)
            t = self.train_time_alignment[mask]
            plt.figure(figsize=(10,6))
            plt.plot(t, self.eval1_rateofchange_mean[mask], label = "Eigenvalue 1 rate of change")
            plt.plot(t, self.eval2_rateofchange_mean[mask], label = "Eigenvalue 2 rate of change")
            plt.plot(t, self.eval3_rateofchange_mean[mask], label = "Eigenvalue 3 rate of change")
            plt.fill_between(t, (self.eval1_rateofchange_mean+ self.eval1_rateofchange_uncert)[mask], 
                            (self.eval1_rateofchange_mean -self.eval1_rateofchange_uncert)[mask], alpha = 0.3)
            plt.fill_between(t, (self.eval2_rateofchange_mean+ self.eval2_rateofchange_uncert)[mask], 
                            (self.eval2_rateofchange_mean -self.eval2_rateofchange_uncert)[mask], alpha = 0.3)
            plt.fill_between(t, (self.eval3_rateofchange_mean+ self.eval3_rateofchange_uncert)[mask], 
                            (self.eval3_rateofchange_mean -self.eval3_rateofchange_uncert)[mask], alpha = 0.3)
            plt.legend()
            plt.ylabel(f"Absolute relative rate of change of eigenvalue")
            plt.xlabel(f"Training time")
            plt.title(f"Absolute relative rate of change of eigenvalues 1, 2 and 3 as they vary with training time")
            if self.SaveFig:
                plt.savefig(fr'C:\Users\Logan\Downloads\SummerWork\{self.Filename}\RelAbsChange_{t_start}_{t_end}')
            #plt.show()
            plt.close()

        """Eigenvalue derivatives"""
        # for (t_start, t_end) in self.regions:
        #     mask = self.get_mask(self.train_time_alignment, t_start, t_end)
        #     t = self.train_time_alignment[mask]
        #     plt.figure(figsize=(10,6))
        #     plt.plot(t, self.eval1_deriv_mean[mask], label = "Eigenvalue 1 rate of change")
        #     plt.plot(t, self.eval2_deriv_mean[mask], label = "Eigenvalue 2 rate of change")
        #     plt.plot(t, self.eval3_deriv_mean[mask], label = "Eigenvalue 3 rate of change")
        #     plt.fill_between(t, (self.eval1_deriv_mean+ self.eval1_deriv_uncert)[mask], 
        #                     (self.eval1_deriv_mean - self.eval1_deriv_uncert)[mask], alpha = 0.3)
        #     plt.fill_between(t, (self.eval2_deriv_mean+ self.eval2_deriv_uncert)[mask], 
        #                     (self.eval2_deriv_mean - self.eval2_deriv_uncert)[mask], alpha = 0.3)
        #     plt.fill_between(t, (self.eval3_deriv_mean+ self.eval3_deriv_uncert)[mask], 
        #                     (self.eval3_deriv_mean - self.eval3_deriv_uncert)[mask], alpha = 0.3)
        #     plt.legend()
        #     plt.ylabel(f"Derivative of the eigenvalues")
        #     plt.xlabel(f"Training time")
        #     plt.title(f"Time derivative of eigenvalues 1, 2 and 3 as they vary with training time")
        #     if self.SaveFig:
        #         plt.savefig(fr'C:\Users\Logan\Downloads\SummerWork\{self.Filename}\EvalDeriv_{t_start}_{t_end}')

        #     #plt.show()
        #     plt.close()

        """Pre-activation rates"""
        for (t_start, t_end) in self.regions:
            mask = self.get_mask(self.train_time_rate, t_start, t_end)
            t = self.train_time_rate[mask]
            fig, ax = plt.subplots(figsize=(10, 6)) 
            for k in range(len(self.ensemble_means)): 
                mean = self.ensemble_means[k]
                #std = ensemble_uncertainty[k]
                ax.plot(t, mean[mask], label=f'Layer {k+1}') #Convention is to use layer 0 as input, so need to shift everything up by 1
                #ax.fill_between(epochs_axis, mean - std, mean + std, alpha=0.3)

            #Adding important regions to plot
            for j in range(len(self.NTK_points)):
                if t_start <= self.NTK_points[j] <= t_end:
                    #ax.axvspan(self.NTK_points[j] - self.NTK_point_uncert[j], self.NTK_points[j] + self.NTK_point_uncert[j], color='purple', alpha=0.3)
                    ax.axvline(self.NTK_points[j], color = 'purple', alpha = 0.3)
                    #Axvspan expects a single point, cannot use an array, hence need the for loop    
            ax.set_xlabel('Training Time')
            ax.set_ylabel('Pre-Activation Derivative')
            ax.set_title('Finite Difference of Layer Pre-Activations')
            ax.legend()
            if self.SaveFig:
                plt.savefig(fr'C:\Users\Logan\Downloads\SummerWork\{self.Filename}\Activity_{t_start}_{t_end}')
            #plt.show()
            plt.close()


        """Pre-activation excluding outer layer"""
        for (t_start, t_end) in self.regions:
            mask = self.get_mask(self.train_time_rate, t_start, t_end)
            t = self.train_time_rate[mask]
            fig, ax = plt.subplots(figsize=(10, 6)) 
            for k in range(len(self.ensemble_means)-1): 
                mean = self.ensemble_means[k]
                #std = ensemble_uncertainty[k]
                ax.plot(t, mean[mask], label=f'Layer {k+1}') #Convention is to use layer 0 as input, so need to shift everything up by 1
                #ax.fill_between(epochs_axis, mean - std, mean + std, alpha=0.3)

            #Adding important regions to plot
            for j in range(len(self.NTK_points)):
                if t_start <= self.NTK_points[j] <= t_end:
                    #ax.axvspan(self.NTK_points[j] - self.NTK_point_uncert[j], self.NTK_points[j] + self.NTK_point_uncert[j], color='purple', alpha=0.3)
                    ax.axvline(self.NTK_points[j], color = 'purple', alpha = 0.3)
                    #Axvspan expects a single point, cannot use an array, hence need the for loop    
            ax.set_xlabel('Training Time')
            ax.set_ylabel('Pre-Activation Derivative')
            ax.set_title('Finite Difference of Layer Pre-Activations')
            ax.legend()
            if self.SaveFig:
                plt.savefig(fr'C:\Users\Logan\Downloads\SummerWork\{self.Filename}\Activity_NoOuter_{t_start}_{t_end}')
            #plt.show()
            plt.close()


        """Phi squared values"""
        for (t_start, t_end) in self.regions:
            mask = self.get_mask(self.train_time_total, t_start, t_end)
            t = self.train_time_total[mask]
            fig, ax = plt.subplots(figsize=(10, 6)) 
            for k in range(len(self.phi_sq)): 
                mean = self.phi_sq[k]
                #std = ensemble_uncertainty[k]
                ax.plot(t, mean[mask], label=f'Layer {k+1}') #Convention is to use layer 0 as input, so need to shift everything up by 1
                #ax.fill_between(epochs_axis, mean - std, mean + std, alpha=0.3)

            #Adding important regions to plot
            for j in range(len(self.NTK_points)):
                if t_start <= self.NTK_points[j] <= t_end:
                    #ax.axvspan(self.NTK_points[j] - self.NTK_point_uncert[j], self.NTK_points[j] + self.NTK_point_uncert[j], color='purple', alpha=0.3)
                    ax.axvline(self.NTK_points[j], color = 'purple', alpha = 0.3)
                    #Axvspan expects a single point, cannot use an array, hence need the for loop    
            ax.set_xlabel('Training Time')
            ax.set_ylabel('Phi Squared')
            ax.set_title('Phi Squared vs Training Time')
            ax.legend()
            if self.SaveFig:
                plt.savefig(fr'C:\Users\Logan\Downloads\SummerWork\{self.Filename}\PhiSq_{t_start}_{t_end}')
            #plt.show()
            plt.close()


        """NTK Alignment Value"""
        # for (t_start, t_end) in self.regions:
        #     mask = self.get_mask(self.train_time_alignment, t_start, t_end)
        #     t = self.train_time_alignment[mask]
        #     plt.figure(figsize=(8,6))
        #     plt.plot(t,self.alignment_mean[mask])
        #     plt.fill_between(t, (self.alignment_mean +self.alignment_uncert)[mask], (self.alignment_mean-self.alignment_uncert)[mask], alpha = 0.3)
        #     plt.xlabel(f'Training time')
        #     plt.ylabel(f'Alignment')
        #     plt.title(f'Alignment of the NTK vs Training Time')
        #     if self.SaveFig:
        #         plt.savefig(fr'C:\Users\Logan\Downloads\SummerWork\{self.Filename}\NTKAlignment_{t_start}_{t_end}')
        #     #plt.show()
        #     plt.close()

        """Eigenvector Alignment"""
        # for (t_start, t_end) in self.regions:
        #     mask = self.get_mask(self.train_time_alignment, t_start, t_end)
        #     t = self.train_time_alignment[mask]
        #     plt.figure(figsize=(8,6))
        #     for k in range(self.evec_alignment_mean.shape[1]):
        #         plt.plot(t,self.evec_alignment_mean[:,k][mask], label = f"Eigenvector {k+1}")
        #         plt.fill_between(t, (self.evec_alignment_mean[:,k]+self.evec_alignment_uncert[:,k])[mask],
        #                             (self.evec_alignment_mean[:,k]-self.evec_alignment_uncert[:,k])[mask], alpha = 0.3 )
        #     plt.xlabel(f"Training time")
        #     plt.ylabel(f"Normalised eigenvector alignment value")
        #     plt.title(f"5 Largest Eigenvector Alignment Values vs Training Time")
        #     plt.legend()
        #     if self.SaveFig:
        #         plt.savefig(fr'C:\Users\Logan\Downloads\SummerWork\{self.Filename}\EigenvectorAlignment_{t_start}_{t_end}')
        #     #plt.show()
        #     plt.close()


        """Top selected eigenvalues"""

        for (t_start, t_end) in self.regions:
            mask = self.get_mask(self.train_time_alignment, t_start, t_end)
            t = self.train_time_alignment[mask]
            plt.figure(figsize=(8,6))
            for k in range(self.evec_alignment_mean.shape[1]):
                plt.plot(t,self.eval_5_array[:,k][mask], label = f"Eigenvalue {k+1}")
                plt.fill_between(t, (self.eval_5_array[:,k] + self.eval_5_uncert[:,k])[mask],(self.eval_5_array[:,k] - self.eval_5_uncert[:,k])[mask]
                                    , alpha = 0.3 )
            plt.xlabel(f"Training time")
            plt.ylabel(f" Eigenvalue")
            plt.title(f"5 Largest Eigenvalues vs Training Time")
            plt.legend()
            if self.SaveFig:
                plt.savefig(fr'C:\Users\Logan\Downloads\SummerWork\{self.Filename}\Top5Evals_{t_start}_{t_end}')
            #plt.show()
            plt.close()

        """Eigenvalues excluding largest"""
        for (t_start, t_end) in self.regions:
            mask = self.get_mask(self.train_time_alignment, t_start, t_end)
            t = self.train_time_alignment[mask]
            plt.figure(figsize=(8,6))
            for k in range(self.evec_alignment_mean.shape[1]-1):
                plt.plot(t,self.eval_5_array[:,k+1][mask], label = f"Eigenvalue {k+2}")
                plt.fill_between(t, (self.eval_5_array[:,k+1] + self.eval_5_uncert[:,k+1])[mask],(self.eval_5_array[:,k+1] - self.eval_5_uncert[:,k+1])[mask]
                                    , alpha = 0.3 )
            plt.xlabel(f"Training time")
            plt.ylabel(f" Eigenvalue")
            plt.title(f"4 Largest Eigenvalues vs Training Time (Excluding Eval 1)")
            plt.legend()
            if self.SaveFig:
                plt.savefig(fr'C:\Users\Logan\Downloads\SummerWork\{self.Filename}\Top5Evals_No1_{t_start}_{t_end}')
            #plt.show()
            plt.close()


        """Loss"""
        for (t_start, t_end) in self.regions:
            mask = self.get_mask(self.train_time_total, t_start, t_end)
            t = self.train_time_total[mask]
            ensemble_loss = np.mean(self.loss_array[mask], axis=1)
            ensemble_loss_uncert = np.std(self.loss_array[mask], axis = 1)/np.sqrt(self.ensemble)
            plt.figure(figsize=(10,6))
            plt.plot(t,ensemble_loss)
            plt.fill_between(t,ensemble_loss+ensemble_loss_uncert,ensemble_loss-ensemble_loss_uncert, alpha = 0.3)
            for j in range(len(self.NTK_points)):
                if t_start <= self.NTK_points[j] <= t_end:
                #plt.axvspan(self.NTK_points[j] - self.NTK_point_uncert[j], self.NTK_points[j] + self.NTK_point_uncert[j], color='purple', alpha=0.3)
                    plt.axvline(self.NTK_points[j], color = 'purple', alpha = 0.3)
            plt.xlabel(f'Training time')
            plt.ylabel(f"Average ensemble loss value")
            plt.title(f'Ensemble loss vs training time')
            if self.SaveFig:
                plt.savefig(fr'C:\Users\Logan\Downloads\SummerWork\{self.Filename}\Loss_{t_start}_{t_end}')
            #plt.show()
            plt.close()
        
        """Performances"""
         #Need to include training points here
        mean_ensemble_val = np.mean(self.performance_array,axis=1)
        ensemble_uncert = np.std(self.performance_array,axis = 1)/np.sqrt(self.ensemble)
        x_vals = self.x_test.squeeze().numpy()
        for k in range(self.performance_array.shape[0]):
            # epoch_value = (k+1)*self.step
            epoch_value = self.snapshot_array[k] *self.lr
            plt.figure(figsize=(8,6))
            plt.plot(x_vals, mean_ensemble_val[k,:], label = f'Predicted Values')
            plt.plot(x_vals,self.y_test.numpy(), label = f'True Values')
            plt.scatter(self.x_train,self.y_train,label = "Training points")
            plt.fill_between(x_vals, mean_ensemble_val[k,:]+ensemble_uncert[k,:],mean_ensemble_val[k,:]-ensemble_uncert[k,:],color = 'blue', alpha = 0.3)
            #plt.fill_between is finicky, needs inputs to be explicitly 1 dimensional, hence the X_test_sorted.squeeze().numpy()
            plt.xlabel(f'X')
            plt.ylabel(f'Y')
            plt.title(f'Performance of the model at training time {epoch_value}')
            plt.legend()
            if self.SaveFig:
                plt.savefig(fr'C:\Users\Logan\Downloads\SummerWork\{self.Filename}\Performance{k+1}')
            #plt.show()
            plt.close() 



    def make_plots(self):

        # """Chi plots"""
        # fig, ax = plt.subplots(figsize=(10, 6)) 
        # for k in range(len(self.chi_array)): 
        #     mean = self.chi_array[k+1]
        #     #std = ensemble_uncertainty[k]
        #     ax.plot(self.train_time_rate, mean, label=f'Chi Layer {k+1}') #Convention is to use layer 0 as input, so need to shift everything up by 1
        #     #ax.fill_between(epochs_axis, mean - std, mean + std, alpha=0.3)

        # #Adding important regions to plot
        # for j in range(len(self.NTK_points)):
        #     #ax.axvspan(self.NTK_points[j] - self.NTK_point_uncert[j], self.NTK_points[j] + self.NTK_point_uncert[j], color='purple', alpha=0.3)
        #     ax.axvline(self.NTK_points[j], color = 'purple', alpha = 0.3)
        #     #Axvspan expects a single point, cannot use an array, hence need the for loop    
        # ax.set_xlabel('Training Time')
        # ax.set_ylabel('Chi Value')
        # ax.set_title('Chi vs Training Time')
        # ax.legend()
        # plt.tight_layout()
        # if self.SaveFig:
        #     plt.savefig(fr'C:\Users\Logan\Downloads\SummerWork\{self.Filename}\Chi')
        # #plt.show()
        # plt.close()


        """Absolute relative rate of change"""
    
        plt.figure(figsize=(10,6))
        plt.plot(self.train_time_alignment, self.eval1_rateofchange_mean, label = "Eigenvalue 1 rate of change")
        plt.plot(self.train_time_alignment, self.eval2_rateofchange_mean, label = "Eigenvalue 2 rate of change")
        plt.plot(self.train_time_alignment, self.eval3_rateofchange_mean, label = "Eigenvalue 3 rate of change")
        plt.fill_between(self.train_time_alignment, self.eval1_rateofchange_mean+ self.eval1_rateofchange_uncert, 
                        self.eval1_rateofchange_mean -self.eval1_rateofchange_uncert, alpha = 0.3)
        plt.fill_between(self.train_time_alignment, self.eval2_rateofchange_mean+ self.eval2_rateofchange_uncert, 
                        self.eval2_rateofchange_mean -self.eval2_rateofchange_uncert, alpha = 0.3)
        plt.fill_between(self.train_time_alignment, self.eval3_rateofchange_mean+ self.eval3_rateofchange_uncert, 
                        self.eval3_rateofchange_mean -self.eval3_rateofchange_uncert, alpha = 0.3)
        plt.legend()
        plt.ylabel(f"Relative rate of change of eigenvalue")
        plt.xlabel(f"Training time")
        plt.title(f"Absolute rate of change of eigenvalues 1, 2 and 3 as they vary with training time")
        if self.SaveFig:
            plt.savefig(fr'C:\Users\Logan\Downloads\SummerWork\{self.Filename}\RelAbsChange')
        #plt.show()
        plt.close()


        """Eigenvalue derivatives"""
    
        # plt.figure(figsize=(10,6))
        # plt.plot(self.train_time_alignment, self.eval1_deriv_mean, label = "Eigenvalue 1 rate of change")
        # plt.plot(self.train_time_alignment, self.eval2_deriv_mean, label = "Eigenvalue 2 rate of change")
        # plt.plot(self.train_time_alignment, self.eval3_deriv_mean, label = "Eigenvalue 3 rate of change")
        # plt.fill_between(self.train_time_alignment, self.eval1_deriv_mean+ self.eval1_deriv_uncert, 
        #                 self.eval1_deriv_mean - self.eval1_deriv_uncert, alpha = 0.3)
        # plt.fill_between(self.train_time_alignment, self.eval2_deriv_mean+ self.eval2_deriv_uncert, 
        #                 self.eval2_deriv_mean - self.eval2_deriv_uncert, alpha = 0.3)
        # plt.fill_between(self.train_time_alignment, self.eval3_deriv_mean+ self.eval3_deriv_uncert, 
        #                 self.eval3_deriv_mean - self.eval3_deriv_uncert, alpha = 0.3)
        # plt.legend()
        # plt.ylabel(f"Derivative of the eigenvalues")
        # plt.xlabel(f"Training time")
        # plt.title(f"Time derivative of eigenvalues 1, 2 and 3 as they vary with training time")
        # if self.SaveFig:
        #     plt.savefig(fr'C:\Users\Logan\Downloads\SummerWork\{self.Filename}\EvalDeriv')
        # #plt.show()
        # plt.close()


        """Pre-activation rate"""
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
        #plt.show()
        plt.close()

        """Excluding the final layer"""
        fig, ax = plt.subplots(figsize=(10, 6)) 
        for k in range(len(self.ensemble_means)-1): 
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
            plt.savefig(fr'C:\Users\Logan\Downloads\SummerWork\{self.Filename}\Activity_NoOuter')
        #plt.show()
        plt.close()


        """Phi_squared"""
        fig, ax = plt.subplots(figsize=(10, 6)) 
        for k in range(len(self.ensemble_means)): 
            mean = self.phi_sq[k]
            #std = ensemble_uncertainty[k]
            ax.plot(self.train_time_total, mean, label=f'Layer {k+1}') #Convention is to use layer 0 as input, so need to shift everything up by 1
            #ax.fill_between(epochs_axis, mean - std, mean + std, alpha=0.3)

        #Adding important regions to plot
        for j in range(len(self.NTK_points)):
            #ax.axvspan(self.NTK_points[j] - self.NTK_point_uncert[j], self.NTK_points[j] + self.NTK_point_uncert[j], color='purple', alpha=0.3)
            ax.axvline(self.NTK_points[j], color = 'purple', alpha = 0.3)
            #Axvspan expects a single point, cannot use an array, hence need the for loop    
        ax.set_xlabel('Training Time')
        ax.set_ylabel('Phi Squared')
        ax.set_title('Phi Squared vs training time')
        ax.legend()
        plt.tight_layout()
        if self.SaveFig:
            plt.savefig(fr'C:\Users\Logan\Downloads\SummerWork\{self.Filename}\PhiSq')
        #plt.show()
        plt.close()


        # """NTK alignment value"""
        # plt.figure(figsize=(8,6))
        # plt.plot(self.train_time_alignment,self.alignment_mean)
        # plt.fill_between(self.train_time_alignment, self.alignment_mean +self.alignment_uncert, self.alignment_mean-self.alignment_uncert, alpha = 0.3)
        # plt.xlabel(f'Training time')
        # plt.ylabel(f'Alignment')
        # plt.title(f'Alignment of the NTK vs Training Time')
        # if self.SaveFig:
        #     plt.savefig(fr'C:\Users\Logan\Downloads\SummerWork\{self.Filename}\NTKAlignment')
        # #plt.show()
        # plt.close()

        # """Plotting the eigenvector alignment value"""
        # plt.figure(figsize=(8,6))
        # for k in range(self.evec_alignment_mean.shape[1]):
        #     plt.plot(self.train_time_alignment,self.evec_alignment_mean[:,k], label = f"Eigenvector {k+1}")
        #     plt.fill_between(self.train_time_alignment, self.evec_alignment_mean[:,k]+self.evec_alignment_uncert[:,k],self.evec_alignment_mean[:,k]-self.evec_alignment_uncert[:,k], alpha = 0.3 )
        # plt.xlabel(f"Training time")
        # plt.ylabel(f"Normalised eigenvector alignment value")
        # plt.title(f"5 Largest Eigenvector Alignment Values vs Training Time")
        # plt.legend()
        # if self.SaveFig:
        #     plt.savefig(fr'C:\Users\Logan\Downloads\SummerWork\{self.Filename}\EigenvectorAlignment')
        # #plt.show()
        # plt.close()


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
        #plt.show()
        plt.close()

        """Plotting eigenvalues 2-5"""
        plt.figure(figsize=(8,6))
        for k in range(self.evec_alignment_mean.shape[1]-1):
            plt.plot(self.train_time_alignment,self.eval_5_array[:,k+1], label = f"Eigenvalue {k+2}")
            plt.fill_between(self.train_time_alignment, (self.eval_5_array[:,k+1] + self.eval_5_uncert[:,k+1]),(self.eval_5_array[:,k+1] - self.eval_5_uncert[:,k+1])
                                , alpha = 0.3 )
        plt.xlabel(f"Training time")
        plt.ylabel(f" Eigenvalue")
        plt.title(f"4 Largest Eigenvalues vs Training Time (Excluding Eval 1)")
        plt.legend()
        if self.SaveFig:
                plt.savefig(fr'C:\Users\Logan\Downloads\SummerWork\{self.Filename}\Top5Evals_No1')
        #plt.show()
        plt.close()



        """Plotting Losses"""
        plt.figure(figsize=(10,6))
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
        #plt.show()
        plt.close()

        """Plotting Performance"""
        #Need to include training points here
        mean_ensemble_val = np.mean(self.performance_array,axis=1)
        ensemble_uncert = np.std(self.performance_array,axis = 1)/np.sqrt(self.ensemble)
        x_vals = self.x_test.squeeze().numpy()
        for k in range(self.performance_array.shape[0]):
            # epoch_value = (k+1)*self.step
            epoch_value = self.snapshot_array[k] *self.lr
            plt.figure(figsize=(8,6))
            plt.plot(x_vals, mean_ensemble_val[k,:], label = f'Predicted Values')
            plt.plot(x_vals,self.y_test.numpy(), label = f'True Values')
            plt.scatter(self.x_train,self.y_train,label = "Training points")
            plt.fill_between(x_vals, mean_ensemble_val[k,:]+ensemble_uncert[k,:],mean_ensemble_val[k,:]-ensemble_uncert[k,:],color = 'blue', alpha = 0.3)
            #plt.fill_between is finicky, needs inputs to be explicitly 1 dimensional, hence the X_test_sorted.squeeze().numpy()
            plt.xlabel(f'X')
            plt.ylabel(f'Y')
            plt.title(f'Performance of the model at training time {epoch_value}')
            plt.legend()
            if self.SaveFig:
                plt.savefig(fr'C:\Users\Logan\Downloads\SummerWork\{self.Filename}\Performance{k+1}')
            #plt.show()
            plt.close() 



    def get_mask(self, time_array, time_start,time_end):
        return (time_array >= time_start) & (time_array <= time_end)

    def run_no_plot(self):
        self.save_data()
        for j in range(self.ensemble):
            self.train_model(j)
            print(f"Completed replica number {j}")
        self.compute_data()

    def run_plot(self):
        self.save_data()
        for j in range(self.ensemble):
            self.train_model(j)
            print(f"Completed replica number {j}")
        self.compute_data()
        if self.regions is not None:
            self.make_plots_regions()
        else:
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
    parser.add_argument('--STD',type=float, help='Determines the standard deviation (width) of the normal distribution for the hidden layers weights', default = 0.1)
    parser.add_argument('--EnsembleNum', type = int, help= ' Determines the number of models to create for the purposes of ensemble averages', default= 10)
    parser.add_argument('--Performances', type = int, help='Determines the number of printouts of model performance desired', default=4)
    parser.add_argument('--Bootstraps', type= int, help='Determines the number of bootstraps to calculate for error propagation', default=100)
    parser.add_argument('--AlignmentInterval', type = int, help='Determines how frequently to calculate the NTK alignment', default=100)
    parser.add_argument('--SaveFig', action='store_true', help='If set, saves figures to args.Filename')
    parser.add_argument('--Filename', type = str, help='Determines the file to save data to', default= 'Unsorted')
    parser.add_argument('--Linear', type = bool, help = 'Determines whether to run as a linear model', default = False)
    parser.add_argument('--EvalAmount', type = int, help = 'Determines the number of eigenvalues/eigenvectors of interest', default = 5)
    parser.add_argument('--onnx_filename', type = str, help='Determines the filename for the onnx data', default= 'onnx_unsorted')

    args = parser.parse_args()

    # regions = [ (0,1),
    #            (1,10),
    #            (10,140),
    #            (140,250),
    #            (250,310),
    #            (310,400),
    #            (400,600),
    #            (600,800)

    # ]
    step = 10
    regions = [(t, t + step) for t in np.arange(300, 400, step)]

    # performances_array = np.arange(30000,50000,1000)
    performances_array = np.arange(60000,80000,2000)



    trial = Trial(args.InputSize, args.OutputSize,args.HiddenLayerWidth,args.HiddenLayerDepth,args.lr,args.Epochs,args.STD,args.EnsembleNum,args.Performances,
                  args.Bootstraps,args.AlignmentInterval,X_train_sorted,Y_train_sorted,X_eval,Y_eval,args.Filename,args.SaveFig,regions,args.Linear, args.EvalAmount,
                  performances_array)
    trial.run_plot()
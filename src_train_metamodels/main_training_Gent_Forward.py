"""
Objective
    Training/Validation/Testing of MLP Neural Network for solving the inverse
    problem with a least squares ML approach. For the FE data produced with
    Gent material model.
    
Author
    Patrick Giolando
    BME PhD at The University of Texas at Austin
    pgioland@utexas.edu


Parameters
----------
EPOCH:
    EPOCH
Ntrain:
    Number of training samples
Npred:
    Number of tesing samples
Nval:
    Number of validation samples


"""

# =============================================================================
# Import Libraries and Functions
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
import time
import scipy.optimize as optimize
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import tensorflow as tf
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from scipy.signal import argrelmax
from scipy.ndimage.filters import gaussian_filter1d
import pickle
from keras import layers, initializers
import sys
import os


from tensorflow.keras.layers import LeakyReLU

current_dir = os.getcwd()
func_files = current_dir[:-9] + 'Functions'
nn_files = current_dir[:] + '\\trained_metamodels'
syn_files = current_dir[:] + '\\data\\SynthData'
sys.path.append(func_files)

from subfunctions_training_Gent_Forward import *


    
# =============================================================================
# Load Finite Element Data
# =============================================================================
Model_Output = np.loadtxt(syn_files + "\\Model_Output_ExtraSmallJm")
Model_Input = np.loadtxt(syn_files + "\\Model_Input_ExtraSmallJm")

Model_Output = np.loadtxt(syn_files + "\\Model_Output_ValPred")
Model_Input = np.loadtxt(syn_files + "\\Model_Input_ValPred")
Ndat_VP = len(Model_Input_ValPred[:,0])


# =============================================================================
# User Defined Parameters
# =============================================================================
EPOCHS = 30000
Ntrain = 10000
Npred =  1250
Nval = 1250


# =============================================================================
# Resample FE Data
# =============================================================================
FEA_fit_params, FEA_fit_params_VP, Model_Output_Resampled, Model_Output_Resampled_ValPred = FindParamFits(Model_Input,Model_Output,Model_Input_ValPred,Model_Output_ValPred)
        
        
# =============================================================================
# Scale Features for Taining, Validation, and Prediction
# =============================================================================
xtrain, ytrain, xtest, ytest, xpred, ypred = Scale_Input_Output(FEA_fit_params, FEA_fit_params_VP, Model_Input, Model_Input_ValPred, Model_Output_Resampled, Model_Output_Resampled_ValPred)        
        
        
        
# =============================================================================
# Neural Network Training/Validation/Testing
# =============================================================================
def ModelRun():
    
    n_inputs, n_outputs = xtrain.shape[1], ytrain.shape[1]
    
    # Sample Weights
    sample_weight = np.copy(xtrain[:,2])
    sample_weight[sample_weight <= 0.1] =100
    sample_weight[sample_weight > 0.1] = 0.1
    
    # Neural Network Architecture
    model = Sequential()
    initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
    bias_initializer = initializers.Constant(-0.1) #tf.keras.initializers.RandomNormal(mean=1., stddev=1.)
    
    model.add(Dense(int(n_inputs*1),kernel_initializer=initializer,bias_initializer=bias_initializer, activation=LeakyReLU(alpha=0.3)))
    model.add(Dense(int(n_inputs*1),kernel_initializer=initializer,bias_initializer=bias_initializer, activation=LeakyReLU(alpha=0.3)))
    model.add(Dense(int(n_outputs*1),kernel_initializer=initializer,bias_initializer=bias_initializer, activation=LeakyReLU(alpha=0.3)))
    model.add(Dense(int(n_outputs*1),kernel_initializer=initializer,bias_initializer=bias_initializer, activation=LeakyReLU(alpha=0.3)))
    model.add(Dense(int(n_outputs*1),kernel_initializer=initializer,bias_initializer=bias_initializer, activation=LeakyReLU(alpha=0.3)))
    model.add(Dense(n_outputs,kernel_initializer=initializer,bias_initializer=bias_initializer,activation=LeakyReLU(alpha=0.3)))# activation='linear') )
    
    model.compile(loss='mse', optimizer = 'adam')
    hist = model.fit(xtrain, ytrain, epochs=EPOCHS, sample_weight=sample_weight, verbose=0,validation_data=(xtest, ytest),batch_size=25)
        
    Nd = 100
    
    # Error in Predicting Validation Data
    Error = np.zeros(3)
    R2 = np.zeros(3)
    error_test = np.zeros((len(xtest[:,0]),1))#n_outputs))
    Yhat_test = np.zeros((len(xtest[:,0]),n_outputs))
    Y_test = np.zeros((len(xtest[:,0]),n_outputs))
    for i in range(0,len(xtest[:,0])):
        # Neural Network Prediction
        yhat_test = model.predict([xtest[i,:].tolist()])
        
        # Rescale Output
        Ppred = np.zeros(Nd);  Pact = np.zeros(Nd); Err = np.zeros(Nd)
        for k in range(0,Nd):
            minP = min((Model_Output_Resampled[k,:])); maxP = max((Model_Output_Resampled[k,:]))
            Ppred[k] = ((yhat_test[0][k]*(maxP-minP) + minP))
            Pact[k] = ((ytest[i,k]*(maxP-minP) + minP))
                        
            Err[k] = (abs(Ppred[k]-Pact[k])/abs(Pact[k])*100)
                
        # Store Model Prediction and Error
        Yhat_test[i,:] = Ppred
        Y_test[i,:] = Pact
        error_test[i] = np.average(Err)    #np.average(abs(yhat_test-ytest[i])/yhat_test*100)
        
    # Error in Predicting Testing Data
    error_pred = np.zeros((len(xpred[:,0]),1))#n_outputs))
    Y_pred = np.zeros((len(xpred[:,0]),n_outputs))
    Yhat_pred = np.zeros((len(xpred[:,0]),n_outputs))
    for i in range(0,len(xpred[:,0])):
        # Neural Network Prediction
        yhat_pred = model.predict([xpred[i,:].tolist()])
        
        # Rescale Output
        Ppred = np.zeros(Nd);  Pact = np.zeros(Nd);  Err = np.zeros(Nd)
        for k in range(0,Nd):
            minP = min((Model_Output_Resampled[k,:])); maxP = max((Model_Output_Resampled[k,:]))
            Ppred[k] = ((yhat_pred[0][k]*(maxP-minP) + minP))
            Pact[k] = ((ypred[i,k]*(maxP-minP) + minP))
            Err[k] = (abs(Ppred[k]-Pact[k])/abs(Pact[k])*100)
               
        # Store Model Prediction and Error
        Yhat_pred[i,:] = Ppred
        Y_pred[i,:] = Pact
        error_pred[i] = np.average(Err)    #np.average(abs(yhat_test-ytest[i])/yhat_test*100)
        
    # Error in Predicting Training Data
    error_train = np.zeros((len(xtrain[:,0]),n_outputs))
    Y_train = np.zeros((len(xtrain[:,0]),n_outputs))
    Yhat_train = np.zeros((len(xtrain[:,0]),n_outputs))
    for i in range(0,len(xtrain[:,0])):
        # Neural Network Prediction
        yhat_train = model.predict([xtrain[i,:].tolist()])
        
        # Rescale Output
        Ppred = np.zeros(Nd);  Pact = np.zeros(Nd);  Err = np.zeros(Nd)
        for k in range(0,Nd):
            minP = min((Model_Output_Resampled[k,:])); maxP = max((Model_Output_Resampled[k,:]))
            Ppred[k] = ((yhat_train[0][k]*(maxP-minP) + minP))
            Pact[k] = ((ytrain[i,k]*(maxP-minP) + minP))
            Err[k] = (abs(Ppred[k]-Pact[k])/abs(Pact[k])*100)
                
        # Store Model Prediction and Error
        Yhat_train[i,:] = Ppred
        Y_train[i,:] = Pact
        error_train[i] = np.average(Err) 
        
    # Store Error and R^2 
    Error[0] = np.average(error_train[:])
    Error[1] = np.average(error_test[:])
    Error[2] = np.average(error_pred[:])
    
    R2[0] = r2_score(ytrain, Yhat_train )
    R2[0] = r2_score(ytest, Yhat_test )
    R2[0] = r2_score(ypred, Yhat_pred )

    return error_test, Yhat_test, hist, R2,Error, Exp_Brain_Params, Exp_Brain_Fix_Params, model


        
        


# =============================================================================
# Run NN Model 
# =============================================================================
start = time.process_time()
error, Yhat, hist, R2,Error, Exp_Brain_Params, Exp_Brain_Fix_Params, model = ModelRun()

end = time.process_time()
print('Walltime: ',(end-start),'(s)' )



        



























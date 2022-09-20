"""
Objective
    Training/Validation/Testing of MLP Neural Network for solving the inverse
    problem with a direct inverse approach. For the FE data produced with
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
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LeakyReLU
import time
import scipy.optimize as optimize
from sklearn.metrics import r2_score
import matplotlib.cm as cm
import tensorflow as tf
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
from keras import layers, initializers

from sklearn.metrics import r2_score
import pickle
import sys
import os

current_dir = os.getcwd()
func_files = current_dir[:-9] + 'Functions'
sys.path.append(func_files)

from NN_Gent_Inverse_ABfits_Functions import *


    
    
# =============================================================================
# Load Finite Element Data
# =============================================================================
Model_Input = np.loadtxt('C:/Users/pgiol/Documents/DrRausch/NeuralNetworks/SynthData/FinalData/GentModel/Model_Input_ExtraSmallJm')
Model_Output = np.loadtxt('C:/Users/pgiol/Documents/DrRausch/NeuralNetworks/SynthData/FinalData/GentModel/Model_Output_ExtraSmallJm')

Model_Input_ValPred = np.loadtxt('C:/Users/pgiol/Documents/DrRausch/NeuralNetworks/SynthData/FinalData/GentModel/Model_Input_ValPred')
Model_Output_ValPred = np.loadtxt('C:/Users/pgiol/Documents/DrRausch/NeuralNetworks/SynthData/FinalData/GentModel/Model_Output_ValPred')
Ndat_VP = len(Model_Input_ValPred[:,0])



# =============================================================================
# User Defined Parameters
# =============================================================================
EPOCHS = 30000
Ntrain = 10000
Npred =  1250
Nval = 1250
            
    

# =============================================================================
# Identify parameter fits for FE Data
# =============================================================================
FEA_fit_params, FEA_fit_params_VP,Model_Output_Resampled, Model_Output_Resampled_ValPred, u_Hertz, u_ModHertz, R2_H, R2_MH =  FindParamFits(Model_Input,Model_Output,Model_Input_ValPred,Model_Output_ValPred)



# =============================================================================
# Scale Features for Taining, Validation, and Prediction
# =============================================================================
xtrain, ytrain, xtest, ytest, xpred, ypred = Scale_Input_Output(FEA_fit_params, FEA_fit_params_VP, Model_Input, Model_Input_ValPred, Model_Output, Model_Output_ValPred)




# =============================================================================
# Neural Network Training/Validation/Testing
# =============================================================================
def ModelRun(xtrain, ytrain, xtest, ytest, xpred, ypred ):

    # Neural Network Architecture
    n_inputs, n_outputs = xtrain.shape[1], ytrain.shape[1]

    model = Sequential()
    initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
    bias_initializer = initializers.RandomNormal(mean=0, stddev=1.) 
    
    model.add(Dense(int(n_inputs*10),kernel_initializer=initializer,bias_initializer=bias_initializer, activation=LeakyReLU(alpha=0.3)))
    model.add(Dense(int(n_inputs*10),kernel_initializer=initializer,bias_initializer=bias_initializer, activation=LeakyReLU(alpha=0.3)))
    model.add(Dense(int(n_inputs*10),kernel_initializer=initializer,bias_initializer=bias_initializer, activation=LeakyReLU(alpha=0.3)))
    model.add(Dense(int(n_inputs*10),kernel_initializer=initializer,bias_initializer=bias_initializer, activation=LeakyReLU(alpha=0.3)))
    model.add(Dense(int(n_inputs*10),kernel_initializer=initializer,bias_initializer=bias_initializer, activation=LeakyReLU(alpha=0.3)))
    model.add(Dense(n_outputs,kernel_initializer=initializer,bias_initializer=bias_initializer,activation= 'linear' ))
    
    model.compile(loss='mae', optimizer = 'adam', metrics=['mean_absolute_error'])
    
    hist = model.fit(xtrain, ytrain, epochs=EPOCHS,verbose=0,validation_data=(xtest, ytest),batch_size=32)
    
    
    
    
    # Error in Predicting Validation Data
    error_test_scaled = np.zeros((len(xtest[:,0]),n_outputs))
    error_test = np.zeros((len(xtest[:,0]),n_outputs))
    Yhat_test = np.zeros((len(xtest[:,0]),n_outputs))
    
    minU = 2; maxU = 6
    minJ = min(Model_Input[:,3]); maxJ = max(Model_Input[:,3])
    for i in range(0,len(xtest[:,0])):
        yhat_test = model.predict([xtest[i,:].tolist()])
        Yhat_test[i] = yhat_test
        
        u_pred = 10**( ((yhat_test[0][0]/1 + 0.0 ) * (maxU - minU)*1)  + minU )
        Jm_pred = ((yhat_test[0][1]/1 + 0.0 ) * (maxJ - minJ)*1)  + minJ
        u_act = 10**( ((ytest[i,0]/1 + 0.0 ) * (maxU - minU)*1)  + minU )
        Jm_act = ((ytest[i,1]/1 + 0.0 ) * (maxJ - minJ)*1)  + minJ
        error_test_scaled[i,0] = (abs(u_pred-u_act)/abs(u_act)*100)
        error_test_scaled[i,1] = (abs(Jm_pred-Jm_act)/abs(Jm_act)*100)
        
        error_test[i,0] = error_test_scaled[i,0]
        error_test[i,1] = error_test_scaled[i,1]
        
        
    # Error in Predicting Testing Data
    error_pred_scaled = np.zeros((len(xtrain[:,0]),n_outputs))
    error_pred = np.zeros((len(xtrain[:,0]),n_outputs))
    Yhat_pred = np.zeros((len(xtrain[:,0]),n_outputs))

    for i in range(0,len(xpred[:,0])):
        yhat_pred = model.predict([xpred[i,:].tolist()])
        Yhat_pred[i] = yhat_pred
        
        u_pred = 10**( ((yhat_pred[0][0]/1 + 0.0 ) * (maxU - minU)*1)  + minU )
        Jm_pred = ((yhat_pred[0][1]/1 + 0.0 ) * (maxJ - minJ)*1)  + minJ
        u_act = 10**( ((ypred[i,0]/1 + 0.0 ) * (maxU - minU)*1)  + minU )
        Jm_act = ((ypred[i,1]/1 + 0.0 ) * (maxJ - minJ)*1)  + minJ
        error_pred_scaled[i,0] = (abs(u_pred-u_act)/abs(u_act)*100)
        error_pred_scaled[i,1] = (abs(Jm_pred-Jm_act)/abs(Jm_act)*100)
        
        error_pred[i,0] = error_pred_scaled[i,0]
        error_pred[i,1] = error_pred_scaled[i,1]
        
        
    # Error in Predicting Training Data
    error_train_scaled = np.zeros((len(xtrain[:,0]),n_outputs))
    error_train = np.zeros((len(xtrain[:,0]),n_outputs))
    Yhat_train = np.zeros((len(xtrain[:,0]),n_outputs))

    for i in range(0,len(xtrain[:,0])):
        yhat_train = model.predict([xtrain[i,:].tolist()])
        Yhat_train[i] = yhat_train
        
        u_pred = 10**( ((yhat_train[0][0]/1 + 0.0 ) * (maxU - minU)*1)  + minU )
        Jm_pred = ((yhat_train[0][1]/1 + 0.0 ) * (maxJ - minJ)*1)  + minJ
        u_act = 10**( ((ytrain[i,0]/1 + 0.0 ) * (maxU - minU)*1)  + minU )
        Jm_act = ((ytrain[i,1]/1 + 0.0 ) * (maxJ - minJ)*1)  + minJ
        error_train_scaled[i,0] = (abs(u_pred-u_act)/abs(u_act)*100)
        error_train_scaled[i,1] = (abs(Jm_pred-Jm_act)/abs(Jm_act)*100)
        
        error_train[i,0] = error_train_scaled[i,0] 
        error_train[i,1] = error_train_scaled[i,1]
    
    
    
    # Store average error and R^2 between known and predicted values
    Error = np.zeros(4)
    R2 = np.zeros(4)
    R2[0] = r2_score(ytrain[:,0], Yhat_train[:,0] )
    R2[1] = r2_score(ytrain[:,1], Yhat_train[:,1] )
    R2[2] = r2_score(ytest[:,0], Yhat_test[:,0])
    R2[3] = r2_score(ytest[:,1], Yhat_test[:,1])
    Error[0] = np.average(error_train[:],axis=0)[0]
    Error[1] = np.average(error_train[:],axis=0)[1]
    Error[2] = np.average(error_test[:],axis=0)[0]
    Error[3] = np.average(error_test[:],axis=0)[1]
    
    
    return error_test, error_train, Yhat_test, Yhat_train, hist, R2, Error, model
    
    









# =============================================================================
# Run NN Model 
# =============================================================================
start = time.process_time()
error_test, error_train, Yhat_test, Yhat_train, hist, R2, Error, model = ModelRun(xtrain, ytrain, xtest, ytest, xpred, ypred )
end = time.process_time()
print('Walltime: ',(end-start),'(s)' )










def PlotS():
    plt.figure(1,figsize=(7,6))

    # plt.plot(ytest2,u_Hertz,'or',label='Hertzian Model, $R^2=$ %0.5f'%(R2_H), alpha = 0.25);plt.xlabel('Target Shear Modulus (Pa)'); plt.ylabel('Predicted Shear Modulus (Pa)')
    # plt.plot(ytest3,u_ModHertz,'og',label='Modified Hertzian Model, $R^2=$ %0.5f'%(R2_MH), alpha = 0.25);plt.xlabel('Target Shear Modulus (Pa)'); plt.ylabel('Predicted Shear Modulus (Pa)')
    
    plt.plot(ytest[:,1],Yhat_test[:,1],'ob',label='ML Model %i Samples, $R^2=$ %0.7f'%(Ntrain,R2[0]), alpha = 0.5);plt.xlabel('Target Shear Modulus (Pa)'); plt.ylabel('Predicted Shear Modulus (Pa)')
    plt.plot(ytest[:,0],Yhat_test[:,0],'ok',label='ML Model %i Samples, $R^2=$ %0.7f'%(Ntrain,R2[0]), alpha = 0.5);plt.xlabel('Target Shear Modulus (Pa)'); plt.ylabel('Predicted Shear Modulus (Pa)')

    plt.plot(np.linspace(0,1e6,100),np.linspace(0,1e6,100),'r--')
    plt.legend(loc='best')
    plt.xlim((0,1))
    plt.ylim((0,1))
    
    
    
PlotS()
    



























            
            
            
            
            
            
            
            
            
            
            
            
            
            
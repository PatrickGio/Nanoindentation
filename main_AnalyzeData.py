
# =============================================================================
# Import Libraries and Load Functions
# =============================================================================
import sys
import os
import numpy as np
import math as ma
import matplotlib.pyplot as plt
import keras
import pickle
import scipy.optimize as optimize
import time

current_dir = os.getcwd()
func_files = current_dir[:-9] + 'functions'
nn_files = current_dir[:] + '\\trained_metamodels'
exp_files = current_dir[:] + '\\data\\ExperimentalData'
syn_files = current_dir[:] + '\\data\\SynthData'
res_files = current_dir[:] + '\\results'

sys.path.append(current_dir)
sys.path.append(func_files)
sys.path.append(nn_files)
sys.path.append(exp_files)
sys.path.append(syn_files)

from subfunctions_Analyze import *



# =============================================================================
# User Defined Parameters
# =============================================================================
# Define How the Data is Stored Either 'dict' or 'raw'
Data_Type = 'dict'

# Define Filename
FileName = 'fixed_mouse_brain_data'

# Define Material Model to Use Either 'neoHookean' or 'Gent'
Mat_Mod = 'Gent'

# Define RMSE Threshold
Thresh = 1e-8

# If Data_Type = 'raw' Define Sweep size (Xs, Ys)
Xs = 3
Ys = 3
# If Data_Type = 'raw' Define Tissue Sample Size in meters
Material_Width = 4e-3 # (m)
Material_Thick = 4e-3 # (m)




# Data_Type = 'dict'
# FileName = 'Test_Example_Dict'
# FileName = 'fixed_mouse_brain_data'
# FileName = 'mouse_brain_data'


# Data_Type = 'raw'
# FileName = '1w 40 min_2_09.29.21'
# DirName = 'Test_Example_Raw'




# =============================================================================
# Load Experimental Data
# =============================================================================
if Data_Type == 'dict':
    DataFile = exp_files + '\\%s.pickle'%(FileName)
    # DataFile = exp_files + '\\%s.pkl'%(FileName)
    ExpData = pickle.load(open(DataFile,'rb'))
    Keys = list( ExpData.keys() )

    # Convert data to micron length scale
    ExpData_new = {}
    for n in range(len(ExpData)):
        ExpData_new['Sample_%i'%(n)] = {'radius':(25e-6), 
                                        'Height':40*(25e-6), 
                                        'Width':40*(25e-6), 
                                        'Load': ExpData[Keys[n]][4]*(1e-6), 
                                        'Indentation':  ExpData[Keys[n]][3]*(1e-6) }
    ExpData = ExpData_new
    Keys = list( ExpData.keys() )
    
    
    # filename = 'data_fixed_mouse_brain'
    # file = '%s.pickle'%(filename)
    # with open(file, 'wb') as fp:
    #     pickle.dump(ExpData, fp)
    #     print('dictionary saved successfully to file')
    






elif Data_Type == 'raw':
    DataFile = exp_files + '\\%s\\%s 1 S-1 X-1 Y-1 I-1.txt'%(DirName,FileName)
    ExpData = AnalyzeExpData(exp_files,DirName,FileName, Xs, Ys, Material_Width, Material_Thick)
    Keys = list( ExpData.keys() )





# =============================================================================
# Load Trained Neural Networks
# =============================================================================
model_Gent_Inverse_ABFits = keras.models.load_model(nn_files+  "\\NN_Gent_Inverse_ABFits")
model_Gent_Forward = keras.models.load_model(nn_files + "\\NN_Gent_Forward")
model_NeoHookean_Inverse = keras.models.load_model(nn_files + "\\NN_NeoHookean_Inverse")
model_NeoHookean_Forward = keras.models.load_model(nn_files + "\\NN_NeoHookean_Forward")




# =============================================================================
# Extract Mouse and Brain Data, and Resample
# =============================================================================
ExpData_u_Hertz, ExpData_u_ModHertz, ExpData_Load, ExpData_Dim, RMSE_Hertz, RMSE_ModHertz, ExpData,Keys = ExtractData_HertzFits(ExpData,Keys)



# =============================================================================
#  Prepare Experimental Data for Direct Inverse Approach
# =============================================================================
if Mat_Mod == 'neoHookean':
    Exp_Input_NH =  Prepare_DirInv_NH(ExpData,Keys,syn_files, ExpData_Load, ExpData_Dim)

elif Mat_Mod == 'Gent':
    Exp_Input_GT, Scales =  Prepare_DirInv_ABFits_GT(ExpData,Keys,syn_files)



# =============================================================================
# Direct Inverse ML Approach to Parameter Identification
# =============================================================================
if Mat_Mod == 'neoHookean':
    Exp_Params_DirInv_NH = Direct_Inverse_ML_Approach_neoHookean_Data(model_NeoHookean_Inverse, Exp_Input_NH, ExpData, Keys)

elif Mat_Mod == 'Gent':
    Exp_Params_DirInv_GT = Direct_Inverse_ML_Approach_Gent_Data(model_Gent_Inverse_ABFits, Exp_Input_GT, Scales,ExpData,Keys)



# =============================================================================
# Calculate RMSE Using Forward Gent Model
# =============================================================================
if Mat_Mod == 'neoHookean':
    Exp_Params_DirInv_NH, Fits_DirInv_NH = RMSE_Approx(model_Gent_Forward,syn_files,ExpData_Dim,ExpData_Load, Exp_Params_DirInv_NH, Scales)

elif Mat_Mod == 'Gent':
    Exp_Params_DirInv_GT, Fits_DirInv_GT = RMSE_Approx(model_Gent_Forward,syn_files,ExpData_Dim,ExpData_Load, Exp_Params_DirInv_GT, Scales)

# Exp_Params_DirInv = CleanPredictions(Exp_Params_DirInv_GT, Exp_Params_DirInv_NH)



# =============================================================================
# Least Squares ML Approach to Increase Accuracy
# =============================================================================
if Mat_Mod == 'neoHookean':
    Exp_Params_Lsq_NH, Fits_Lsq_NH  =  LeastSquaresPrediction_NH(Thresh, syn_files, model_Gent_Forward, Exp_Params_DirInv_NH, ExpData_Dim,ExpData_Load,Scales, Fits_DirInv_NH)

elif Mat_Mod == 'Gent':
    Exp_Params_Lsq_GT, Fits_Lsq_GT  =  LeastSquaresPrediction(Thresh, syn_files, model_Gent_Forward, Exp_Params_DirInv_GT, ExpData_Dim,ExpData_Load,Scales, Fits_DirInv_GT)


# Exp_Params_Lsq_GT_GN, Fits_Lsq_GT_GN  =  GaussNewtonMLPrediction(Thresh, syn_files, model_Gent_Forward, Exp_Params_DirInv_GT, ExpData_Dim, ExpData_Load,Scales, Fits_DirInv_GT)


# =============================================================================
# Plotting
# =============================================================================
if Mat_Mod == 'neoHookean':
    Plot(ExpData, Fits_Lsq_NH, Exp_Params_Lsq_NH)

elif Mat_Mod == 'Gent':
    Plot(ExpData, Fits_Lsq_GT, Exp_Params_Lsq_GT)

# np.savetxt('%s\\%s_MaterialParameters.txt'%(res_files,FileName), Exp_Params_DirInv_GT)






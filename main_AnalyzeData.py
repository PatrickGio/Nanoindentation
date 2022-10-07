


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
import matplotlib.cm as cm

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
Data_Type = 'dict'
FileName = 'Test_Example_Dict'

# Data_Type = 'raw'
# FileName = '1w 40 min_2_09.29.21'
# DirName = 'Test_Example_Raw'
Xs = 3
Ys = 3
Material_Width = 4e-3
Material_Thick = 4e-3


Thresh = 5e-8






# =============================================================================
# Load Experimental Data
# =============================================================================
if Data_Type == 'dict':
    DataFile = exp_files + '\\%s.pkl'%(FileName)
    ExpData = pickle.load(open(DataFile,'rb'))
    Keys = list( ExpData.keys() )

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
Exp_Input, Scales =  Prepare_DirInv_ABFits_GT(ExpData,Keys,syn_files)

# =============================================================================
# Direct Inverse ML Approach to Parameter Identification
# =============================================================================
Exp_Params_DirInv_GT = Direct_Inverse_ML_Approach_Gent_Data(model_Gent_Inverse_ABFits, Exp_Input, Scales,ExpData,Keys)

# =============================================================================
# Calculate RMSE Using Forward Gent Model
# =============================================================================
Exp_Params_DirInv_GT, Fits_DirInv = RMSE_Approx(model_Gent_Forward,syn_files,ExpData_Dim,ExpData_Load, Exp_Params_DirInv_GT)

# =============================================================================
# Least Squares ML Approach to Increase Accuracy
# =============================================================================
Exp_Params_DirInv_GT  =  CleanDirectInversePrediction(Thresh, syn_files, model_Gent_Forward, Exp_Params_DirInv_GT, ExpData_Dim,ExpData_Load)





# =============================================================================
# Plotting
# =============================================================================
Plot(ExpData,Fits_DirInv,Exp_Params_DirInv_GT)
np.savetxt('%s\\%s_MaterialParameters.txt'%(res_files,FileName), Exp_Params_DirInv_GT)








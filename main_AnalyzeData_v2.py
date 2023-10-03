# =============================================================================
#   Import Libraries
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
import argparse

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


# =============================================================================
#   Import User Defined Functions
# =============================================================================
current_path = os.getcwd()
current_dir  = os.path.basename(current_path)
parent_path  = os.path.abspath(os.path.join(current_path, os.pardir))

func_files = os.path.join(current_path, 'functions')
nn_files   = os.path.join(current_path, 'trained_metamodels')
exp_files  = os.path.join(current_path, 'data', 'ExperimentalData')
syn_files  = os.path.join(current_path, 'data', 'SynthData')
res_files  = os.path.join(current_path, 'results')

sys.path.append(current_dir)
sys.path.append(func_files)
sys.path.append(nn_files)
sys.path.append(exp_files)
sys.path.append(syn_files)

from subfunctions_Analyze_v2 import *


    
# =============================================================================
#   User Defined Parameters
# =============================================================================
# Define How the Data is Stored Either 'dict' or 'raw'
Data_Type = 'dict'

# Define Filename
FileName = 'data_mouse_brain'

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
#   Load Experimental Data
# =============================================================================
if Data_Type == 'dict':
    DataFile = exp_files + '\\%s.pickle'%(FileName)
    # DataFile = exp_files + '\\%s.pkl'%(FileName)
    ExpData = pickle.load(open(DataFile,'rb'))
    Keys = list( ExpData.keys() )

    # # Rename 
    # ExpData_new = {}
    # for n in range(len(ExpData)):
    #     ExpData_new['Sample_%i'%(n)] = {'radius': ExpData[Keys[n]]['radius'], 
    #                                     'Height': np.min([ExpData[Keys[n]]['Height'], 40*(25e-6)]), 
    #                                     'Width':  np.min([ExpData[Keys[n]]['Width'],  40*(25e-6)]), 
    #                                     'Load':         ExpData[Keys[n]]['Load'], 
    #                                     'Indentation':  ExpData[Keys[n]]['Indentation']}
    # ExpData = ExpData_new
    # Keys = list( ExpData.keys() )


elif Data_Type == 'raw':
    DataFile = exp_files + '\\%s\\%s 1 S-1 X-1 Y-1 I-1.txt'%(DirName,FileName)
    ExpData  = AnalyzeExpData(exp_files,DirName,FileName, Xs, Ys, Material_Width, Material_Thick)
    Keys     = list( ExpData.keys() )



# =============================================================================
#   Load Trained Neural Networks
# =============================================================================
model_Gent_Inverse_ABFits = keras.models.load_model(os.path.join(nn_files, "NN_Gent_Inverse_ABFits"))
model_Gent_Forward        = keras.models.load_model(os.path.join(nn_files, "NN_Gent_Forward"       ))
model_NeoHookean_Inverse  = keras.models.load_model(os.path.join(nn_files, "NN_NeoHookean_Inverse_Fits" ))
model_NeoHookean_Forward  = keras.models.load_model(os.path.join(nn_files, "NN_NeoHookean_Forward" ))



# def main():
# =============================================================================
#    Extract Mouse and Brain Data, and Resample
# =============================================================================
(ExpData_u_Hertz, ExpData_u_ModHertz, ExpData_Load, ExpData_Dim, RMSE_Hertz, RMSE_ModHertz, ExpData, Keys) = ExtractData_HertzFits(ExpData,Keys)



# =============================================================================
#    Prepare Experimental Data for Direct Inverse Approach
# =============================================================================
# if Mat_Mod == 'neoHookean':
Exp_Input_NH, Scales_NH = Prepare_DirInv_NH(ExpData,Keys,syn_files, ExpData_Load, ExpData_Dim)

# elif Mat_Mod == 'Gent':
Exp_Input_GT, Scales_GT =  Prepare_DirInv_ABFits_GT(ExpData,Keys,syn_files)



# =============================================================================
# Direct Inverse ML Approach to Parameter Identification
# =============================================================================
start = time.localtime()
if Mat_Mod == 'neoHookean':                                           
    Exp_Params_DirInv_NH = Direct_Inverse_ML_Approach_neoHookean_Data(model_NeoHookean_Inverse,  Exp_Input_NH, Scales_NH, ExpData, Keys)

elif Mat_Mod == 'Gent':
    Exp_Params_DirInv_GT = Direct_Inverse_ML_Approach_Gent_Data(      model_Gent_Inverse_ABFits, Exp_Input_GT, Scales_GT, ExpData, Keys)

end1 = time.localtime()
print('DirInv Walltime:', (end1[3]-start[3])*3600 + (end1[4]-start[4])*60 + (end1[5]-start[5])*1)



# =============================================================================
# Calculate RMSE Using Forward Gent Model
# =============================================================================
if Mat_Mod == 'neoHookean':
    Exp_Params_DirInv_NH, Fits_DirInv_NH = RMSE_Approx(model_Gent_Forward,syn_files,ExpData_Dim,ExpData_Load, Exp_Params_DirInv_NH, Scales_GT)

elif Mat_Mod == 'Gent':
    Exp_Params_DirInv_GT, Fits_DirInv_GT = RMSE_Approx(model_Gent_Forward,syn_files,ExpData_Dim,ExpData_Load, Exp_Params_DirInv_GT, Scales_GT)



# =============================================================================
# Least Squares ML Approach to Increase Accuracy
# =============================================================================
start2 = time.localtime()
if Mat_Mod == 'neoHookean':
    Exp_Params_Lsq_NH, Fits_Lsq_NH  =  LeastSquaresPrediction_NH(Thresh, syn_files, model_Gent_Forward, Exp_Params_DirInv_NH, ExpData_Dim, ExpData_Load, Scales_GT, Fits_DirInv_NH)

elif Mat_Mod == 'Gent':
    Exp_Params_Lsq_GT, Fits_Lsq_GT  =  LeastSquaresPrediction_GT(Thresh, syn_files, model_Gent_Forward, Exp_Params_DirInv_GT, ExpData_Dim, ExpData_Load, Scales_GT, Fits_DirInv_GT)

end2 = time.localtime()
print('Lsq Walltime:', (end2[3]-start2[3])*3600 + (end2[4]-start2[4])*60 + (end2[5]-start2[5])*1)
# Exp_Params_Lsq_GT_GN, Fits_Lsq_GT_GN  =  GaussNewtonMLPrediction(Thresh, syn_files, model_Gent_Forward, Exp_Params_DirInv_GT, ExpData_Dim, ExpData_Load,Scales, Fits_DirInv_GT)



# =============================================================================
# Plotting
# =============================================================================
if Mat_Mod == 'neoHookean':
    Plot(ExpData, Fits_Lsq_NH, Exp_Params_Lsq_NH)

elif Mat_Mod == 'Gent':
    Plot(ExpData, Fits_Lsq_GT, Exp_Params_Lsq_GT)

# np.savetxt('%s\\%s_MaterialParameters.txt'%(res_files,FileName), Exp_Params_DirInv_GT)

























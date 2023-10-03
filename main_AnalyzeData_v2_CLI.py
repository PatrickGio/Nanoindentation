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
#       Command Line Arguments
# =============================================================================
def parse_cmd_line():
    parser = argparse.ArgumentParser(description="Command Line Inputs for the \
                                                  Inverse Nanoindenter Model")
    
    parser.add_argument('--Data_Type',
                        type    = str,
                        default = 'dict',
                        help    = 'Define how the data is stored either <dict> or <raw>,' 
                                  'default = dict')
    
    parser.add_argument('--FileName',
                        type    = str,
                        default = 'data_mouse_brain',
                        help    = 'Name of experimental data file,' 
                                  'default = data_mouse_brain')
    
    parser.add_argument('--Mat_Mod',
                        type    = str,
                        default = 'Gent',
                        help    = 'Material model used in analysis currently either <Gent> or <neoHookean>,' 
                                  'default = Gent')
    
    parser.add_argument('--Thresh',
                        type    = float,
                        default = 1e-8,
                        help    = 'Define RMSE threshold value to determine whether to run LSq correction,' 
                                  'default = 1e-8')
    
    parser.add_argument('--DirName',
                        type    = str,
                        default = 'Data',
                        help    = 'If Data_Type = <raw>, name of experimental data directory,' 
                                  'default = Data')
    
    parser.add_argument('--Xs',
                        type    = int,
                        default = 3,
                        help    = 'If Data_Type = <raw> Define Sweep size (Xs, Ys),' 
                                  'default = 3')
    
    parser.add_argument('--Ys',
                        type    = int,
                        default = 3,
                        help    = 'If Data_Type = <raw> Define Sweep size (Xs, Ys),' 
                                  'default = 3')
    
    parser.add_argument('--Material_Width',
                        type    = float,
                        default =  4e-3,
                        help    = 'If Data_Type = <raw> Define Tissue Sample Size in meters,' 
                                  'default = 4e-3')
    
    parser.add_argument('--Material_Thick',
                        type    = float,
                        default =  4e-3,
                        help    = 'If Data_Type = <raw> Define Tissue Sample Size in meters,' 
                                  'default = 4e-3')
    
    args = parser.parse_args()

    return args

    




def main():
    ''' main function
        -------------
        |-> Load Experimental Data
        |-> Load Trained Neural Networks
        |-> Prepare Experimental Data for Direct Inverse Approach
        |-> Direct Inverse ML Approach to Parameter Identification 
        |-> Calculate RMSE Using Forward Gent Model 
        |-> Least Squares ML Approach to Increase Accuracy
    '''
    
    
    # =========================================================================
    #   Load Experimental Data
    # =========================================================================
    if args.Data_Type == 'dict':
        DataFile = os.path.join( exp_files, '%s.pickle'%(args.FileName) )
        # DataFile = exp_files + '\\%s.pkl'%(FileName)
        ExpData = pickle.load(open(DataFile,'rb'))
        Keys = list( ExpData.keys() )

    elif args.Data_Type == 'raw':
        DataFile = os.path.join( exp_files, args.DirName, '%s 1 S-1 X-1 Y-1 I-1.txt'%(args.FileName) )
        ExpData  = AnalyzeExpData(exp_files, args.DirName, args.FileName, args.Xs, args.Ys, args.Material_Width, args.Material_Thick)
        Keys     = list( ExpData.keys() )



    # =========================================================================
    #   Load Trained Neural Networks
    # =========================================================================
    model_Gent_Inverse_ABFits = keras.models.load_model(os.path.join(nn_files, "NN_Gent_Inverse_ABFits"))
    model_Gent_Forward        = keras.models.load_model(os.path.join(nn_files, "NN_Gent_Forward"       ))
    model_NeoHookean_Inverse  = keras.models.load_model(os.path.join(nn_files, "NN_NeoHookean_Inverse_Fits" ))
    model_NeoHookean_Forward  = keras.models.load_model(os.path.join(nn_files, "NN_NeoHookean_Forward" ))

    
    
    # =========================================================================
    #    Extract Mouse and Brain Data, and Resample
    # =========================================================================
    (ExpData_u_Hertz, ExpData_u_ModHertz, ExpData_Load, ExpData_Dim, RMSE_Hertz, RMSE_ModHertz, ExpData, Keys) = ExtractData_HertzFits(ExpData,Keys)
    
    
    
    # =========================================================================
    #    Prepare Experimental Data for Direct Inverse Approach
    # =========================================================================
    if args.Mat_Mod == 'neoHookean':
        Exp_Input_NH, Scales_NH = Prepare_DirInv_NH(ExpData,Keys,syn_files, ExpData_Load, ExpData_Dim)
        
    elif args.Mat_Mod == 'Gent':
        Exp_Input_GT, Scales_GT =  Prepare_DirInv_ABFits_GT(ExpData,Keys,syn_files)
        
    
    
    # =========================================================================
    #   Direct Inverse ML Approach to Parameter Identification
    # =========================================================================
    start = time.localtime()
    if args.Mat_Mod == 'neoHookean':                                           
        Exp_Params_DirInv_NH = Direct_Inverse_ML_Approach_neoHookean_Data(model_NeoHookean_Inverse,  Exp_Input_NH, Scales_NH, ExpData, Keys)
    
    elif args.Mat_Mod == 'Gent':
        Exp_Params_DirInv_GT = Direct_Inverse_ML_Approach_Gent_Data(      model_Gent_Inverse_ABFits, Exp_Input_GT, Scales_GT, ExpData, Keys)
    
    end1 = time.localtime()
    print('DirInv Walltime:', (end1[3]-start[3])*3600 + (end1[4]-start[4])*60 + (end1[5]-start[5])*1)
    
    
    
    # =========================================================================
    #   Calculate RMSE Using Forward Gent Model
    # =========================================================================
    if args.Mat_Mod == 'neoHookean':
        Exp_Params_DirInv_NH, Fits_DirInv_NH = RMSE_Approx(model_Gent_Forward,syn_files,ExpData_Dim,ExpData_Load, Exp_Params_DirInv_NH, Scales_GT)
    
    elif args.Mat_Mod == 'Gent':
        Exp_Params_DirInv_GT, Fits_DirInv_GT = RMSE_Approx(model_Gent_Forward,syn_files,ExpData_Dim,ExpData_Load, Exp_Params_DirInv_GT, Scales_GT)
    

    
    # =========================================================================
    #   Least Squares ML Approach to Increase Accuracy
    # =========================================================================
    start2 = time.localtime()
    if args.Mat_Mod == 'neoHookean':
        Exp_Params_Lsq_NH, Fits_Lsq_NH  =  LeastSquaresPrediction_NH(args.Thresh, syn_files, model_Gent_Forward, Exp_Params_DirInv_NH, ExpData_Dim, ExpData_Load, Scales_GT, Fits_DirInv_NH)
    
    elif args.Mat_Mod == 'Gent':
        Exp_Params_Lsq_GT, Fits_Lsq_GT  =  LeastSquaresPrediction_GT(args.Thresh, syn_files, model_Gent_Forward, Exp_Params_DirInv_GT, ExpData_Dim, ExpData_Load, Scales_GT, Fits_DirInv_GT)
    
    end2 = time.localtime()
    print('Lsq Walltime:', (end2[3]-start2[3])*3600 + (end2[4]-start2[4])*60 + (end2[5]-start2[5])*1)
    # Exp_Params_Lsq_GT_GN, Fits_Lsq_GT_GN  =  GaussNewtonMLPrediction(args.Thresh, syn_files, model_Gent_Forward, Exp_Params_DirInv_GT, ExpData_Dim, ExpData_Load,Scales, Fits_DirInv_GT)
    
    
    
    # =========================================================================
    #   Plotting
    # =========================================================================
    if args.Mat_Mod == 'neoHookean':
        Plot(ExpData, Fits_Lsq_NH, Exp_Params_Lsq_NH)
    
    elif args.Mat_Mod == 'Gent':
        Plot(ExpData, Fits_Lsq_GT, Exp_Params_Lsq_GT)
    
    # np.savetxt('%s\\%s_MaterialParameters.txt'%(res_files,args.FileName), Exp_Params_DirInv_GT)
    
    
    
    if args.Mat_Mod == 'neoHookean':
        return Exp_Params_Lsq_NH, Fits_Lsq_NH
    
    elif args.Mat_Mod == 'Gent':
        return Exp_Params_Lsq_GT, Fits_Lsq_GT





if __name__ == '__main__':
    
    # =========================================================================
    #   Command Line Arguements
    # =========================================================================
    args = parse_cmd_line()



    # =========================================================================
    #   main function
    # =========================================================================
    Exp_Params, Fits = main()















import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as optimize
import math as ma
from scipy.signal import argrelmax
from scipy.ndimage.filters import gaussian_filter1d


def Fit_model(inputs, a, b):
    x = (inputs[:])
    return  a*(x)**b

def Hertz(inputs, a):
    x = (inputs[:])
    return  a*(x)**1.5
    
def ModHertz(inputs, a):
    x = (inputs[:])
    return  4/3*np.sqrt(25e-6) * x**1.5 * (a/(1-0.4995**2)) * (1 - 0.15*x/(25e-6))




def ExtractData_HertzFits(ExpData,Keys):
    """
    Objective
        Extract experimental data and resample

    Parameters
    ----------
    ExpData : (dict)
        Stores experimental data, height, width, radius, Indentation and load 
        data.
    Keys : (list)
        Extracted name of outer dictionary keyes.

    Returns
    -------
    ExpData_u_Hertz : (array)
        Stores the shear moduli predicted by a Hertzian fit.
    ExpData_u_ModHertz : (array)
        Stores the shear moduli predicted by a Hertzian fit.
    ExpData_Load : (array)
        Stores load data from samples.
    ExpData_Dim : (array)
        Stores spatial data from samples.
    RMSE_Hertz : (array)
        Stores RMSE of Hetzian fit.
    RMSE_ModHertz : (array)
        Stores RMSE of Hetzian fit.

    """
    
    Ndat = len(ExpData)
    Nd = 100 
    
    # Arrays to store the loading curves, sample dimensions, and indenter radii 
    ExpData_Load_Fit = np.zeros((Ndat, Nd))
    ExpData_Load = np.zeros((Ndat, Nd))
    ExpData_Dim = np.zeros((Ndat, 2))
    ExpData_Rad = np.zeros((Ndat))

    # Loop over samples 
    for i in range(0,Ndat):
        # Extract data
        ExpData_Rad[i] = ExpData[Keys[i]]['radius']
        ExpData_Dim[i,0] = ExpData[Keys[i]]['Width']/ExpData[Keys[i]]['radius']
        ExpData_Dim[i,1] = ExpData[Keys[i]]['Height']/ExpData[Keys[i]]['radius']
        if ExpData_Dim[i,0] > 40: ExpData_Dim[i,0] = 40
        if ExpData_Dim[i,1] > 40: ExpData_Dim[i,1] = 40
        
        disp = ExpData[Keys[i]]['Indentation']/ExpData[Keys[i]]['radius']
        load = ExpData[Keys[i]]['Load']
        
        # Resampling
        xi = np.linspace(0, 0.5, Nd+1 )[1:]; xi_clip = np.copy(xi); xi_clip[xi_clip  >  max(disp)] = 0
        newp, pcov_new = optimize.curve_fit(Fit_model, abs(disp), load ,ftol=1e-15, xtol=1e-15, maxfev=800000 )
        
        ExpData_Load_Fit[i,:] = newp[0]*xi**newp[1]
        ExpData_Load[i,:] = np.interp(xi_clip, disp, load)[:]

        
        
    # Averaged Experimental Data
    ExpData_Load_Avg = np.average(ExpData_Load_Fit,axis=0)
    ExpData_W_Avg = np.average(ExpData_Dim[:,0])
    ExpData_H_Avg = np.average(ExpData_Dim[:,1])
    ExpData_R_Avg = np.average(ExpData_Rad[:])
    ExpData['Averaged'] = {'radius':ExpData_R_Avg, 'Height':ExpData_H_Avg, 'Width':ExpData_W_Avg, 'Load':ExpData_Load_Avg, 'Indentation': xi*ExpData_R_Avg}
    
    # include averaged data
    Keys = list( ExpData.keys() )
    Ndat = len(ExpData)
    ExpData_Load_Fit = np.append(ExpData_Load_Fit, np.array([ExpData_Load_Avg]), axis=0)
    ExpData_Dim = np.append(ExpData_Dim, np.array([[ExpData_W_Avg, ExpData_H_Avg]]), axis=0  )

    # Find Hertzian and Modified Hertzian fits for all data samples
    ExpData_u_Hertz = np.zeros((Ndat, 1)); ExpData_E_Hertz = np.zeros((Ndat, 1))
    ExpData_u_ModHertz = np.zeros((Ndat, 1)); ExpData_E_ModHertz = np.zeros((Ndat, 1))
    RMSE_Hertz = np.zeros(Ndat)
    RMSE_ModHertz = np.zeros(Ndat)
    for i in range(0,Ndat):
        # Extract Data
        radius = ExpData[Keys[i]]['radius']
        disp = ExpData[Keys[i]]['Indentation']/radius
        load = ExpData[Keys[i]]['Load']
                
        # Find Fits
        newp_h, pcov_new = optimize.curve_fit(Hertz, abs(disp)*radius, load[:], ftol=1e-15, xtol=1e-15, maxfev=800000 )
        newp_mh, pcov_new = optimize.curve_fit(ModHertz, abs(disp)*radius,  load[:],ftol=1e-15, xtol=1e-15, maxfev=800000 )
        
        # Calculate Effective Youngs Modulus
        nu = 0.4995
        Eff_h = (newp_h*3/4/np.sqrt(25e-6))
        Eff_mh = (newp_mh*3/4/np.sqrt(25e-6))
        
        # Calculate Shear Modulus and Youngs Modulus
        ExpData_u_Hertz[i] = (Eff_h*(1-nu**2))   /(2+2*nu) #*((radius*unit_rad)**2)/((25e-6)**2)
        ExpData_E_Hertz[i] = Eff_h
        ExpData_u_ModHertz[i] = newp_mh/(2+2*nu) # / ( ((radius*unit_rad)**2)/((25e-6)**2) )
        ExpData_E_ModHertz[i] = Eff_mh
    
        # Find RMSE between the Hertzian Fits and Data
        RMSE_Hertz[i] = np.sqrt(np.square( np.subtract(ExpData_Load_Fit[i,:], Hertz(xi*radius, newp_h[0])  ) )).mean() 
        RMSE_ModHertz[i] = np.sqrt(np.square( np.subtract(ExpData_Load_Fit[i,:], ModHertz(xi*radius, newp_mh[0])  ))).mean() 
        
    return ExpData_u_Hertz, ExpData_u_ModHertz, ExpData_Load_Fit, ExpData_Dim, RMSE_Hertz, RMSE_ModHertz, ExpData,Keys
    
    



def Prepare_DirInv_NH(ExpData,Keys,syn_files, ExpData_Load, ExpData_Dim):
    """
    Objective
        Prepare and reshape data for direct inverse ML approach trained with the 
        neo-Hookean material model

    Parameters
    ----------
    ExpData : (dict)
        Stores experimental data, height, width, radius, Indentation and load 
        data.
    Keys : (list)
        Extracted name of outer dictionary keyes.
    syn_files : (string)
        path for synthetic data.
    ExpData_Load : (array)
        Load curve of experimental data
    ExpData_Dim : (array)
        Spatial dimensions of experimental data.
        
    Returns
    -------
    Exp_Input : (array)
        Input for direct inverse neural network.

    """
    # Load synthetic data
    Model_Output = np.loadtxt(syn_files + "\\Model_Output_NH")
    Model_Input = np.loadtxt(syn_files + "\\Model_Input_NH")
    
    Nd = 100
    Nx= len(Model_Output[:,0])
    Nsamp= len(Model_Output[0,:])
    Ndat = len(ExpData)
    
    # Resample Synthetic Data
    FEA_disp = np.linspace(0,0.5,int(Nx+1))[1:]
    Model_Output_Resampled = np.zeros((Nd, Nsamp))
    for i in range(0,Nsamp):
        # Fitting and Resample
        FEA_Load = Model_Output[:,i]
        newp, pcov_new = optimize.curve_fit( Fit_model, FEA_disp, FEA_Load, ftol=1e-15, xtol=1e-15, maxfev=800000)
        xi = np.linspace(0, 0.5, Nd+1 )[1:]
        Model_Output_Resampled[:,i] = Fit_model(xi,*newp)

    # Create and Scale Neural Network Features
    Exp_Input = np.zeros((Ndat,2+Nd))
    minw = np.min(ExpData_Dim[:,0]);  maxw = np.max(ExpData_Dim[:,0])
    minh = np.min(ExpData_Dim[:,1]);  maxh = np.max(ExpData_Dim[:,1])
    if minw == maxw: Exp_Input[:,0] =  1
    else:  Exp_Input[:,0] =  ((np.copy(ExpData_Dim[:,0]) - minw)/(maxw - minw)*1 - 0) * 1
    if minh == maxh: Exp_Input[:,1] =  1
    else:  Exp_Input[:,1] =  ((np.copy(ExpData_Dim[:,1]) - minh)/(maxh - minh)*1 - 0) * 1
    for i in range(0,Nd):
        minP = min(Model_Output_Resampled[i,:]); maxP = max(Model_Output_Resampled[i,:])
        Exp_Input[:,i+2] =  ((np.copy(ExpData_Load[:,i]) - minP)/(maxP - minP)*1 - 0) * 1
        
    return Exp_Input









def Prepare_DirInv_ABFits_GT(ExpData, Keys, syn_files):
    """
    Objective
        Prepare and reshape data for direct inverse ML approach

    Parameters
    ----------
    ExpData : (dict)
        Stores experimental data, height, width, radius, Indentation and load 
        data.
    Keys : (list)
        Extracted name of outer dictionary keyes.
    syn_files : (string)
        path for synthetic data.

    Returns
    -------
    Exp_Input : (array)
        Input for direct inverse neural network.
    Scales : (dict)
        Stores scales for inputs and outputs of neural network.

    """
    
    # =============================================================================
    # Scale Features for Experimental Prediction
    # =============================================================================
    # Load synthetic data
    Model_Output = np.loadtxt(syn_files + "\\Model_Output_ExtraSmallJm")
    Model_Input = np.loadtxt(syn_files + "\\Model_Input_ExtraSmallJm")
    
    nparams = 2; Nd = 100;  Ndat = len(ExpData)
    Nx= len(Model_Output[:,0])
    Nsamp= len(Model_Output[0,:])
    
    # Resampling of Synthetic Loading Curves
    FEA_disp = np.linspace(0,0.5,int(Nx+1))[1:]
    FEA_fit_params = np.zeros((Nsamp,nparams))
    Model_Output_Resampled = np.zeros((Nd, Nsamp))
    for i in range(0,Nsamp):
        FEA_Load = Model_Output[:,i]
        newp, pcov_new = optimize.curve_fit( Fit_model, FEA_disp, FEA_Load, ftol=1e-15, xtol=1e-15, maxfev=800000)
        FEA_fit_params[i,:] = newp
        xi = np.linspace(0, 0.5, Nd+1 )[1:]
        Model_Output_Resampled[:,i] = Fit_model(xi,*newp)
        
    # Create Scales for Scaling NN Input Features and Output
    minA = min(np.log10(FEA_fit_params[:,0])); maxA = max(np.log10(FEA_fit_params[:,0]))
    minw = min(Model_Input[:,0]); maxw = max(Model_Input[:,0])
    minh = min(Model_Input[:,1]); maxh = max(Model_Input[:,1])
    minB = min(FEA_fit_params[:,1]); maxB = max(FEA_fit_params[:,1])
    minU = 2; maxU = 6
    # minJ = min(Model_Input[:,3]); maxJ = max(Model_Input[:,3])
    minJ = min(np.log10(Model_Input[:,3])); maxJ = max(np.log10(Model_Input[:,3]))
    minP = np.zeros(Nd); maxP = np.zeros(Nd)
    for i in range(0,Nd):
        minP[i] = min(np.log10(Model_Output_Resampled[i,:])); maxP[i] = max(np.log10(Model_Output_Resampled[i,:]))
        
    Scales = {'minw':minw, 'maxw':maxw,'minh':minh, 'maxh':maxh,'minA':minA, 'maxA':maxA, 'minB':minB, 'maxB':maxB, 'minU':minU, 'maxU':maxU, 'minJ':minJ, 'maxJ':maxJ, 'minP':minP, 'maxP':maxP}        
        

    
    Exp_fit_params = np.zeros((Ndat,nparams))
    ExpData_Dim = np.zeros((Ndat, 2))
    for i in range(0,Ndat):
        # Extract Information from Experimental Data
        radius = ExpData[Keys[0]]['radius']
        ExpData_Dim[i,0] = ExpData[Keys[i]]['Width']/radius
        ExpData_Dim[i,1] = ExpData[Keys[i]]['Height']/radius
        if ExpData_Dim[i,0] > 40: ExpData_Dim[i,0] = 40
        if ExpData_Dim[i,1] > 40: ExpData_Dim[i,1] = 40
        
        # Fit Experimental Loading Curves
        disp = ExpData[Keys[i]]['Indentation']/radius
        load = ExpData[Keys[i]]['Load']
        mxdisp = max(disp)
        xi = np.linspace(0, 0.5, Nd+1 )[1:]; xi_clip = np.copy(xi); xi_clip[xi_clip  >=  mxdisp] = 0
        newp, pcov_new = optimize.curve_fit(Fit_model, abs(disp), load, ftol=1e-15, xtol=1e-15, maxfev=800000 )
        
        # Store the parameters of fit
        Exp_fit_params[i,:] = newp
        fit = newp[0]*xi_clip**newp[1]

    # Scale Neural Network Features
    Exp_Input = np.zeros((Ndat, 4))
    minw = np.min(ExpData_Dim[:,0]);  maxw = np.max(ExpData_Dim[:,0])
    minh = np.min(ExpData_Dim[:,1]);  maxh = np.max(ExpData_Dim[:,1])
    # Spatial Features
    if minw == maxw: Exp_Input[:,0] =  1
    else:  Exp_Input[:,0] =  ((np.copy(ExpData_Dim[:,0]) - minw)/(maxw - minw)*1 - 0) * 1
    if minh == maxh: Exp_Input[:,1] =  1
    else:  Exp_Input[:,1] =  ((np.copy(ExpData_Dim[:,1]) - minh)/(maxh - minh)*1 - 0) * 1
    # y=a*x^b Fit Features
    Exp_Input[:,2] =  ((np.log10(np.copy(Exp_fit_params[:,0])) - minA)/(maxA - minA)*1 - 0.0) * 1
    Exp_Input[:,3] =  ((np.copy(Exp_fit_params[:,1]) - minB)/(maxB - minB)*1 - 0.0) * 1
    
    return Exp_Input, Scales












# =============================================================================
# Direct Inverse ML Approach to Parameter Identification
# =============================================================================

def Direct_Inverse_ML_Approach_neoHookean_Data(model_NeoHookean_Inverse, Exp_Input, ExpData, Keys):
    """
    Objective
        Solve the inverse problem for parameter identification with the direct
        inverse ML approach

    Parameters
    ----------
    model_NeoHookean_Inverse : (keras model)
        Trained weights and architecture of direct inverse neural network.
    Exp_Input : (array)
        Input for direct inverse neural network.
    Scales : (dict)
        Stores scales for inputs and outputs of neural network.
    ExpData : (dict)
        Stores experimental data, height, width, radius, Indentation and load 
        data.
        
    Returns
    -------
    Exp_Params_Cl : (array)
        Identified material parameters.

    """
        
    Exp_Params = np.zeros((len(Exp_Input[:,0]),2 ))
    
    for i in range(0,len(Exp_Input[:,0])):
        # Predict Shear Modulus
        yhat = model_NeoHookean_Inverse.predict([Exp_Input[i,:].tolist()]);
        # Re-dimensionalize 
        Exp_Params[i,0] = (yhat / ((ExpData[Keys[i]]['radius'])**2  / (25e-6)**2 )  ) 
        # Not Necessary, Stores 10 as the max Jm value
        Exp_Params[i,1] = 10

    return Exp_Params
    




def Direct_Inverse_ML_Approach_Gent_Data(model_Gent_Inverse_ABFits, Exp_Input, Scales, ExpData, Keys):
    """
    Objective
        Solve the inverse problem for parameter identification with the direct
        inverse ML approach

    Parameters
    ----------
    model_Gent_Inverse_ABFits : (keras model)
        Trained weights and architecture of direct inverse neural network.
    Exp_Input : (array)
        Input for direct inverse neural network.
    Scales : (dict)
        Stores scales for inputs and outputs of neural network.
    ExpData : (dict)
        Stores experimental data, height, width, radius, Indentation and load 
        data.
        
    Returns
    -------
    Exp_Params_Cl : (array)
        Identified material parameters.

    """
    Exp_Params_Scaled = np.zeros((len(Exp_Input[:,0]),2 ))
    Exp_Params = np.zeros((len(Exp_Input[:,0]),2 ))
    
    for i in range(0,len(Exp_Input[:,0])):
        # Predict the Shear Modulus and the Parameter Jm
        yhat = model_Gent_Inverse_ABFits.predict([Exp_Input[i,:].tolist()])
        
        # Re-dimensionalized and Scale Output
        Exp_Params_Scaled[i] = (yhat / ((ExpData[Keys[i]]['radius'])**2  / (25e-6)**2 )  ) 
        Exp_Params[i,0] = 10**( ((Exp_Params_Scaled[i,0]/1 + 0.0 ) * (Scales['maxU'] - Scales['minU'])*1)  + Scales['minU'] )
        # Exp_Params[i,1] = 1*(((Exp_Params_Scaled[i,1]/1 + 0.0 ) * (Scales['maxJ'] - Scales['minJ'])*1)  + Scales['minJ'])
        Exp_Params[i,1] = 10**(((Exp_Params_Scaled[i,1]/1 + 0.0 ) * (Scales['maxJ'] - Scales['minJ'])*1)  + Scales['minJ'])
    
    # Ensure the Model Doesn't try to Predict to Far Out of The Trained 
    # Parameter Space
    Exp_Params_Cl = np.copy(Exp_Params)
    Exp_Params_Cl[Exp_Params_Cl[:, 0]<=10, 0] = 10  
    Exp_Params_Cl[Exp_Params_Cl[:, 1]<=0.005, 1] = 0.005
    Exp_Params_Cl[Exp_Params_Cl[:, 1]>=5 ,1] = 5  
    
    return Exp_Params_Cl
    












def RMSE_Approx(model_Gent_Forward,syn_files,ExpData_Dim,ExpData_Load, Exp_Params, Scales):
    """
    Objective
        Calculate RMSE of Forward Model Fed with Identified Material Parameters
        and Experimental Data    
    
    Parameters
    ----------
    model_Gent_Forward : (keras model)
        Trained weights and architecture of forward neural network.
    syn_files : (list)
        Path to synthetic data files.
    ExpData_Dim : (array)
        Stores spatial dimensions of experimental data.
    ExpData_Load : (array)
        Stores loading data of experimental data.
    Exp_Params : (array)
        Stores identified material parameters of experimental data.
    Scales : (dict)
        Scales for the neural networks.

    Returns
    -------
    Exp_Params : (array)
        Stores identified material parameters of experimental data, and the 
        RMSE fit of the parameters to the experimental data
    PPred : (array)
        The loading curve produced from the identified material parameters.
    """
    Nd = 100
    Ndat = len(ExpData_Dim[:,0])
    
    # Generate Forward Neural Network Features
    x_input = np.array([ExpData_Dim[:,0], ExpData_Dim[:,1], Exp_Params[:,0], Exp_Params[:,1]]).T
    
    # Properly Scale Neural Network Features
    Exp_Input = np.zeros((Ndat,2+2))
    # Scale Spatial Features
    minw = min(x_input[:,0]); maxw = max(x_input[:,0])
    minh = min(x_input[:,1]); maxh = max(x_input[:,1])
    if minw == maxw:  Exp_Input[:,0] = 1
    else:             Exp_Input[:,0] =  ((np.copy(x_input[:,0]) - minw)/(maxw - minw)*1 - 0) * 1
    if minh == maxh:  Exp_Input[:,1] = 1
    else:             Exp_Input[:,1] =  ((np.copy(x_input[:,1]) - minh)/(maxh - minh)*1 - 0) * 1
    # Scale Material Parameter Features
    Exp_Input[:,2] =  ((np.log10(np.copy(x_input[:,2]) ) - Scales['minU'])/(Scales['maxU'] - Scales['minU'])*1 - 0.0) * 1
    Exp_Input[:,3] =  ((np.log10(np.copy(x_input[:,3]) ) - Scales['minJ'])/(Scales['maxJ'] - Scales['minJ'])*1 - 0.0) * 1
    
    
    RMSE = np.zeros(Ndat)
    PAct = np.zeros((Ndat,Nd))
    PPred = np.zeros((Ndat,Nd))
    for i in range(0,len(x_input[:,0])):
        # Foraward Neural Network Prediction
        yhat_train = model_Gent_Forward.predict([Exp_Input[i,:].tolist()])
        
                
        # Resample and Scale Neural Network Prediction
        Ppred = np.zeros(Nd);
        xi = np.linspace(0, 0.5, Nd+1 )[1:]
        ppred =  np.interp(xi, np.linspace(0,0.5,len(yhat_train[0][:])+1)[1:], yhat_train[0][:])[:]
        for k in range(0,Nd):
            Ppred[k] = 10**((ppred[k]*(Scales['maxP'][k] - Scales['minP'][k]) + Scales['minP'][k]))
                  
        # Calculate RMSE of Fit
        RMSE[i] = ma.sqrt(np.square(np.subtract(ExpData_Load[i,:], Ppred)).mean() )
        PAct[i,:] = ExpData_Load[i,:]
        PPred[i,:] = Ppred
    
    # Final Calculated Material Parameters
    Exp_Params = np.append(Exp_Params, np.row_stack(RMSE), axis=1 )

    return Exp_Params , PPred








def CleanPredictions( Exp_Params_1, Exp_Params_2):
    """
    Objective
        Find the most accurate material parameters and combine them.
    Parameters
    ----------
    Exp_Params_1 : (array)
        Stored material parameters 1.
    Exp_Params_2 : (array)
        Stored material parameters 2.

    Returns
    -------
    Exp_Params_12 : (array)
        Combination of stored material parameters 1 and 2.

    """
    
    Exp_Params_12 = np.copy(Exp_Params_1)
    for k in range(0, len(Exp_Params_1[:,0])):
        if Exp_Params_2[k,2] < Exp_Params_1[k,2]:
            Exp_Params_12[k,:] = Exp_Params_2[k,:]

    return Exp_Params_12






def LeastSquaresPrediction(Thresh, syn_files, model_Gent_Forward, Exp_Params_DirInv_GT, ExpData_Dim, ExpData_Load, Scales, Fits_DirInv):
    """
    Objective
        Use the material parameters identified by the direct inverse ML model 
        trained on the Gent material model as an initial guess for the iterative
        least squares ML approach.

    Parameters
    ----------
    Thresh : (float)
        Threshold rmse value for acceptable material parameters identified by
        the direct inverse ML approach.
    syn_files : (string)
        Path to synthetic data files.
    model_Gent_Forward : (keras model)
        Trained weights and architecture of forward NN.
    Exp_Params_DirInv_GT : (array)
        Stored identified material parameters from direct inverse approach trained
        with Gent material model synthetic data.
    ExpData_Dim : (array)
        Stored spatial dimensions of experimental data.
    ExpData_Load : (data)
        Stored loading curves of experimental data.
    Scales : (dict)
        Stores scales of synthetic data.
    Fits_DirInv : (array)
        Loading curves produced from identified material parameters.

    Returns
    -------
    Exp_Params_DirInv_GT_Lsq_GT : (array)
        Stored Material Parameters.
    Fits_DirInv_Lsq_Fix : (array)
        Stored loading curves produced from material parameters.
    """
    
    # Ensure that none of the identified paramters fall too far out of the 
    # trained parameter space
    Exp_Params_DirInv_GT_Lsq_GT = np.copy(Exp_Params_DirInv_GT)
    Exp_Params_DirInv_GT_Lsq_GT[Exp_Params_DirInv_GT_Lsq_GT[:, 0]<=25 ,0] = 25  
    Exp_Params_DirInv_GT_Lsq_GT[Exp_Params_DirInv_GT_Lsq_GT[:, 1]<=0.0025 ,1] = 0.0025  
    Exp_Params_DirInv_GT_Lsq_GT[Exp_Params_DirInv_GT_Lsq_GT[:, 1]>=10 ,1] = 10  

    Nd = 100;  Ndat = len(ExpData_Dim[:,0])
    xi = np.linspace(0, 0.5, Nd+1 )[1:]
    
    # Function for finding fit based on the forward model
    def LSQ_func(inputs, ExpData_Dim, MU_guess, JM_guess, K):
        W = (ExpData_Dim[K,0] - Scales['minw'])/(Scales['maxw'] - Scales['minw'])
        H = (ExpData_Dim[K,1] - Scales['minh'])/(Scales['maxh'] - Scales['minh'])
        # Input Features for Forward NN
        x_input = np.array([W, H, MU_guess, JM_guess])
        yhat_test = model_Gent_Forward.predict([x_input.tolist()])
        
        # Scaling Predicted Loading Curve
        Ppred = np.zeros(Nd);  ppred =  np.interp(xi, np.linspace(0,0.5,len(yhat_test[0][:])+1)[1:], yhat_test[0][:])[:]
        for k in range(0,Nd):
            Ppred[k] = 10**((ppred[k]*(Scales['maxP'][k] - Scales['minP'][k]) + Scales['minP'][k]))
            
        return  Ppred
            
    # Run Iterative Least Squares Model
    Fits_DirInv_Lsq_Fix = np.copy(Fits_DirInv)
    for K in range(0,Ndat):
        if Exp_Params_DirInv_GT_Lsq_GT[K,2] >= Thresh:
            # Load-Displacement Data
            LSQ_Disp = np.linspace(0,0.5,Nd+1)[1:]
            LSQ_Load = ExpData_Load[K,:]
    
            # Mapping function composed of Forward NN
            def LSQ_Func(inputs, mu_guess, Jm_guess):
                W = (ExpData_Dim[K,0] - Scales['minw'])/(Scales['maxw'] - Scales['minw'])
                H = (ExpData_Dim[K,1] - Scales['minh'])/(Scales['maxh'] - Scales['minh'])
                MU_guess = x0[0]  +  (mu_guess - x0[0])*100000
                JM_guess = x0[1]  +  (Jm_guess - x0[1])*100000
                
                # Scaled Forward Neural Networks Features 
                x_input = np.array([W, H, MU_guess, JM_guess])
                # Predict Loading Curve 
                yhat_test = model_Gent_Forward.predict([x_input.tolist()])

                # Resample and Rescale the predicted loading curve
                Ppred = np.zeros(Nd);  ppred =  np.interp(xi, np.linspace(0,0.5,len(yhat_test[0][:])+1)[1:], yhat_test[0][:])[:]
                for k in range(0,Nd):
                    Ppred[k] = 10**((ppred[k]*(Scales['maxP'][k] - Scales['minP'][k]) + Scales['minP'][k]))
                        
                return  Ppred
            
            # Initial Guess for Iterative Least Squares Method
            x0 =  np.copy(Exp_Params_DirInv_GT_Lsq_GT[K,:])
            x0[0] =  (np.log10(np.copy(Exp_Params_DirInv_GT_Lsq_GT[K,0]))-Scales['minU'])/(Scales['maxU'] - Scales['minU'])
            x0[1] =  (np.log10(np.copy(Exp_Params_DirInv_GT_Lsq_GT[K,1]))-Scales['minJ'])/(Scales['maxJ'] - Scales['minJ']) 
            
            # Run Optimizer
            try:
                Lsq_params, pcov_neww = optimize.curve_fit(LSQ_Func, LSQ_Disp,LSQ_Load,p0=(x0[0], x0[1]), bounds = ([-0.25,-0.25],[1.25,1.25]),ftol = 1e-15, xtol = 1e-15,maxfev=200)
            except RuntimeError:
                print("Error - curve_fit failed")
                       
            # Rescale Identified Parameters
            Lsq_Params = np.zeros(2)
            Lsq_Params[0] =  x0[0]  +  (Lsq_params[0] - x0[0])*100000
            Lsq_Params[1] =  x0[1]  +  (Lsq_params[1] - x0[1])*100000
            
            # Determine the loading curve from the identified parameters                
            Fit = LSQ_func(LSQ_Disp, ExpData_Dim, (Lsq_Params[0]), (Lsq_Params[1]), K )
            
            # Rescale Identified Parameters
            Exp_Params_DirInv_GT_Lsq_GT[K,0] = 10**(Lsq_Params[0]* (Scales['maxU'] - Scales['minU'])  + Scales['minU'])
            Exp_Params_DirInv_GT_Lsq_GT[K,1] = 10**(Lsq_Params[1]* (Scales['maxJ'] - Scales['minJ'])  + Scales['minJ']) 
            # Store loading curves and RMSE of the fit and experimental data
            Fits_DirInv_Lsq_Fix[K,:]  =  Fit
            Exp_Params_DirInv_GT_Lsq_GT[K,2] = np.sqrt(np.square(np.subtract(LSQ_Load,Fit)).mean() )

    return Exp_Params_DirInv_GT_Lsq_GT, Fits_DirInv_Lsq_Fix
















def LeastSquaresPrediction_NH(Thresh, syn_files, model_Gent_Forward, Exp_Params_DirInv_GT, ExpData_Dim,ExpData_Load,Scales,Fits_DirInv):
    Exp_Params_DirInv_GT_Cl = np.copy(Exp_Params_DirInv_GT)
    Exp_Params_DirInv_GT_Cl[Exp_Params_DirInv_GT_Cl[:, 0]<=25 ,0] = 25  
    Exp_Params_DirInv_GT_Cl[Exp_Params_DirInv_GT_Cl[:, 1]<=0.0025 ,1] = 0.0025  
    Exp_Params_DirInv_GT_Cl[Exp_Params_DirInv_GT_Cl[:, 1]>=10 ,1] = 10  
    
    
    
    model = model_Gent_Forward # keras.models.load_model('C:\\Users\\pgiol\\Documents\\DrRausch\\NeuralNetworks\\FinalCode\\Trained_NN_Models\\NN_Gent_Forward')

    
    Model_Output = np.loadtxt(syn_files + "\\Model_Output_ExtraSmallJm")
    Model_Input = np.loadtxt(syn_files + "\\Model_Input_ExtraSmallJm")
    
        
    Nx= len(Model_Output[:,0])
    Nsamp= len(Model_Output[0,:])
    FEA_disp = np.linspace(0,0.5,int(Nx+1))[1:]
    Nd = 100
    Ndat = len(ExpData_Dim[:,0])

    FEA_disp = np.linspace(0,0.5,int(Nx+1))[1:]
    FEA_fit_params = np.zeros((Nsamp,2))
    Model_Output_Resampled = np.zeros((Nd, Nsamp))
    for i in range(0,Nsamp):
        FEA_Load = Model_Output[:,i]
        newp, pcov_new = optimize.curve_fit( Fit_model, FEA_disp, FEA_Load, ftol=1e-15, xtol=1e-15, maxfev=800000)
        FEA_fit_params[i,:] = newp
        xi = np.linspace(0, 0.5, Nd+1 )[1:]
        Model_Output_Resampled[:,i] = Fit_model(xi,*newp)
        
    
    minU = 2; maxU = 6
    minJ = min(np.log10(Model_Input[:,3])); maxJ = max(np.log10(Model_Input[:,3]))
            
    # model = keras.models.load_model('C:\\Users\\pgiol\\Documents\\DrRausch\\NeuralNetworks\\FinalCode\\Trained_NN_Models\\NN_Gent_Forward')
    def LSQ_func(inputs, MU_guess, JM_guess):
        x_input = np.array([1,1,MU_guess, JM_guess])
        yhat_test = model.predict([x_input.tolist()])
        # Ppred = np.zeros(Nd)
        # for k in range(0,Nd):
        #     minP = min(np.log10(Model_Output_Resampled[k,:])); maxP = max(np.log10(Model_Output_Resampled[k,:]))
        #     Ppred[k] = 10**((yhat_test[0][k]*(maxP - minP) + minP))

        Ppred = np.zeros(Nd)
        ppred =  np.interp(xi, np.linspace(0,0.5,len(yhat_test[0][:])+1)[1:], yhat_test[0][:])[:]
        for k in range(0,Nd):
            # Ppred[k] = 10**((yhat_train[0][k]*(Scales['maxP'][k] - Scales['minP'][k]) + Scales['minP'][k]))
            Ppred[k] = 10**((ppred[k]*(Scales['maxP'][k] - Scales['minP'][k]) + Scales['minP'][k]))
            # Pact[k] = ((ytrain[i,k]*(Scales['maxP'][k] - Scales['minP'][k]) + Scales['minP'][k]))

        return  Ppred
            
    Exp_Params_DirInv_Lsq_Fix = np.zeros((Ndat,3))
    Exp_Params_DirInv_Lsq_Fix[:,2] = 100
    # Fits_DirInv_Lsq_Fix = np.zeros((Ndat,Nd))
    Fits_DirInv_Lsq_Fix = np.copy(Fits_DirInv)

    for K in range(0,Ndat):
        if Exp_Params_DirInv_GT_Cl[K,2] >= Thresh:
            LSQ_Disp = np.linspace(0,0.5,Nd+1)[1:]
            LSQ_Load = ExpData_Load[K,:]
    
            def LSQ_Func(inputs, mu_guess, Jm_guess):
                W = (ExpData_Dim[K,0] - Scales['minw'])/(Scales['maxw'] - Scales['minw'])
                H = (ExpData_Dim[K,1] - Scales['minh'])/(Scales['maxh'] - Scales['minh'])
                MU_guess = x0[0]  +  (mu_guess - x0[0])*100000
                JM_guess = x0[1]  +  (Jm_guess - x0[1])*100000
 
                x_input = np.array([W, H, MU_guess, JM_guess])
                yhat_test = model.predict([x_input.tolist()])
                    
                Ppred = np.zeros(Nd)
                ppred =  np.interp(xi, np.linspace(0,0.5,len(yhat_test[0][:])+1)[1:], yhat_test[0][:])[:]
                for k in range(0,Nd):
                    Ppred[k] = 10**((ppred[k]*(Scales['maxP'][k] - Scales['minP'][k]) + Scales['minP'][k]))
                    # Pact[k] = ((ytrain[i,k]*(Scales['maxP'][k] - Scales['minP'][k]) + Scales['minP'][k]))       
                        
                return  Ppred
            
            x0 =  np.copy(Exp_Params_DirInv_GT_Cl[K,:])
            x0[0] =  (np.log10(np.copy(Exp_Params_DirInv_GT_Cl[K,0]))-minU)/(maxU - minU)
            x0[1] =  1#(np.log10(np.copy(Exp_Params_DirInv_GT_Cl[K,1]))-minJ)/(maxJ - minJ) 
            
            try:
                Lsq_params, pcov_neww = optimize.curve_fit(LSQ_Func, LSQ_Disp,LSQ_Load,p0=(x0[0], x0[1]), bounds = ([-0.25,1.0],[1.25,1.1]),ftol = 1e-15, xtol = 1e-15,maxfev=200)
            
            except RuntimeError:
                print("Error - curve_fit failed")
                        
            Lsq_Params = np.zeros(2)
            Lsq_Params[0] =  x0[0]  +  (Lsq_params[0] - x0[0])*100000
            Lsq_Params[1] =  x0[1]  +  (Lsq_params[1] - x0[1])*100000
                            
            Fit = LSQ_func(LSQ_Disp, (Lsq_Params[0]), (Lsq_Params[1]) )
            
            
            Exp_Params_DirInv_Lsq_Fix[K,0] = 10**(Lsq_Params[0]* (maxU - minU)  + minU)
            Exp_Params_DirInv_Lsq_Fix[K,1] = 10**(Lsq_Params[1]* (maxJ - minJ)  + minJ) 
            Fits_DirInv_Lsq_Fix[K,:]  =  Fit
            Exp_Params_DirInv_Lsq_Fix[K,2] = np.sqrt(np.square(np.subtract(LSQ_Load,Fit)).mean() )

    for K in range(0,Ndat):
        if Exp_Params_DirInv_GT_Cl[K,2] >= Thresh:
            # print(K)
            Exp_Params_DirInv_GT_Cl[K,0] = Exp_Params_DirInv_Lsq_Fix[K,0]
            Exp_Params_DirInv_GT_Cl[K,1] = Exp_Params_DirInv_Lsq_Fix[K,1]
            Exp_Params_DirInv_GT_Cl[K,2] = Exp_Params_DirInv_Lsq_Fix[K,2]

            
    return Exp_Params_DirInv_GT_Cl, Fits_DirInv_Lsq_Fix


























def GaussNewtonMLPrediction(Thresh, syn_files, model_Gent_Forward, Exp_Params_DirInv_GT, ExpData_Dim, ExpData_Load, Scales, Fits_DirInv_GT):
    """
    Objective
        Use the material parameters identified by the direct inverse ML model 
        trained on the Gent material model as an initial guess for the iterative
        least squares ML approach. The Gauss-Newton Method is used as the optimizer.

    Parameters
    ----------
    Thresh : (float)
        Threshold rmse value for acceptable material parameters identified by
        the direct inverse ML approach.
    syn_files : (string)
        Path to synthetic data files.
    model_Gent_Forward : (keras model)
        Trained weights and architecture of forward NN.
    Exp_Params_DirInv_GT : (array)
        Stored identified material parameters from direct inverse approach trained
        with Gent material model synthetic data.
    ExpData_Dim : (array)
        Stored spatial dimensions of experimental data.
    ExpData_Load : (data)
        Stored loading curves of experimental data.
    Scales : (dict)
        Stores scales of synthetic data.
    Fits_DirInv_GT : (array)
        Loading curves produced from identified material parameters.

    Returns
    -------
    Exp_Params_DirInv_GT_Lsq_GT : (array)
        Stored Material Parameters.
    Fits_DirInv_Lsq_Fix : (array)
        Stored loading curves produced from material parameters.
    """
    
    
    # Ensure that none of the identified paramters fall too far out of the 
    # trained parameter space
    Exp_Params_DirInv_GT_Lsq_GT = np.copy(Exp_Params_DirInv_GT)
    Exp_Params_DirInv_GT_Lsq_GT[Exp_Params_DirInv_GT_Lsq_GT[:, 0]<=25 ,0] = 25  
    Exp_Params_DirInv_GT_Lsq_GT[Exp_Params_DirInv_GT_Lsq_GT[:, 1]<=0.0025 ,1] = 0.0025  
    Exp_Params_DirInv_GT_Lsq_GT[Exp_Params_DirInv_GT_Lsq_GT[:, 1]>=10 ,1] = 10  

    Nd = 100;  Ndat = len(ExpData_Dim[:,0])
    xi = np.linspace(0, 0.5, Nd+1 )[1:]
    

    # Function for finding fit based on the forward model
    def LSQ_func(inputs, guess, GN_params):
        # Input Features for Forward NN
        x_input = np.array([GN_params['W'], GN_params['H'], guess[0], guess[1]])
        yhat_test = model_Gent_Forward.predict([x_input.tolist()])

        # Scaling Predicted Loading Curve
        Ppred = np.zeros(Nd); ppred =  np.interp(xi, np.linspace(0,0.5,len(yhat_test[0][:])+1)[1:], yhat_test[0][:])[:]
        for k in range(0,Nd):
            Ppred[k] = 10**((ppred[k]*(Scales['maxP'][k] - Scales['minP'][k]) + Scales['minP'][k]))
            
        return  Ppred
        
            

    # Calculate MSE
    def mean_squared_error(y_true, y_predicted):
    	cost = np.sum((y_true - y_predicted)**2)/len(y_true)
    	return cost
        
    # Calculate the Jacobian Numerically 
    def Jacobian(Runner, x, Guesses, GN_params, i):
        # Perturbation to calculate numerical differential
        eps_mu = 1
        eps_jm = 1e-4
        Grad = np.zeros(( len(Guesses[0,:]), len(x) ))
        for j in range(len(Guesses[0,:])):
            if j == 0: eps = eps_jm
            elif j == 1: eps = eps_mu 
            t = np.zeros_like(Guesses[i,:]).astype(float)
            t[j] = t[j] + eps
            # Calculate Gradient
            grad = ( LSQ_func(x, Guesses[i,:] + t[:], GN_params) 
                   - LSQ_func(x, Guesses[i,:] - t[:], GN_params))/(2*eps)
            Grad[j, :] = grad
    
        return np.column_stack(Grad)


    # Runs The Iterative Gauss-Newton Method
    def gauss_newton(Guesses, LSQ_Disp, LSQ_Load, GN_params):
    	
        n = float(len(LSQ_Disp))
        costs = []
        weights = []
        previous_cost = 1
        for i in range(0,GN_params['iterations']-1):
            # Calculate Jacobian
            J = Jacobian(LSQ_func, LSQ_Disp, Guesses, GN_params, i)
            
            # Loading Curve of Current Guess
            Ypred = LSQ_func(LSQ_Disp, Guesses[i,:], GN_params ) 
            dy = LSQ_Load - Ypred 
    
            # Calculate new Guess
            Guesses[i+1,:] = Guesses[i,:] + np.linalg.inv(J.T @ J) @ J.T @ dy
            
            # Adjust Guess if it passes a bounds
            for j in range(0, len(Guesses[0,:]) ):
                if Guesses[i+1, j]  < GN_params['bounds'][0,j]:
                    Guesses[i+1, j] = GN_params['bounds'][0,j]
                if Guesses[i+1, j]  > GN_params['bounds'][1,j]:
                    Guesses[i+1, j] = GN_params['bounds'][1,j]

            # Append Stored Cost List        
            current_cost = mean_squared_error(LSQ_Load, Ypred)
            previous_cost = current_cost
            costs.append(current_cost)
            
            # Check for Convergence
            if abs(previous_cost-current_cost) < GN_params['stopping_threshold']:
                break
                
        return Guesses[:i+2,:], costs
        


    Fits_DirInv_Lsq_Fix = np.copy(Fits_DirInv_GT)
    for K in range(0,Ndat):
        if Exp_Params_DirInv_GT_Lsq_GT[K,2] >= Thresh:
            LSQ_Disp = np.linspace(0,0.5,Nd+1)[1:]
            LSQ_Load = ExpData_Load[K,:]
            
            # Initial Guess from Direct Inverse Model
            x0 =  np.copy(Exp_Params_DirInv_GT_Lsq_GT[K,:])
            x0[0] =  (np.log10(np.copy(Exp_Params_DirInv_GT_Lsq_GT[K,0])) - Scales['minU'])/(Scales['maxU'] - Scales['minU'])
            x0[1] =  (np.log10(np.copy(Exp_Params_DirInv_GT_Lsq_GT[K,1])) - Scales['minJ'])/(Scales['maxJ'] - Scales['minJ']) 
            
            # Define and Scale Bounds
            bounds = np.array([[10,  0.0001], [1e6, 1000]])
            bounds[:,0] =  (np.log10(np.copy(bounds[:,0])) - Scales['minU'])/(Scales['maxU'] - Scales['minU'])
            bounds[:,1] =  (np.log10(np.copy(bounds[:,1])) - Scales['minJ'])/(Scales['maxJ'] - Scales['minJ']) 

            # Important Parameters for Gauss Newton Optimizer
            GN_params = {'iterations':1000,
                         'stopping_threshold':1e-16,
                         'bounds': bounds,
                         'W': (ExpData_Dim[K,0]-5)/(40-5),
                         'H': (ExpData_Dim[K,1]-5)/(40-5)}
            
            Guesses = np.zeros((GN_params['iterations'], len(x0)-1 ))
            Guesses[0,:] = x0[:-1]
            
            # Run Gauss Newton Method
            Lsq_Params = np.zeros(2)
            try:
                Guesses, costs = gauss_newton(Guesses, LSQ_Disp, LSQ_Load,  GN_params)
                Lsq_Params[0] =  10**((Guesses[-1,0]*(Scales['maxU'] - Scales['minU'])+Scales['minU']))
                Lsq_Params[1] =  10**((Guesses[-1,1]*(Scales['maxJ'] - Scales['minJ'])+Scales['minJ']))
            except RuntimeError:
                print("Error - curve_fit failed")                            
                Lsq_Params[0] =  10**((Guesses[0,0]*(Scales['maxU'] - Scales['minU'])+Scales['minU']))
                Lsq_Params[1] =  10**((Guesses[0,1]*(Scales['maxJ'] - Scales['minJ'])+Scales['minJ']))
                            
            # Calculate the Loading Curve for the Identified Material Parameter
            Fit = LSQ_func(LSQ_Disp, Guesses[-1,:], GN_params )
        
            # Scale Identified Material Parameters
            Guesses[:,0] =  10**((Guesses[:,0]*(Scales['maxU'] - Scales['minU'])+Scales['minU']))
            Guesses[:,1] =  10**((Guesses[:,1]*(Scales['maxJ'] - Scales['minJ'])+Scales['minJ']))
                        
            # Rescale Identified Parameters
            Exp_Params_DirInv_GT_Lsq_GT[K,0] = Lsq_Params[0]
            Exp_Params_DirInv_GT_Lsq_GT[K,1] = Lsq_Params[1]
            # Store loading curves and RMSE of the fit and experimental data
            Fits_DirInv_Lsq_Fix[K,:]  =  Fit
            Exp_Params_DirInv_GT_Lsq_GT[K,2] = np.sqrt(np.square(np.subtract(LSQ_Load,Fit)).mean() )


            
    return Exp_Params_DirInv_GT_Lsq_GT, Fits_DirInv_Lsq_Fix

















def Plot(ExpData,Fits_DirInv,Exp_Params_DirInv_GT):
    
    import matplotlib.font_manager as font_manager
    title_font = {'fontname':'Times New Roman', 'size':'24', 'color':'black', 'weight':'normal',
                  'verticalalignment':'bottom'} # Bottom vertical alignment for more space
    axis_font = {'fontname':'Times New Roman', 'size':'16'}
    font_path = 'C:\Windows\Fonts\%s.ttf'%('Times New Roman')
    font_prop = font_manager.FontProperties(fname=font_path, size=20)
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = "Times New Roman"
    plt.rcParams["mathtext.default"] = "regular"
    
    Ndat = len(ExpData)
    Nd = 100
    fig = plt.figure(1,figsize=(12,12))
    ax = fig.add_subplot(2, 2, 1)
    plt.plot( ExpData['Averaged']['Indentation'][::5]/ExpData['Averaged']['radius'], ExpData['Averaged']['Load'][::5],'ok',label='Averaged Experimental Data')
    plt.plot(np.linspace(0,0.5,Nd+1)[1:], Fits_DirInv[-1,:],color='navy',linewidth=3,label='ML Prediction: $\mu$=%5.2f, Jm=%5.3f'%(Exp_Params_DirInv_GT[-1,0],Exp_Params_DirInv_GT[-1,1]),alpha=0.5)
    leg = plt.legend(loc='upper left',fontsize='12',handletextpad=0.25)
    leg.get_frame().set_linewidth(0.0)
    leg.get_frame().set_facecolor('none')
    plt.xlabel('$\delta y$/R',**axis_font)
    plt.ylabel('Load (N)',**axis_font)
    
    ax = fig.add_subplot(2, 2, 2)
    plt.hist((Exp_Params_DirInv_GT[:,2]),alpha=0.5)
    plt.xlabel('ML Prediction RMSE',**axis_font)
    plt.ylabel('Counts',**axis_font)


    # cv = abs(Exp_Params_DirInv_GT[:-1,2]-np.min(Exp_Params_DirInv_GT[:-1,2]))/(Thresh-np.min(Exp_Params_DirInv_GT[:-1,2]))
    ax = fig.add_subplot(2, 2, 3)
    plt.semilogy(np.linspace(1,Ndat-1,Ndat-1),Exp_Params_DirInv_GT[:-1,0], 'ok')#c=cv,cmap='coolwarm', s=50)
    plt.semilogy([0,len(Exp_Params_DirInv_GT[:-1,0])], [Exp_Params_DirInv_GT[-1,0],Exp_Params_DirInv_GT[-1,0]],color='navy',linestyle='--',linewidth=3,label='Parameter Identification \n Averaged Experimental Data',alpha=0.5)
    leg = plt.legend(loc='upper right',fontsize='12',handletextpad=0.25)
    leg.get_frame().set_linewidth(0.0)
    leg.get_frame().set_facecolor('none')

    plt.xlabel('Sample',**axis_font)
    plt.ylabel('$\mu$ (Pa)',**axis_font)   
    
    ax = fig.add_subplot(2, 2, 4)
    plt.semilogy(np.linspace(1,Ndat-1,Ndat-1),Exp_Params_DirInv_GT[:-1,1], 'ok')#, c=cv,cmap='coolwarm', s=50)
    plt.semilogy([0,len(Exp_Params_DirInv_GT[:-1,1])], [Exp_Params_DirInv_GT[-1,1],Exp_Params_DirInv_GT[-1,1]],color='navy',linestyle='--',linewidth=3,label='Averaged Experimental Data',alpha=0.5)
    # plt.legend(loc='upper right',fontsize=12)
    plt.xlabel('Sample',**axis_font)
    plt.ylabel('Jm',**axis_font)   





    











def ReadLines(exp_files,DirName,FileName,xx,yy,ncol):
    """
    Objective
        Read through .txt file and find key data

    Parameters
    ----------
    exp_files : (string)
        Path to experimental data files.
    DirName : (string)
        Name of directory files are in.
    FileName : (sting)
        Filename.
    xx : (int)
        x index of sweep.
    yy : (int)
        y index of sweep.
    ncol : (int)
        Number of columns of data.

    Returns
    -------
    Data : (array)
        Extracted data columns.
    lines : (list)
        Raw .txt file.
    nstart : (int)
        Where the data columns begin.
    Ns : (list)
        List of positions of important information.

    """
    lines = []   # Define the name of the files
    with open('%s\\%s\\%s S-1 X-%d Y-%d I-1.txt' %(exp_files,DirName,FileName,xx,yy), 'rt') as file: 
        for line in file: 
            lines.append(line)
            
    Phrase = ["Time (s)", "E[v=","radius","Z-position","Z surface"]; nph = len(Phrase); Ns = []
    for i in range(0,nph):
        Start = Phrase[i]   #Looks through the data file for the phrase "Time (s)"
        ns = 0
        for line in lines: 
          index = 0   
          ns += 1
          while index < len(line): 
            index = line.find(Start, index)
            if index == -1: 
              break        
            Ns = np.append(Ns,int(ns-1))
            index += len(Start) 
            
    nend = len(lines)
    nstart = int(Ns[0])+1
    Data = np.zeros(((nend-nstart),ncol))
    
    for n in range(nstart,nend):
        Data[n-nstart,0] = eval(lines[n].split()[0]) # Pulls out the Time data
        Data[n-nstart,1] = eval(lines[n].split()[1]) # Pulls out the Load data
        Data[n-nstart,2] = eval(lines[n].split()[2]) # Pulls out the Indentation data
        Data[n-nstart,3] = eval(lines[n].split()[3]) # Pulls out the Cantilever data
        Data[n-nstart,4] = eval(lines[n].split()[4]) # Pulls out the Peizo data
        Data[n-nstart,5] = eval(lines[n].split()[5]) # Pulls out the Auxilary data
        
    return Data,lines,nstart,Ns




def FindUnits(lines,Ns):
    """
    Objective
        Determines Units from the .txt file
        Probably could be improved upon...

    Parameters
    ----------
    lines : (list)
        Raw .txt file.
    Ns : (list)
        List of positions of important information.
        
    Returns
    -------
    units : (dict)
        Stores all of the relvant units.

    """
    # Find poisson ratio and radius of indenter
    nu = float(lines[int(Ns[1])][4:8].split()[0])
    radius = float(lines[int(Ns[2])][:].split()[3])
    
    # Find the distance the user manually moved the probe from detected contact
    if lines[int(Ns[3])][12] == 'u': unit_zp = 1e-6
    elif lines[int(Ns[3])][12] == 'n': unit_zp = 1e-9
    if lines[int(Ns[4])][11] == 'u': unit_zs = 1e-6
    elif lines[int(Ns[4])][11] == 'n': unit_zs = 1e-9
    manu_disp = float(lines[int(Ns[4])][:].split()[3])*unit_zs - float(lines[int(Ns[3])][:].split()[2])*unit_zp
    
    # Units for the radius of the indenter
    if lines[int(Ns[2])][12] == 'u': unit_rad = 1e-6
    elif lines[int(Ns[2])][12] == 'n': unit_rad = 1e-9
    
    # Units for the loading indentation curve
    DataUnits = lines[int(Ns[0])].split()
    if DataUnits[3][1] == 'u': unit_load = 1e-6
    if DataUnits[3][1] == 'n': unit_load = 1e-9
    if DataUnits[5][1] == 'u': unit_ind = 1e-6
    if DataUnits[5][1] == 'n': unit_ind = 1e-9
    
    # Youngs modulus predicted by the machine (probably inaccurate :P)
    E_Instrument = float(lines[int(Ns[1])].split()[2])
    
    # manu_disp, radius, nu,unit_rad, unit_load, unit_ind, E_Instrument
    units = {'manu_disp':manu_disp, 'radius':radius, 'nu':nu, 'unit_rad':unit_rad, 'unit_load':unit_load, 'unit_ind':unit_ind, 'E_Instrument':E_Instrument}
    return units




def CleanData(Data,units):
    """
    Objective
        Clean The Data, Removes Data Before Contact And Force Offset (Noise)

    Parameters
    ----------
    Data : (array)
        Extracted data columns.
    units : (dict)
        Stores all of the relvant units.

    Returns
    -------
    Data_cln : (dict)
        Extracted data columns with cropped data for the loading curve and noise
        removed.
    Data_cl : (dict)
        Extracted data columns with cropped data for the loading curve.
    Fnoise : (float)
        Initial load noise.
    skip : (int)
        Informs whether data is junk or not
    """
    
    # Find Starting Point (FixMe: Clean up)
    skip = 0
    mov_avg = np.average(Data[:100,1])
    
    # Calculate Background Noise and produce initial guess for contact
    for n in range(0,len(Data[:,1])-20): 
        if n%100 == 100: 
            mov_avg = np.average(Data[:n,1])
        
        ContThresh = 0.5*abs(mov_avg)
        if ( (abs(Data[n,1]-mov_avg) >= ContThresh)  & (abs(Data[n+20,1]-mov_avg) >= ContThresh)  ) :
            guess_ind_start = n
            break
        if n == int(0.5*len(Data[:,1])): 
            guess_ind_start = 1
            skip = 1
            break       
    
    # Calculate Slope of Loading Curve
    PiezMaxReg = np.copy(Data[:,4])
    PiezMaxReg[PiezMaxReg <= 0.999*max(Data[:,4])] = 0
    PiezDif = np.diff(PiezMaxReg)
    
    # Find Max Indentation
    max_index = np.argmax(PiezDif)
    # Combination of smoothing and finding the curvature of the loading curve
    Smooth_2Der = np.gradient(gaussian_filter1d(np.gradient(Data[:max_index,1]),200))
    
    # Find Change in Curvature (contact)
    Max = argrelmax(gaussian_filter1d(Smooth_2Der,100))[0]
    Max[Max > 1.1*guess_ind_start] = 0
    r = np.sqrt((Max - guess_ind_start)**2)
    ind_start = Max[np.argmin(r)]
    
    # Removes data before indentation
    Data_cl = Data[ind_start:,:]
    Piezo_Canti_ManDisp = Data[ind_start:,4] - Data[ind_start:,3] - units['manu_disp']#*(1e6) #Consider converting all units to standard m, kg, s
    Data_cl[:,2] = Piezo_Canti_ManDisp - Piezo_Canti_ManDisp[0]
    
    # Removes Noise from Exteranal (?) Forces
    Fnoise =  Data[ind_start,1] #np.average(Data[:ind_start)
    Data_cln = np.copy(Data_cl)
    Data_cln[:,1] = Data_cl[:,1] - Fnoise
    
    if max(Data_cln[:,1])*units['unit_load'] < 1e-9: skip = 1 

    return Data_cln,Data_cl,Fnoise, skip




def FindKeyValues(Data_cln,Data_cl,nstart,Fnoise,units):
    """
    Objective
        Find Key Values From Indentation-Load Plots

    Parameters
    ----------
    Data_cln : (dict)
        Extracted data columns with cropped data for the loading curve and noise
        removed.
    Data_cl : (dict)
        Extracted data columns with cropped data for the loading curve.
    nstart : (int)
        Start position of data columns.
    Fnoise : (float)
        Initial Load noise.
    units : (dict)
        Relevant units.

    Returns
    -------
    Fmax : (float)
        Max load.
    deltamax : (float)
        Max indentation.
    max_index : (int)
        Final index of loading cuve.
    deltaf : TYPE
        DESCRIPTION.
    Data_cln_load : (array)
        Fully processed loading curve.

    """
    
    # Finds values useful for computation of material properties
    Fmax = max(Data_cln[:,1])  
    deltamax = max(Data_cln[:,2])
    # max_index = np.argmax(Data_cln[:,1])
    
    Piz = np.copy(Data_cln[:,4]) 
    Piz[Piz>=0.9999*max(Piz)] = max(Piz)
    max_index = np.argmax(Piz[:])
    
    # Final Processed loading curve in SI
    Data_cln_load = Data_cln[:max_index,:]
    Data_cln_load[:,1] = Data_cln[:max_index,1]*units['unit_load']
    Data_cln_load[:,2] = Data_cln[:max_index,2]*units['unit_ind']#/(units['radius']*units['unit_rad'])
    
    c = 0
    for n in range(max_index + nstart,len(Data_cl)):
        if (((Data_cl[n,1]-Fnoise) <= 0)&(c==0)):
            deltaf = Data_cl[n,2]; c=1
    if c == 0: deltaf = 0
    
    return Fmax, deltamax,max_index,deltaf, Data_cln_load 




def AnalyzeExpData(exp_files,DirName,FileName, Xs, Ys, Material_Width, Material_Thick):
    """
    Objective
        Analyze the Raw data file and extract useful information

    Parameters
    ----------
    exp_files : (string)
        Path to experimental data files.
    DirName : (string)
        Name of directory files are in.
    FileName : (sting)
        Filename.
    Xs : (int)
        Sweep size - x.
    Ys : (int)
        Sweep size - y.
    Material_Width : (float)
        Material width in meters.
    Material_Thick : (float)
        Material thickness in meters.

    Returns
    -------
    Exp_Out : (dict)
        Properly formated data derived from the raw .txt files.

    """
    Nd = 100
    ncol = 6
    EMat_Instr = np.zeros((Ys,Xs)) #Stores the Youngs Modulus for every run
    
    Exp_Out = {}
    
    n = 0
    for xx in range(1,Xs+1): # Sweeps through runs
        for yy in range(1,Ys+1):
            
            Data,lines,nstart,Ns = ReadLines(exp_files,DirName,FileName,xx,yy,ncol)
            
            units = FindUnits(lines,Ns)
            
            Data_cln,Data_cl,Fnoise,skip  = CleanData(Data,units)
            
            if skip == 0:
                Fmax, deltamax,max_index,deltaf,  Data_cln_load = FindKeyValues(Data_cln,Data_cl,nstart,Fnoise,units)
                
                newp, pcov_new = optimize.curve_fit(Fit_model, abs(Data_cln_load[:,2]/(units['radius']*units['unit_rad'])), abs(Data_cln_load[:,1]) ,ftol=1e-15, xtol=1e-15, maxfev=800000 )#, bounds= ([0,minb],[1,maxb]) )

                xi = np.linspace(0, np.max(Data_cln_load[:,2]), Nd+1 )[1:]
                Exp_load = newp[0]*xi**newp[1]

                Exp_Out['Sample_%s_%s'%(xx,yy)] = {}
                Exp_Out['Sample_%s_%s'%(xx,yy)]['Height'] = Material_Thick
                Exp_Out['Sample_%s_%s'%(xx,yy)]['Width'] = Material_Width
                Exp_Out['Sample_%s_%s'%(xx,yy)]['radius'] = (units['radius']*units['unit_rad'])
                Exp_Out['Sample_%s_%s'%(xx,yy)]['Load'] = Data_cln_load[:,1]#Exp_load
                Exp_Out['Sample_%s_%s'%(xx,yy)]['Indentation'] = Data_cln_load[:,2]#xi 
            
            else:
                print('Skip')
                EMat_Instr[xx-1,yy-1] = 0
    
            n+=1


    return  Exp_Out










































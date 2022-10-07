


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
    ExpData_u_Hertz : TYPE
        DESCRIPTION.
    ExpData_u_ModHertz : TYPE
        DESCRIPTION.
    ExpData_Load : TYPE
        DESCRIPTION.
    ExpData_Dim : TYPE
        DESCRIPTION.
    RMSE_Hertz : TYPE
        DESCRIPTION.
    RMSE_ModHertz : TYPE
        DESCRIPTION.

    """
    
    Ndat = len(ExpData)
    Nd = 100 
    
    ExpData_Load_Fit = np.zeros((Ndat, Nd))
    ExpData_Load = np.zeros((Ndat, Nd))
    ExpData_Dim = np.zeros((Ndat, 2))
    ExpData_Rad = np.zeros((Ndat))

    for i in range(0,Ndat):
        radius = ExpData[Keys[i]]['radius']
        ExpData_Rad[i] = radius
        w = ExpData[Keys[i]]['Width']/radius
        h = ExpData[Keys[i]]['Height']/radius
        if w > 40: w = 40
        if h > 40: h = 40
        disp = ExpData[Keys[i]]['Indentation']/radius
        load = ExpData[Keys[i]]['Load']
        mxdisp = max(disp)
        
        xi = np.linspace(0, 0.5, Nd+1 )[1:]; xi_clip = np.copy(xi)
        xi_clip[xi_clip  >  mxdisp] = 0
        newp, pcov_new = optimize.curve_fit(Fit_model, abs(disp), load ,ftol=1e-15, xtol=1e-15, maxfev=800000 )
        ExpData_Load_Fit[i,:] = newp[0]*xi**newp[1]
        ExpData_Load[i,:] = np.interp(xi_clip, disp, load)[:]
        ExpData_Dim[i,0] = w
        ExpData_Dim[i,1] = h
        
        
        
    ExpData_Load_Avg = np.average(ExpData_Load_Fit,axis=0)
    ExpData_W_Avg = np.average(ExpData_Dim[:,0])
    ExpData_H_Avg = np.average(ExpData_Dim[:,1])
    ExpData_R_Avg = np.average(ExpData_Rad[:])
    ExpData['Averaged'] = {'radius':ExpData_R_Avg, 'Height':ExpData_H_Avg, 'Width':ExpData_W_Avg, 'Load':ExpData_Load_Avg, 'Indentation': xi*ExpData_R_Avg}
    Keys = list( ExpData.keys() )
    Ndat = len(ExpData)
    
    ExpData_Load = np.append(ExpData_Load, np.array([ExpData_Load_Avg]), axis=0)
    ExpData_Dim = np.append(ExpData_Dim, np.array([[ExpData_W_Avg, ExpData_H_Avg]]), axis=0  )

    ExpData_u_Hertz = np.zeros((Ndat, 1)); ExpData_E_Hertz = np.zeros((Ndat, 1))
    ExpData_u_ModHertz = np.zeros((Ndat, 1)); ExpData_E_ModHertz = np.zeros((Ndat, 1))
    RMSE_Hertz = np.zeros(Ndat)
    RMSE_ModHertz = np.zeros(Ndat)
    for i in range(0,Ndat):
        # newp, pcov_new = optimize.curve_fit(Fit_model, abs(Data_cln_load[:,2]/(units['radius']*units['unit_rad'])), abs(Data_cln_load[:,1]) ,ftol=1e-15, xtol=1e-15, maxfev=800000 )#, bounds= ([0,minb],[1,maxb]) )

        newp_h, pcov_new = optimize.curve_fit(Hertz, abs(disp)*radius, load[:], ftol=1e-15, xtol=1e-15, maxfev=800000 )
        newp_mh, pcov_new = optimize.curve_fit(ModHertz, abs(disp)*radius,  load[:],ftol=1e-15, xtol=1e-15, maxfev=800000 )
        
        nu = 0.4995
        Eff_h = (newp_h*3/4/np.sqrt(25e-6))
        Eff_mh = (newp_mh*3/4/np.sqrt(25e-6))
        
        ExpData_u_Hertz[i] = (Eff_h*(1-nu**2))   /(2+2*nu) #*((radius*unit_rad)**2)/((25e-6)**2)
        ExpData_E_Hertz[i] = Eff_h
        ExpData_u_ModHertz[i] = newp_mh/(2+2*nu) # / ( ((radius*unit_rad)**2)/((25e-6)**2) )
        ExpData_E_ModHertz[i] = Eff_mh
    
        RMSE_Hertz[i] = np.sqrt(np.square( np.subtract(ExpData_Load[i,:], Hertz(xi*radius, newp_h[0])  ) )).mean() 
        RMSE_ModHertz[i] = np.sqrt(np.square( np.subtract(ExpData_Load[i,:], ModHertz(xi*radius, newp_mh[0])  ))).mean() 
        
        
    return ExpData_u_Hertz, ExpData_u_ModHertz, ExpData_Load, ExpData_Dim, RMSE_Hertz, RMSE_ModHertz, ExpData,Keys
    
    












def Prepare_DirInv_ABFits_GT(ExpData,Keys,syn_files):
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
    Model_Output = np.loadtxt(syn_files + "\\Model_Output_ExtraSmallJm")
    Model_Input = np.loadtxt(syn_files + "\\Model_Input_ExtraSmallJm")
    nparams = 2; Nd = 100;  Ndat = len(ExpData)
    Nx= len(Model_Output[:,0])
    Nsamp= len(Model_Output[0,:])
    
    FEA_disp = np.linspace(0,0.5,int(Nx+1))[1:]
    FEA_fit_params = np.zeros((Nsamp,nparams))
    Model_Output_Resampled = np.zeros((Nd, Nsamp))
    for i in range(0,Nsamp):
        FEA_Load = Model_Output[:,i]
        newp, pcov_new = optimize.curve_fit( Fit_model, FEA_disp, FEA_Load, ftol=1e-15, xtol=1e-15, maxfev=800000)
        FEA_fit_params[i,:] = newp
        xi = np.linspace(0, 0.5, Nd+1 )[1:]
        Model_Output_Resampled[:,i] = Fit_model(xi,*newp)
        
    minA = min(np.log10(FEA_fit_params[:,0])); maxA = max(np.log10(FEA_fit_params[:,0]))
    minB = min(FEA_fit_params[:,1]); maxB = max(FEA_fit_params[:,1])
    minU = 2; maxU = 6
    minJ = min(Model_Input[:,3]); maxJ = max(Model_Input[:,3])
    minP = np.zeros(Nd); maxP = np.zeros(Nd)
    for i in range(0,Nd):
        minP[i] = min(np.log10(Model_Output_Resampled[i,:])); maxP[i] = max(np.log10(Model_Output_Resampled[i,:]))
        
    Scales = {'minA':minA, 'maxA':maxA, 'minB':minB, 'maxB':maxB, 'minU':minU, 'maxU':maxU, 'minJ':minJ, 'maxJ':maxJ, 'minP':minP, 'maxP':maxP}        
        

    # =============================================================================
    #  Prepare Experimental Data for Direct Inverse Approach
    # =============================================================================
    Exp_fit_params = np.zeros((Ndat,nparams))
    ExpData_Dim = np.zeros((Ndat, 2))
    for i in range(0,Ndat):
        radius = ExpData[Keys[0]]['radius']
        w = ExpData[Keys[i]]['Width']/radius
        h = ExpData[Keys[i]]['Height']/radius
        if w > 40: w = 40
        if h > 40: h = 40
        disp = ExpData[Keys[i]]['Indentation']/radius
        load = ExpData[Keys[i]]['Load']
        mxdisp = max(disp)
    
        xi = np.linspace(0, 0.5, Nd+1 )[1:]; xi_clip = np.copy(xi)
        xi_clip[xi_clip  >=  mxdisp] = 0
        newp, pcov_new = optimize.curve_fit(Fit_model, abs(disp), load ,
                                            bounds = ([-np.inf,minB],[np.inf,np.inf]),
                                            ftol=1e-15, xtol=1e-15, maxfev=800000 )
        Exp_fit_params[i,:] = newp
        fit = newp[0]*xi_clip**newp[1]
        ExpData_Dim[i,0] = w
        ExpData_Dim[i,1] = h
        
    Exp_Input = np.zeros((Ndat, 4))
    minw = np.min(ExpData_Dim[:,0]);  maxw = np.max(ExpData_Dim[:,0])
    minh = np.min(ExpData_Dim[:,1]);  maxh = np.max(ExpData_Dim[:,1])
    if minw == maxw: Exp_Input[:,0] =  1
    else:  Exp_Input[:,0] =  ((np.copy(ExpData_Dim[:,0]) - minw)/(maxw - minw)*1 - 0) * 1
    if minh == maxh: Exp_Input[:,1] =  1
    else:  Exp_Input[:,1] =  ((np.copy(ExpData_Dim[:,1]) - minh)/(maxh - minh)*1 - 0) * 1
    
    Exp_Input[:,2] =  ((np.log10(np.copy(Exp_fit_params[:,0])) - minA)/(maxA - minA)*1 - 0.0) * 1
    Exp_Input[:,3] =  ((np.copy(Exp_fit_params[:,1]) - minB)/(maxB - minB)*1 - 0.0) * 1
    
    return Exp_Input, Scales












# =============================================================================
# Direct Inverse ML Approach to Parameter Identification
# =============================================================================

    
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
        yhat = model_Gent_Inverse_ABFits.predict([Exp_Input[i,:].tolist()])
        Exp_Params_Scaled[i] = (yhat / ((ExpData[Keys[i]]['radius'])**2  / (25e-6)**2 )  ) 
        Exp_Params[i,0] = 10**( ((Exp_Params_Scaled[i,0]/1 + 0.0 ) * (Scales['maxU'] - Scales['minU'])*1)  + Scales['minU'] )
        Exp_Params[i,1] = 1*(((Exp_Params_Scaled[i,1]/1 + 0.0 ) * (Scales['maxJ'] - Scales['minJ'])*1)  + Scales['minJ'])
    
    
    Exp_Params_Cl = np.copy(Exp_Params)
    Exp_Params_Cl[Exp_Params_Cl[:, 0]<=50, 0] = 50  
    Exp_Params_Cl[Exp_Params_Cl[:, 1]<=0.0025, 1] = 0.0025  
    Exp_Params_Cl[Exp_Params_Cl[:, 1]>=10 ,1] = 10  
    
    return Exp_Params_Cl
    












def RMSE_Approx(model_Gent_Forward,syn_files,ExpData_Dim,ExpData_Load, Exp_Params):
    # Model_Output = np.loadtxt(syn_files + "\\Model_Output_GT_ReducedJm")
    # Model_Input = np.loadtxt(syn_files + "\\Model_Input_GT_ReducedJm")
    Model_Output = np.loadtxt(syn_files + "\\Model_Output_ExtraSmallJm")
    Model_Input = np.loadtxt(syn_files + "\\Model_Input_ExtraSmallJm")
    Nd = 100
    Ndat = len(ExpData_Dim[:,0])

        
    Nx= len(Model_Output[:,0])
    Nsamp= len(Model_Output[0,:])
    FEA_disp = np.linspace(0,0.5,int(Nx+1))[1:]
    
    FEA_disp = np.linspace(0,0.5,int(Nx+1))[1:]
    FEA_fit_params = np.zeros((Nsamp,2))
    Model_Output_Resampled = np.zeros((Nd, Nsamp))
    for i in range(0,Nsamp):
        FEA_Load = Model_Output[:,i]
        newp, pcov_new = optimize.curve_fit( Fit_model, FEA_disp, FEA_Load, ftol=1e-15, xtol=1e-15, maxfev=800000)
        FEA_fit_params[i,:] = newp
        xi = np.linspace(0, 0.5, Nd+1 )[1:]
        Model_Output_Resampled[:,i] = Fit_model(xi,*newp)
    
        
        
    minA = min(np.log10(FEA_fit_params[:,0])); maxA = max(np.log10(FEA_fit_params[:,0]))
    minB = min(FEA_fit_params[:,1]); maxB = max(FEA_fit_params[:,1])
    minU = 2; maxU = 6
    minJ = min(np.log10(Model_Input[:,3])); maxJ = max(np.log10(Model_Input[:,3]))
    minP = np.zeros(Nd); maxP = np.zeros(Nd)
    for i in range(0,Nd):
        minP[i] = min(np.log10(Model_Output_Resampled[i,:])); maxP[i] = max(np.log10(Model_Output_Resampled[i,:]))
        
            
    Scales = {'minA':minA, 'maxA':maxA, 'minB':minB, 'maxB':maxB, 'minU':minU, 'maxU':maxU, 'minJ':minJ, 'maxJ':maxJ, 'minP':minP, 'maxP':maxP}        
            
    
    x_input = np.array([ExpData_Dim[:,0], ExpData_Dim[:,1], Exp_Params[:,0], Exp_Params[:,1]]).T
            
    Exp_Input = np.zeros((Ndat,2+2))
    minw = min(x_input[:,0]); maxw = max(x_input[:,0])
    minh = min(x_input[:,1]); maxh = max(x_input[:,1])
    if minw == maxw:  Exp_Input[:,0] = 1
    else:             Exp_Input[:,0] =  ((np.copy(x_input[:,0]) - minw)/(maxw - minw)*1 - 0) * 1
    if minh == maxh:  Exp_Input[:,1] = 1
    else:             Exp_Input[:,1] =  ((np.copy(x_input[:,1]) - minh)/(maxh - minh)*1 - 0) * 1
    
    
    Exp_Input[:,2] =  ((np.log10(np.copy(x_input[:,2]) ) - Scales['minU'])/(Scales['maxU'] - Scales['minU'])*1 - 0.0) * 1
    Exp_Input[:,3] =  ((np.log10(np.copy(x_input[:,3]) ) - Scales['minJ'])/(Scales['maxJ'] - Scales['minJ'])*1 - 0.0) * 1
    Exp_Input[Exp_Input[:,3]>1,3] = 1
    
    
    
    RMSE = np.zeros(Ndat)
    PAct = np.zeros((Ndat,Nd))
    PPred = np.zeros((Ndat,Nd))
    for i in range(0,len(x_input[:,0])):
        yhat_train = model_Gent_Forward.predict([Exp_Input[i,:].tolist()])
        
        Ppred = np.zeros(Nd)
        Pact = np.zeros(Nd)
        Err = np.zeros(Nd)
        for k in range(0,Nd):
            Ppred[k] = 10**((yhat_train[0][k]*(Scales['maxP'][k] - Scales['minP'][k]) + Scales['minP'][k]))
            # Pact[k] = ((ytrain[i,k]*(Scales['maxP'][k] - Scales['minP'][k]) + Scales['minP'][k]))
                       
        Pact = ExpData_Load[i,:]
        RMSE[i] = ma.sqrt(np.square(np.subtract(Pact,Ppred)).mean() )
                
        PAct[i,:] = Pact
        PPred[i,:] = Ppred
    
    Exp_Params = np.append(Exp_Params, np.row_stack(RMSE), axis=1 )

    return Exp_Params , PPred







def CleanDirectInversePrediction(Thresh, syn_files, model_Gent_Forward, Exp_Params_DirInv_GT, ExpData_Dim,ExpData_Load):
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
        Ppred = np.zeros(Nd)
        for k in range(0,Nd):
            minP = min(np.log10(Model_Output_Resampled[k,:])); maxP = max(np.log10(Model_Output_Resampled[k,:]))
            Ppred[k] = 10**((yhat_test[0][k]*(maxP - minP) + minP))
        return  Ppred
            
    Exp_Params_DirInv_Lsq_Fix = np.zeros((Ndat,3))
    Exp_Params_DirInv_Lsq_Fix[:,2] = 100
    Fits_DirInv_Lsq_Fix = np.zeros((Ndat,Nd))

    for K in range(0,Ndat):
        if Exp_Params_DirInv_GT_Cl[K,2] >= Thresh:
            LSQ_Disp = np.linspace(0,0.5,Nd+1)[1:]
            LSQ_Load = ExpData_Load[K,:]
    
            def LSQ_Func(inputs, mu_guess, Jm_guess):
            
                MU_guess = x0[0]  +  (mu_guess - x0[0])*100000
                JM_guess = x0[1]  +  (Jm_guess - x0[1])*100000
 
                x_input = np.array([1,1,MU_guess, JM_guess])
                yhat_test = model.predict([x_input.tolist()])
                Ppred = np.zeros(Nd)
                for k in range(0,Nd):
                    minP = min(np.log10(Model_Output_Resampled[k,:])); maxP = max(np.log10(Model_Output_Resampled[k,:]))
                    Ppred[k] = 10**((yhat_test[0][k]*(maxP - minP) + minP))
            
                # print(   10**(mu_guess* (maxU - minU)  + minU)   
                #       , 10**(Jm_guess * (maxJ - minJ)  + minJ) )
                return  Ppred
            
            x0 =  np.copy(Exp_Params_DirInv_GT_Cl[K,:])
            x0[0] =  (np.log10(np.copy(Exp_Params_DirInv_GT_Cl[K,0]))-minU)/(maxU - minU)
            x0[1] =  (np.log10(np.copy(Exp_Params_DirInv_GT_Cl[K,1]))-minJ)/(maxJ - minJ) 
            
            Lsq_params, pcov_neww = optimize.curve_fit(LSQ_Func, LSQ_Disp,LSQ_Load,p0=(x0[0], x0[1])
                                                       , bounds = ([-0.25,-0.25],[1.25,1.25]),ftol = 1e-15, xtol = 1e-15)
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

            
    return Exp_Params_DirInv_GT_Cl







def FindScales(syn_files):
    Model_Output = np.loadtxt(syn_files + "\\Model_Output_GT")
    Model_Input = np.loadtxt(syn_files + "\\Model_Input_GT")
        
    Nx= len(Model_Output[:,0])
    Nsamp= len(Model_Output[0,:])
    Nd = 100
    FEA_disp = np.linspace(0,0.5,int(Nx+1))[1:]
    
    FEA_disp = np.linspace(0,0.5,int(Nx+1))[1:]
    FEA_fit_params = np.zeros((Nsamp,2))
    Model_Output_Resampled = np.zeros((Nd, Nsamp))
    for i in range(0,Nsamp):
        FEA_Load = Model_Output[:,i]
        newp, pcov_new = optimize.curve_fit( Fit_model, FEA_disp, FEA_Load, ftol=1e-15, xtol=1e-15, maxfev=800000)
        FEA_fit_params[i,:] = newp
        xi = np.linspace(0, 0.5, Nd+1 )[1:]
        Model_Output_Resampled[:,i] = Fit_model(xi,*newp)
        
    minA = min(np.log10(FEA_fit_params[:,0])); maxA = max(np.log10(FEA_fit_params[:,0]))
    minB = min(FEA_fit_params[:,1]); maxB = max(FEA_fit_params[:,1])
    minU = 2; maxU = 6
    minJ = min(np.log10(Model_Input[:,3])); maxJ = max(np.log10(Model_Input[:,3]))
    minP = np.zeros(Nd); maxP = np.zeros(Nd)
    for i in range(0,Nd):
        minP[i] = min(np.log10(Model_Output_Resampled[i,:])); maxP[i] = max(np.log10(Model_Output_Resampled[i,:]))
        
            
    Scales = {'minA':minA, 'maxA':maxA, 'minB':minB, 'maxB':maxB, 'minU':minU, 'maxU':maxU, 'minJ':minJ, 'maxJ':maxJ, 'minP':minP, 'maxP':maxP}        
            
    return Scales







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
    plt.scatter(np.linspace(1,Ndat-1,Ndat-1),Exp_Params_DirInv_GT[:-1,0], color='k',s=50)#c=cv,cmap='coolwarm', s=50)
    plt.plot([0,len(Exp_Params_DirInv_GT[:-1,0])], [Exp_Params_DirInv_GT[-1,0],Exp_Params_DirInv_GT[-1,0]],color='navy',linestyle='--',linewidth=3,label='Parameter Identification \n Averaged Experimental Data',alpha=0.5)
    leg = plt.legend(loc='upper right',fontsize='12',handletextpad=0.25)
    leg.get_frame().set_linewidth(0.0)
    leg.get_frame().set_facecolor('none')

    plt.xlabel('Sample',**axis_font)
    plt.ylabel('$\mu$ (Pa)',**axis_font)   
    
    ax = fig.add_subplot(2, 2, 4)
    plt.scatter(np.linspace(1,Ndat-1,Ndat-1),Exp_Params_DirInv_GT[:-1,1], color='k',s=50)#, c=cv,cmap='coolwarm', s=50)
    plt.plot([0,len(Exp_Params_DirInv_GT[:-1,1])], [Exp_Params_DirInv_GT[-1,1],Exp_Params_DirInv_GT[-1,1]],color='navy',linestyle='--',linewidth=3,label='Averaged Experimental Data',alpha=0.5)
    # plt.legend(loc='upper right',fontsize=12)
    plt.xlabel('Sample',**axis_font)
    plt.ylabel('Jm',**axis_font)   








































# =============================================================================
# Read Through .txt File And Find Key Data
# =============================================================================
def ReadLines(exp_files,DirName,FileName,xx,yy,ncol):
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

# =============================================================================
# Pull out Values and Units From .txt file
# =============================================================================
def FindUnits(lines,Ns):
    # Find some units
    nu = float(lines[int(Ns[1])][4:8].split()[0])
    radius = float(lines[int(Ns[2])][:].split()[3])
    if lines[int(Ns[3])][12] == 'u': unit_zp = 1e-6
    elif lines[int(Ns[3])][12] == 'n': unit_zp = 1e-9
    if lines[int(Ns[4])][11] == 'u': unit_zs = 1e-6
    elif lines[int(Ns[4])][11] == 'n': unit_zs = 1e-9
    manu_disp = float(lines[int(Ns[4])][:].split()[3])*unit_zs - float(lines[int(Ns[3])][:].split()[2])*unit_zp
    if lines[int(Ns[2])][12] == 'u': unit_rad = 1e-6
    elif lines[int(Ns[2])][12] == 'n': unit_rad = 1e-9
    DataUnits = lines[int(Ns[0])].split()
    if DataUnits[3][1] == 'u': unit_load = 1e-6
    if DataUnits[3][1] == 'n': unit_load = 1e-9
    if DataUnits[5][1] == 'u': unit_ind = 1e-6
    if DataUnits[5][1] == 'n': unit_ind = 1e-9
    E_Instrument = float(lines[int(Ns[1])].split()[2])
    
    # manu_disp, radius, nu,unit_rad, unit_load, unit_ind, E_Instrument
    units = {'manu_disp':manu_disp, 'radius':radius, 'nu':nu, 'unit_rad':unit_rad, 'unit_load':unit_load, 'unit_ind':unit_ind, 'E_Instrument':E_Instrument}
    return units

# =============================================================================
# Clean The Data, Removes Data Before Contact And Force Offset (Noise)
# =============================================================================
def CleanData(Data,units):
    # Find Starting Point (FixMe: Clean up)
    skip = 0
    mov_avg = np.average(Data[:100,1])
    for n in range(0,len(Data[:,1])-20): 
        if n%100 == 100: mov_avg = np.average(Data[:n,1])
        ContThresh = 0.5*abs(mov_avg)
#        ContThresh = 0.1*Fmax
        if ( (abs(Data[n,1]-mov_avg) >= ContThresh)  & (abs(Data[n+20,1]-mov_avg) >= ContThresh)  ) :
            guess_ind_start = n
            break
        if n == int(0.5*len(Data[:,1])): 
            guess_ind_start = 1
            skip = 1
            break       
    
    PiezMaxReg = np.copy(Data[:,4])
    PiezMaxReg[PiezMaxReg<=0.999*max(Data[:,4])] = 0
    PiezDif = np.diff(PiezMaxReg)
    max_index = np.argmax(PiezDif)
    Smooth_2Der = np.gradient(gaussian_filter1d(np.gradient(Data[:max_index,1]),200))
    Max = argrelmax(gaussian_filter1d(Smooth_2Der,100))[0]
    Max[Max>1.1*guess_ind_start] = 0
    r = np.sqrt((Max-guess_ind_start)**2)
    ind_start = Max[np.argmin(r)]
    
    # Removes data before indentation
    Data_cl = Data[ind_start:,:]
    Piezo_Canti_ManDisp = Data[ind_start:,4] - Data[ind_start:,3] - units['manu_disp']#*(1e6) #Consider converting all units to standard m, kg, s
    # Piezo_Canti_ManDisp = Data[ind_start:,2] #- Data[ind_start:,3] - manu_disp#*(1e6) #Consider converting all units to standard m, kg, s
    Data_cl[:,2] = Piezo_Canti_ManDisp - Piezo_Canti_ManDisp[0]
    # Removes Noise from Exteranal (?) Forces
    Fnoise =  Data[ind_start,1] #np.average(Data[:ind_start)
    Data_cln = np.copy(Data_cl)
    Data_cln[:,1] = Data_cl[:,1] - Fnoise
    
    if max(Data_cln[:,1])*units['unit_load'] < 1e-9: skip = 1 

    return Data_cln,Data_cl,Fnoise, skip

# =============================================================================
# Find Key Values From Indentation-Load Plots
# =============================================================================
def FindKeyValues(Data_cln,Data_cl,nstart,Fnoise,units):
    # Finds values useful for computation of material properties
    Fmax = max(Data_cln[:,1])  
    deltamax = max(Data_cln[:,2])
    # max_index = np.argmax(Data_cln[:,1])
    
    Piz = np.copy(Data_cln[:,4]) 
    Piz[Piz>=0.9999*max(Piz)] = max(Piz)
    max_index = np.argmax(Piz[:])
    
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
    
    Nd = 100
    ncol = 6
    EMat_Instr = np.zeros((Ys,Xs)) #Stores the Youngs Modulus for every run
    E_Instr = np.zeros((9,1))
    
    uMat_Exp = np.zeros((Ys,Xs)) #Stores the Youngs Modulus for every run
    EMat_Exp = np.zeros((Ys,Xs)) #Stores the Youngs Modulus for every run
    
    newps = np.zeros((9,2))
    rsq_p = np.zeros((9,1))
    
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
                # xi = np.linspace(0, 0.5, Nd+1 )[1:]
                xi = np.linspace(0, np.max(Data_cln_load[:,2]), Nd+1 )[1:]
                Exp_load = newp[0]*xi**newp[1]
                # xi = np.linspace(0, np.max(Data_cln_load[:,2]), Nd+1 )[1:]
                # Exp_load = np.interp(xi, Data_cln_load[:,2], Data_cln_load[:,1])

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










































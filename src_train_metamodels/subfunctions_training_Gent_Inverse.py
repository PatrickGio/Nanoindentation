import numpy as np
import scipy.optimize as optimize
from sklearn.metrics import r2_score




def Fit_model_1(inputs, a, b):
    x = (inputs[:])
    return  a*(x)**b

def Hertz(inputs, a):
    x = (inputs[:])
    return  a*(x)**1.5
    
def ModHertz(inputs, a):
    x = (inputs[:])
    return  4/3*np.sqrt(25e-6) * x**1.5 * (a/(1-0.4995**2)) * (1 - 0.15*x/(25e-6))






def FindParamFits(Model_Input,Model_Output,Model_Input_ValPred,Model_Output_ValPred):
    Ntrain = len(Model_Output[0,:])
    Ndat_VP = len(Model_Output_ValPred[0,:])
    Nx= len(Model_Output[:,0])
    Nd = 100
    nparams = 2
    
    FEA_disp = np.linspace(0,0.5,int(Nx+1))[1:]
    FEA_fit_params = np.zeros((Ntrain,nparams))
    Model_Output_Resampled = np.zeros((Nd, Ntrain))
    for i in range(0,Ntrain):
        FEA_Load = Model_Output[:,i]
        newp, pcov_new = optimize.curve_fit( Fit_model_1, FEA_disp, FEA_Load, ftol=1e-15, xtol=1e-15, maxfev=800000)
        FEA_fit_params[i,:] = newp
    
        xi = np.linspace(0, 0.5, Nd+1 )[1:]
        Model_Output_Resampled[:,i] = Fit_model_1(xi,*newp)
    
                
    FEA_disp = np.linspace(0,0.5,int(Nx+1))[1:]
    FEA_fit_params_VP = np.zeros((Ndat_VP,nparams))
    Model_Output_Resampled_ValPred = np.zeros((Nd, Ndat_VP))
    u_Hertz = np.zeros(Ndat_VP)
    u_ModHertz = np.zeros(Ndat_VP)
    for i in range(0,Ndat_VP):
        FEA_Load = Model_Output_ValPred[:,i]
        newp, pcov_new = optimize.curve_fit( Fit_model_1, FEA_disp, FEA_Load, ftol=1e-15, xtol=1e-15, maxfev=800000)
        FEA_fit_params_VP[i,:] = newp
    
        xi = np.linspace(0, 0.5, Nd+1 )[1:]
        Model_Output_Resampled_ValPred[:,i] = Fit_model_1(xi,*newp)
        
        newp_h, pcov_new = optimize.curve_fit(Hertz, FEA_disp*(25e-6), FEA_Load, ftol=1e-15, xtol=1e-15, maxfev=800000 )
        newp_mh, pcov_new = optimize.curve_fit(ModHertz,FEA_disp*(25e-6), FEA_Load ,ftol=1e-15, xtol=1e-15, maxfev=800000 )
        nu = 0.4995
        Eff_h = (newp_h*3/4/np.sqrt(25e-6))
        Eff_mh = (newp_mh*3/4/np.sqrt(25e-6))
        u_Hertz[i] =    (Eff_h*(1-nu**2))   /(2+2*nu) 
        u_ModHertz[i] = newp_mh/(2+2*nu)     
    
        
    R2_H = r2_score(Model_Input_ValPred[:,2], u_Hertz) 
    R2_MH = r2_score(Model_Input_ValPred[:,2], u_ModHertz) 

    return FEA_fit_params, FEA_fit_params_VP, Model_Output_Resampled, Model_Output_Resampled_ValPred, u_Hertz, u_ModHertz, R2_H, R2_MH







def Scale_Input_Output(FEA_fit_params, FEA_fit_params_VP, Model_Input, Model_Input_ValPred, Model_Output, Model_Output_ValPred):
    Ntrain = len(Model_Output[0,:])
    Ndat_VP = len(Model_Output_ValPred[0,:])
    
    Meta_Input = np.zeros((Ntrain,4))
    minw = min(Model_Input[:,0]); maxw = max(Model_Input[:,0])
    minh = min(Model_Input[:,1]); maxh = max(Model_Input[:,1])
    
    if (maxw - minw) != 0:
        Meta_Input[:,0] =   ((np.copy(Model_Input[:,0]) - minw)/(maxw - minw)*1 - 0.0) * 1
        Meta_Input[:,1] =  ((np.copy(Model_Input[:,1]) - minh)/(maxh - minh)*1 - 0.0) * 1
    else:
        Meta_Input[:,0] =   1
        Meta_Input[:,1] =  1
    
    minA = min(np.log10(FEA_fit_params[:,0])); maxA = max(np.log10(FEA_fit_params[:,0]))
    minB = min(FEA_fit_params[:,1]); maxB = max(FEA_fit_params[:,1])
    
    Meta_Input[:,2] =  ((np.log10(np.copy(FEA_fit_params[:,0])) - minA)/(maxA - minA)*1 - 0.0) * 1
    Meta_Input[:,3] =  ((np.copy(FEA_fit_params[:,1]) - minB)/(maxB - minB)*1 - 0.0) * 1
    
    
    Meta_Output = np.zeros((Ntrain,2))#4))
    minU = 2; maxU = 6
    minJ = min(Model_Input[:,3]); maxJ = max(Model_Input[:,3])
    Meta_Output[:,0] =  ((np.log10(np.copy(Model_Input[:,2]) ) - minU)/(maxU - minU)*1 - 0.0) * 1
    Meta_Output[:,1] =   ((np.copy(Model_Input[:,3]) - minJ)/(maxJ - minJ)*1 - 0.0) * 1
    
    
         
    Meta_Input_VP = np.zeros((Ndat_VP,4))        
    if (maxw - minw) != 0:
        Meta_Input_VP[:,0] =   ((np.copy(Model_Input_ValPred[:,0]) - minw)/(maxw - minw)*1 - 0.0) * 1
        Meta_Input_VP[:,1] =  ((np.copy(Model_Input_ValPred[:,1]) - minh)/(maxh - minh)*1 - 0.0) * 1
    else:
        Meta_Input_VP[:,0] =   1
        Meta_Input_VP[:,1] =  1
    
    Meta_Input_VP[:,2] =  ((np.log10(np.copy(FEA_fit_params_VP[:,0])) - minA)/(maxA - minA)*1 - 0.0) * 1
    Meta_Input_VP[:,3] =  ((np.copy(FEA_fit_params_VP[:,1]) - minB)/(maxB - minB)*1 - 0.0) * 1
    
    
    
    Meta_Output_VP = np.zeros((Ndat_VP,2))#4))
    minU = 2; maxU = 6
    minJ = min(Model_Input[:,3]); maxJ = max(Model_Input[:,3])
    Meta_Output_VP[:,0] =  ((np.log10(np.copy(Model_Input_ValPred[:,2]) ) - minU)/(maxU - minU)*1 - 0.0) * 1
    Meta_Output_VP[:,1] =   ((np.copy(Model_Input_ValPred[:,3]) - minJ)/(maxJ - minJ)*1 - 0.0) * 1
    
    
    # =============================================================================
    # Define Training, Validation and Testing Data Sets
    # =============================================================================
    xtrain = Meta_Input[:Ntrain,:]
    ytrain = Meta_Output[:Ntrain,:]
    xtest = Meta_Input_VP[:1250,:]
    ytest =  Meta_Output_VP[:1250,:]
    xpred =  Meta_Input_VP[1250:,:]
    ypred = Meta_Output_VP[1250:,:]

    # Ntest = len(xtest[:,1])
    # Ltest = np.random.uniform(int(0.05/0.5*Nd),Nd+2,Ntest)
    # # Ltest = np.random.uniform(3,50,Ntest) #Nd+2,Ntest)
    # for i in range(0,Ntest-1):
    #     xtest[i,round(Ltest[i]):] = 0

    # Ntrain = len(xtrain[:,1])
    # Ltrain = np.random.uniform(3,Nd+2,Ntrain)
    # for i in range(0,Ntrain-1):
    #     xtrain[i,round(Ltrain[i]):] = 0

    
    return xtrain, ytrain, xtest, ytest, xpred, ypred
    























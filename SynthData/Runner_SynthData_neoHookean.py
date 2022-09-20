"""
Objective
    Pipeline to run indentation simulations with a neo-Hookean material model.
    Cubit Coreform is used for the mesher, while FEBio is used for its nonlinear
    mixed finite element solver.
    
Author
    Patrick Giolando
    BME PhD at The University of Texas at Austin
    pgioland@utexas.edu


Parameters
----------
Radius:
    radius of indenter
mu:
    shear modulus
mu_kap_rat:
    ratio of bulk modulus and shear modulus
Wx,Wz:
    width of sample
Wy:
    thickness of sample

Sweep:
      No:        Runs a single simulation 
      Yes:       Runs a sweep of simulations based on the array Inputs
      Yes_Load:  Loads, Runs, and Saves a sweep, used for larger runs

Load_Mesh:
      Yes:       Loads Meshes default is the 10,000 for sweep and 20x20x20R 
                 cube for single run
      No:        Runs the mesher to generate new meshes

"""



# =============================================================================
# import libraries and define paths
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time
# from pyDOE import *
# from smt.sampling_methods import LHS

current_dir = os.getcwd()
func_files = current_dir[:-9] + 'Functions'
sys.path.append(func_files)

from NH_SynthData_Functions import *

mesher_filename = current_dir + '\\Mesher_SynthData_NH\\Journal.jou'
mesh_folder = current_dir + '\\Meshes\\NeoHookean_10000'
mesh_filename = current_dir + '\\Mesher_SynthData_NH\\Mesh.inp'

solver_filename = current_dir + '\\Solver_SynthData_NH\\jobs\\FEBioModel_1.feb'

cub_dir = 'C:/Program Files/Coreform Cubit 2022.4/bin/coreform_cubit.exe'
feb_dir = 'C:/Program Files/FEBioStudio/febio/febio3.exe'
sys.path.append("C:/Program Files/FEBioStudio/febio")






# =============================================================================
# Toggles 
# =============================================================================

Sweep = 'No' 
# Sweep = 'Yes'
# Sweep = 'Yes_Load'

# Load_Mesh = 'No'
Load_Mesh = 'Yes'




# =============================================================================
# User Defined Parameters
# =============================================================================

Radius = 25e-6
mu_kap_rat = 1000

# Run a single simulation
if Sweep == 'No':
    mu =  1000
    
    Wx = Wz = 10
    Wy = 10
    Ninp = 1
    n = 0
    
# Runs a sweep of simulations based on the array Inputs
elif Sweep == 'Yes':  
    # W, H, µ
    lims = np.array([[5,  40],
                      [5, 40],
                        [1e2, 1e6]])
    Param_Sample = LHS(xlimits = lims)
    Nsamples = 1000
    x = Param_Sample(Nsamples)
    Inputs = np.copy(x)
    Inputs[:,0] = np.round(Inputs[:,0],1)
    Inputs[:,1] = np.round(Inputs[:,1],1)
    Inputs[:,2] = np.round(Inputs[:,2],-2)
    Ninp = len(Inputs[:,0])
    
#  Loads, Runs, and Saves a sweep, used for larger runs
elif Sweep == 'Yes_Load':
    if os.exists(current_dir+'Inputs_NH'):
        Inputs = np.loadtxt(current_dir+'Inputs_NH')
        n = int(np.loadtxt(current_dir+'n_NH'))
        Ninp = len(Inputs[:,0])
    else:
        lims = np.array([[5,  40],
                         [5, 40],
                         [1e2, 1e6]])
        Param_Sample = LHS(xlimits = lims)
        Nsamples = 1000
        x = Param_Sample(Nsamples)
        Inputs = np.copy(x)
        Inputs[:,0] = np.round(Inputs[:,0],1)
        Inputs[:,1] = np.round(Inputs[:,1],1)
        Inputs[:,2] = np.round(Inputs[:,2],-2)
        Ninp = len(Inputs[:,0])
        n = 0




# Some additional parameters that can be manipulated, but I suggest 
# leaving them as is
Rw = 1
aspect = int(1)
Rh = aspect * Rw
rf = 2
overlap = (0.0)

deltamax = -0.5

ElementType = 8

hc_ref = Rw
hc = Rw         
hf = hc/(3**rf)
hier = rf


# nondimensionalize parameters
kap = mu_kap_rat*mu
E = 9*kap*(mu)/(3*kap+(mu))
nu = (3*kap-2*mu)/(6*kap+2*mu)
kap_nondim = kap*Radius**2
mu_nondim = (mu*2)*Radius**2





# =============================================================================
#  Build Sample Geometry and Mesh
# =============================================================================
def Mesher(params): 
    
    # Modify Cubit Journal file 
    mesher_working_file = ModifyMesh(params)
    
    # Generates Mesh .inp file 
    RunMesher(cub_dir, [ mesher_working_file],params)
    
    time.sleep(1)
    os.system('taskkill /f /im coreform_cubit.exe')

    
# =============================================================================
#  Simulation of Indentation
# =============================================================================
def Runner(params):

    start = time.process_time()
    # Load Old .feb File and Extract Data
    RunFile, NsR,NsM = ReadRunner(solver_filename)
    
    # Load New Mesh and Extract Data
    mesh, Nodes, Elems,  BCx, BCy, BCz, BCs = LoadMesh(params, NsR)
    
    end = time.process_time()
    print('Meshtime: ', (end-start)/60, '(min)')

    start = time.process_time()
    # Use New Mesh Data to Modify .feb File
    working_solver_filename = ModifyRunner(RunFile,NsR,Nodes,Elems, BCx, BCy, BCz, BCs,NsM,params)
    
    # Run Solver with Updated .feb File
    RunSolver(feb_dir, [ working_solver_filename], solver_filename)
    
    end = time.process_time()

    # Load Data From Model
    Data = LoadData(params)
    print('Runtime: ', (end-start)/60, '(min)')
    
    return Data





# =============================================================================
# Load or Initialize 
# =============================================================================
if Sweep == 'Yes':
    Model_Input = np.zeros((Ninp,len(Inputs[0,:])))
    Model_Output = np.zeros((20,Ninp))

elif Sweep =='Yes_Load':
    if os.exists('Model_Input_NH_%s'%(Ninp)):  
        Model_Input = np.loadtxt('Model_Input_NH_%s'%(Ninp))
        Model_Output = np.loadtxt('Model_Output_NH_%s'%(Ninp))
    else:
        Model_Input = np.zeros((Ninp,len(Inputs[0,:])))
        Model_Output = np.zeros((20,Ninp))
        
        
        
        
        
print('# =======================================')
print('# - - - - - Progress Report - - -  - - - ')
print('# =======================================')   
Start = time.time()
if Sweep == 'Yes':
    for n in range(0,Ninp):
        
        Start = time.time()
        print('-----------------------\n', 'Run: %i \n Wx: %i \t Wy: %i \t µ: %i\n' %(n,Inputs[n,0],Inputs[n,1],Inputs[n,2]),'-----------------------')
        
        # Update params
        params = {'Wx':Inputs[n,0], 'Wy':Inputs[n,1], 'Wz': Inputs[n,0], 'Rw':Rw, 'Rh':Rh, 'hf':hf, 'rf':rf, 'hc':hc, 'hier':hier,
                  'overlap':overlap, 'deltamax':deltamax, 'kap_nondim':(mu_kap_rat*Inputs[n,2])*Radius**2, 'mu_nondim':(Inputs[n,2]*2)*Radius**2, 
                  'ElementType':ElementType,'n':n, 'mesh_filename':mesh_filename, 'mesher_filename':mesher_filename, 'solver_filename':solver_filename}

        # Run or Load Mesh
        if Load_Mesh == 'Yes':
            mesh_filename = current_dir+'\\Meshes\\NeoHookean_10000\\Mesh_%s.inp'%(n)
            params["mesh_filename"] = mesh_filename
        else:
            Mesher(params)

        # Run Simulation
        Data = Runner(params)
        
        # Store Data
        locals()["Data_%s" %(n)] = Data
        Model_Output[:,n] = Data[:,1]
        Model_Input[n,:] = Inputs[n,:]
        
        # Walltime
        End = time.time()
        print('WallTime: ', (End-Start)/60, '(min)')
        
        
        
elif Sweep == 'Yes_Load':
    while n <= Ninp-1:

        Start = time.time()
        print('-----------------------\n', 'Run: %i \n Wx: %i \t Wy: %i \t µ: %i\n' %(n,Inputs[n,0],Inputs[n,1],Inputs[n,2]),'-----------------------')

        # Update params
        params = {'Wx':Inputs[n,0], 'Wy':Inputs[n,1], 'Wz': Inputs[n,0], 'Rw':Rw, 'Rh':Rh, 'hf':hf, 'rf':rf, 'hc':hc, 'hier':hier,
                  'overlap':overlap, 'deltamax':deltamax, 'kap_nondim':(mu_kap_rat*Inputs[n,2])*Radius**2, 'mu_nondim':(Inputs[n,2]*2)*Radius**2, 
                  'ElementType':ElementType,'n':n, 'mesh_filename':mesh_filename, 'mesher_filename':mesher_filename, 'solver_filename':solver_filename}

        # Run or Load Mesh
        if Load_Mesh == 'Yes':
            mesh_filename = current_dir+'\\Meshes\\NeoHookean_10000\\Mesh_%s.inp'%(n)
            params["mesh_filename"] = mesh_filename
        else:
            Mesher(params)

        # Run Simulation
        Data = Runner(params)
        
        # Store Data
        locals()["Data_%s" %(n)] = Data
        Model_Output[:,n] = Data[:,1]
        Model_Input[n,:] = Inputs[n,:]
        
        # Walltime
        End = time.time()
        print('WallTime: ', (End-Start), '(s)')
        
        # Save Data
        np.savetxt('Model_Input_NH_%s'%(Ninp), Model_Input)
        np.savetxt('Model_Output_NH_%s'%(Ninp), Model_Output)
        np.savetxt('n_NH', np.array([n]))

        n+=1



else:
    params = {'Wx':Wx, 'Wy':Wy, 'Wz':Wz, 'Rw':Rw, 'Rh':Rh, 'hf':hf, 'rf':rf, 'hc':hc, 'hier':hier,
              'overlap':overlap, 'deltamax':deltamax, 'kap_nondim':kap_nondim, 'mu_nondim':mu_nondim, 'ElementType':ElementType,
              'n':n, 'mesh_filename':mesh_filename, 'mesher_filename':mesher_filename, 'solver_filename':solver_filename}

    # Run or Load Mesh
    if Load_Mesh == 'Yes':
        mesh_filename = current_dir+'\\Meshes\\Mesh_20R_Cube.inp'
        params["mesh_filename"] = mesh_filename
    else:
        Mesher(params)
        
    # Run Simulation
    Data = Runner(params)




End = time.time()
print('Number of Runs: %i in %5.2f (min)' %(Ninp, (End-Start)/60))






# =============================================================================
# Plotting 
# =============================================================================
def Save():
    np.savetxt('Model_Input_544ish', Model_Input)
    np.savetxt('Model_Output_544ish', Model_Output)


def Plot():
    plt.figure(1)
    plt.plot(Data[:,0],Data[:,1],'o-',label='hf = %1.2f' %(hf))
    plt.xlabel('$\delta (nm)$')
    plt.ylabel('Load ($\mu$N)')
    plt.legend(loc='best')
    plt.show()
    

# Plot()

def PlotHW():
    fig = plt.figure(11,figsize=(9,6))
    ax = plt.axes(projection='3d')
    ax.plot_wireframe(W,H,MaxLoad, color='black')
    ax.scatter(np.reshape(W,int(Nh*Nw)),np.reshape(H,int(Nh*Nw)),np.reshape(MaxLoad,int(Nh*Nw)),color='k',s=30 )
    plt.ylabel('Height')
    plt.xlabel('Width')
    plt.show()
    


























































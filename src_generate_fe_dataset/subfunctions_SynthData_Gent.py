import numpy as np
import os
import re
import subprocess
import time
from os.path import exists

def ModifyMesh(params):
    """
    Objective:
        Modify the Cubit Coreform journal file which acts as an input for the 
        software to create both the geometry and mesh.
    
    Parameters
    ----------
    params : dictionary
        A dictionary that stores parameters that define the simulation as well
        as a few paths.

    Returns
    -------
    working_file : string
        Path of the updated journal file.

    """
    print('Editing Mesher')

    # To model samples with a spatial resolution greater than the radius of the 
    # indenter, which the coarse mesh is set to, the mesh is padded at the 
    # farthest sides with elements that are less than one radius wide.
    Wx_hc = int(np.floor(params['Wx'])); Wz_hc = int(np.floor(params['Wz'])); Wy_hc = int(np.floor(params['Wy']))
    Wx_res = params['Wx']-Wx_hc;  Wz_res = params['Wz']-Wz_hc;  Wy_res = params['Wy']-Wy_hc

    if ((Wx_res == 0)&(Wy_res == 0)): # Both width and height is evenly divisible by the radius
        mesher_filename_condition = ('%s%s'%(params['mesh_filename'][:-8],'Journal.JOU') )

    elif ((Wx_res == 0)&(Wy_res != 0)): # Only the width is evenly divisible by the radius
        mesher_filename_condition = ('%s%s'%(params['mesh_filename'][:-8],'Journal_Y.JOU') )

    elif ((Wx_res != 0)&(Wy_res == 0)): # Only the height is evenly divisible by the radius
        mesher_filename_condition = ('%s%s'%(params['mesh_filename'][:-8],'Journal_X.JOU') )

    elif ((Wx_res != 0)&(Wy_res != 0)): # Neither width or height is evenly divisible by the radius
        mesher_filename_condition = ('%s%s'%(params['mesh_filename'][:-8],'Journal_X_Y.JOU') )

    lines = []   # Opens the mesher file in to a list
    with open('%s' %(mesher_filename_condition), 'rt') as file: 
        for line in file: 
            lines.append(line)
                
    # Finds the location of these phrases in the list and stores the positions in Ns_...
    Phrase = ["brick", "move", "size","refine","element type","export"]; nph = len(Phrase); 
    Ns_b = []; Ns_m = []; Ns_s = []; Ns_r = []; Ns_e = []; Ns_ex = []
    for i in range(0,nph):
        Start = Phrase[i]   #Looks through the data file for the phrases
        ns = 0
        for line in lines: 
          index = 0   
          ns += 1
          while index < len(line): 
            index = line.find(Start, index)
            if index == -1: 
              break        
            if i == 0:   Ns_b = np.append(Ns_b,int(ns-1))
            if i == 1:   Ns_m = np.append(Ns_m,int(ns-1))
            if i == 2:   Ns_s = np.append(Ns_s,int(ns-1))
            if i == 3:   Ns_r = np.append(Ns_r,int(ns-1))
            if i == 4:   Ns_e = np.append(Ns_e,int(ns-1))
            if i == 5:   Ns_ex = np.append(Ns_ex,int(ns-1))
            index += len(Start) 

    # Converts the list into an array structure and then modifies the array
    if ((Wx_res != 0)&(Wy_res != 0)):
        arr = np.array(lines)
        arr[int(Ns_b[0])] = 'brick x %s y %s z %s\n' %(params['Wx'],Wy_hc,Wz_hc)
        arr[int(Ns_m[0])] = 'move Volume 1 x %s y %s z %s include_merged' %(params['Wx']/2,-1-Wy_hc/2+params['overlap'],Wz_hc/2)
        arr[int(Ns_b[1])] = 'brick x %s y %s z %s\n' %(Wx_hc,Wy_hc,Wz_hc)
        arr[int(Ns_m[1])] = 'move Volume 2 x %s y %s z %s include_merged' %(Wx_hc/2,-1-Wy_hc/2+params['overlap'],Wz_hc/2)
    
        arr[int(Ns_b[2])] = 'brick x %s y %s z %s\n' %(Wx_hc,params['Wy'],Wz_hc)
        arr[int(Ns_m[2])] = 'move Volume 3 x %s y %s z %s include_merged' %(Wx_hc/2,-1-params['Wy']/2+params['overlap'],Wz_hc/2)
        arr[int(Ns_b[3])] = 'brick x %s y %s z %s\n' %(Wx_hc,Wy_hc,Wz_hc)
        arr[int(Ns_m[3])] = 'move Volume 4 x %s y %s z %s include_merged' %(Wx_hc/2,-1-Wy_hc/2+params['overlap'],Wz_hc/2)
    
        arr[int(Ns_b[4])] = 'brick x %s y %s z %s\n' %(Wx_hc,Wy_hc,params['Wz'])
        arr[int(Ns_m[4])] = 'move Volume 5 x %s y %s z %s include_merged' %(Wx_hc/2,-1-Wy_hc/2+params['overlap'],params['Wz']/2)
        arr[int(Ns_b[5])] = 'brick x %s y %s z %s\n' %(Wx_hc,Wy_hc,Wz_hc)
        arr[int(Ns_m[5])] = 'move Volume 6 x %s y %s z %s include_merged' %(Wx_hc/2,-1-Wy_hc/2+params['overlap'],Wz_hc/2)
    
                
        arr[int(Ns_b[6])] = 'brick x %s y %s z %s\n' %(Wx_hc,Wy_res,Wz_res)
        arr[int(Ns_m[6])] = 'move Volume 7 x %s y %s z %s include_merged' %(Wx_hc/2,-1-Wy_hc-Wy_res/2+params['overlap'],Wz_hc+Wz_res/2)
        arr[int(Ns_b[7])] = 'brick x %s y %s z %s\n' %(Wx_res,Wy_hc,Wz_res)
        arr[int(Ns_m[7])] = 'move Volume 8 x %s y %s z %s include_merged' %(Wx_hc+Wx_res/2,-1-Wy_hc/2+params['overlap'],Wz_hc+Wz_res/2)
        arr[int(Ns_b[8])] = 'brick x %s y %s z %s\n' %(Wx_res,Wy_res,Wz_hc)
        arr[int(Ns_m[8])] = 'move Volume 9 x %s y %s z %s include_merged' %(Wx_hc+Wx_res/2,-1-Wy_hc-Wy_res/2+params['overlap'],Wz_hc/2)
    
        arr[int(Ns_b[9])] = 'brick x %s y %s z %s\n' %(Wx_res,Wy_res,Wz_res)
        arr[int(Ns_m[9])] = 'move Volume 10 x %s y %s z %s include_merged' %(Wx_hc+Wx_res/2,-1-Wy_hc-Wy_res/2+params['overlap'],Wz_hc+Wz_res/2)
                
        arr[int(Ns_b[10])] = 'brick x %s y %s z %s\n' %(Wx_hc,Wy_hc,Wz_hc)
        arr[int(Ns_m[10])] = 'move Volume 11 x %s y %s z %s include_merged' %(Wx_hc/2,-1-Wy_hc/2+params['overlap'],Wz_hc/2)
                
        arr[int(Ns_b[11])] = 'brick x %s y %s z %s\n' %(params['Rw'],params['Rw'],params['Rw'])
        arr[int(Ns_m[11])] = 'move Volume 12 x %s y %s z %s include_merged' %(params['Rw']/2,-1-params['Rw']/2+params['overlap'],params['Rw']/2)
                    
        arr[int(Ns_s[0])] = 'volume 12 size %s' %(params['hf']) 
        arr[int(Ns_s[1])] = 'volume 11 size %s' %(params['hc']) 
        
        arr[int(Ns_s[2])] = 'volume 7 size %s' %(params['hc']) 
        arr[int(Ns_s[3])] = 'volume 8 size %s' %(params['hc']) 
        arr[int(Ns_s[4])] = 'volume 9 size %s' %(params['hc']) 
    
        arr[int(Ns_s[5])] = 'volume 10 size %s' %(np.max([Wx_res,Wy_res])) 
    
        arr[int(Ns_s[6])] = 'volume 1 size %s' %(params['hc']) 
        arr[int(Ns_s[7])] = 'volume 3 size %s' %(params['hc']) 
        arr[int(Ns_s[8])] = 'volume 5 size %s' %(params['hc']) 
    
        arr[int(Ns_r[0])] = 'refine surface 52,53,54 numsplit %s bias 1.0 depth 0' %(params['rf']) 
        
        arr[int(Ns_e[0])] = 'block 1 element type HEX%s' %(params['ElementType']) 


        arr[int(Ns_ex[0])] = 'export abaqus "%s"  dimension 3  overwrite  everything' %(params['mesh_filename']) 
        # arr[int(Ns_ex[1])] = 'export abaqus "%sMesh_%s.inp"  dimension 3  overwrite  everything' %(mesh_filename[:-8],n) 


        
    elif ((Wx_res == 0)&(Wy_res != 0)):

        arr = np.array(lines)
        arr[int(Ns_b[0])] = 'brick x %s y %s z %s\n' %(Wx_hc,params['Wy'],Wz_hc)
        arr[int(Ns_m[0])] = 'move Volume 1 x %s y %s z %s include_merged' %(Wx_hc/2,-1-params['Wy']/2+params['overlap'],Wz_hc/2)
        arr[int(Ns_b[1])] = 'brick x %s y %s z %s\n' %(Wx_hc,Wy_hc,Wz_hc)
        arr[int(Ns_m[1])] = 'move Volume 2 x %s y %s z %s include_merged' %(Wx_hc/2,-1-Wy_hc/2+params['overlap'],Wz_hc/2)
    
    
        arr[int(Ns_b[2])] = 'brick x %s y %s z %s\n' %(Wx_hc,Wy_hc,Wz_hc)
        arr[int(Ns_m[2])] = 'move Volume 3 x %s y %s z %s include_merged' %(Wx_hc/2,-1-Wy_hc/2+params['overlap'],Wz_hc/2)
                
        arr[int(Ns_b[3])] = 'brick x %s y %s z %s\n' %(params['Rw'],params['Rw'],params['Rw'])
        arr[int(Ns_m[3])] = 'move Volume 4 x %s y %s z %s include_merged' %(params['Rw']/2,-1-params['Rw']/2+params['overlap'],params['Rw']/2)
                    
        arr[int(Ns_s[0])] = 'volume 4 size %s' %(params['hf']) 
        arr[int(Ns_s[1])] = 'volume 3 size %s' %(params['hc']) 
        
        arr[int(Ns_s[2])] = 'volume 1 size %s' %(params['hc']) 
    
        arr[int(Ns_r[0])] = 'refine surface 16,17,18 numsplit %s bias 1.0 depth 0' %(params['rf']) 
        
        arr[int(Ns_e[0])] = 'block 1 element type HEX%s' %(params['ElementType']) 
        
        arr[int(Ns_ex[0])] = 'export abaqus "%s"  dimension 3  overwrite  everything' %(params['mesh_filename']) 

        
    elif ((Wx_res != 0)&(Wy_res == 0)):

        arr = np.array(lines)
        arr[int(Ns_b[0])] = 'brick x %s y %s z %s\n' %(params['Wx'],Wy_hc,Wz_hc)
        arr[int(Ns_m[0])] = 'move Volume 1 x %s y %s z %s include_merged' %(params['Wx']/2,-1-Wy_hc/2+params['overlap'],Wz_hc/2)
        arr[int(Ns_b[1])] = 'brick x %s y %s z %s\n' %(Wx_hc,Wy_hc,Wz_hc)
        arr[int(Ns_m[1])] = 'move Volume 2 x %s y %s z %s include_merged' %(Wx_hc/2,-1-Wy_hc/2+params['overlap'],Wz_hc/2)
    

        arr[int(Ns_b[2])] = 'brick x %s y %s z %s\n' %(Wx_hc,Wy_hc,params['Wz'])
        arr[int(Ns_m[2])] = 'move Volume 3 x %s y %s z %s include_merged' %(Wx_hc/2,-1-Wy_hc/2+params['overlap'],params['Wz']/2)
        arr[int(Ns_b[3])] = 'brick x %s y %s z %s\n' %(Wx_hc,Wy_hc,Wz_hc)
        arr[int(Ns_m[3])] = 'move Volume 4 x %s y %s z %s include_merged' %(Wx_hc/2,-1-Wy_hc/2+params['overlap'],Wz_hc/2)
                                                                            
                                                                            
        arr[int(Ns_b[4])] = 'brick x %s y %s z %s\n' %(Wx_res,Wy_hc,Wz_res)
        arr[int(Ns_m[4])] = 'move Volume 5 x %s y %s z %s include_merged' %(Wx_hc+Wx_res/2,-1-Wy_hc/2+params['overlap'],Wz_hc+Wz_res/2)

        
        arr[int(Ns_b[5])] = 'brick x %s y %s z %s\n' %(Wx_hc,Wy_hc,Wz_hc)
        arr[int(Ns_m[5])] = 'move Volume 6 x %s y %s z %s include_merged' %(Wx_hc/2,-1-Wy_hc/2+params['overlap'],Wz_hc/2)
                
        arr[int(Ns_b[6])] = 'brick x %s y %s z %s\n' %(params['Rw'],params['Rw'],params['Rw'])
        arr[int(Ns_m[6])] = 'move Volume 7 x %s y %s z %s include_merged' %(params['Rw']/2,-1-params['Rw']/2+params['overlap'],params['Rw']/2)
                    
        arr[int(Ns_s[0])] = 'volume 7 size %s' %(params['hf']) 
        arr[int(Ns_s[1])] = 'volume 6 size %s' %(params['hc']) 
    
        arr[int(Ns_s[2])] = 'volume 1 size %s' %(params['hc']) 
        arr[int(Ns_s[3])] = 'volume 3 size %s' %(params['hc']) 
        arr[int(Ns_s[4])] = 'volume 5 size %s' %(params['hc']) 
    
        arr[int(Ns_r[0])] = 'refine surface 28,29,30 numsplit %s bias 1.0 depth 0' %(params['rf']) 
        
        arr[int(Ns_e[0])] = 'block 1 element type HEX%s' %(params['ElementType']) 
        
        arr[int(Ns_ex[0])] = 'export abaqus "%s"  dimension 3  overwrite  everything' %(params['mesh_filename']) 

        
    elif ((Wx_res == 0)&(Wy_res == 0)):
    
        arr = np.array(lines)
        arr[int(Ns_b[0])] = 'brick x %s y %s z %s\n' %(params['Wx'],params['Wy'],params['Wz'])
        arr[int(Ns_m[0])] = 'move Volume 1 x %s y %s z %s include_merged' %(params['Wx']/2,-1-params['Wy']/2+params['overlap'],params['Wz']/2)
       
        arr[int(Ns_b[1])] = 'brick x %s y %s z %s\n' %(params['Rw'],params['Rh'],params['Rw'])
        arr[int(Ns_m[1])] = 'move Volume 2 x %s y %s z %s include_merged' %(params['Rw']/2,-1-params['Rh']/2+params['overlap'],params['Rw']/2)
        arr[int(Ns_b[2])] = 'brick x %s y %s z %s\n' %(params['Rw'],params['Rh'],params['Rw'])
        arr[int(Ns_m[2])] = 'move Volume 3 x %s y %s z %s include_merged' %(params['Rw']/2,-1-params['Rh']/2+params['overlap'],params['Rw']/2)
    
        arr[int(Ns_s[0])] = 'volume 3 size %s' %(params['hf']) 
        arr[int(Ns_s[1])] = 'volume 1 size %s' %(params['hc']) 
        
        arr[int(Ns_r[0])] = 'refine surface 4,5,6 numsplit %s bias 1.0 depth 0' %(params['rf']) 
        
        arr[int(Ns_e[0])] = 'block 1 element type HEX%s' %(params['ElementType']) 

        arr[int(Ns_ex[0])] = 'export abaqus "%s"  dimension 3  overwrite  everything' %(params['mesh_filename']) 



    working_file = ('%s%s'%(params['mesher_filename'][:-4],'_working.JOU') )    
    np.savetxt(working_file, arr, fmt='%s')
    
    return working_file










def ReadRunner(solver_filename):
    """
    Objective
        Read the .feb FEBio file and identify the location of the Nodal Matrix,
        Connectivity Matrix, boundary conditions, and material parameters

    Parameters
    ----------
    solver_filename : (string)
        Path to .feb input file

    Returns
    -------
    lines : (list)
        .feb file 
    Ns : (array)
        Stores the locaion of the Nodal Matrix,Connectivity Matrix, boundary 
        conditions
    NsM : (Array)
        Stores the location of material parameters

    """
    lines = []   # Open .feb file
    with open('%s' %(solver_filename), 'rt') as file: 
        for line in file: 
            lines.append(line)
    Phrase = ["Nodes", "Elements", "Surface name","value lc="]; nph = len(Phrase); Ns = []
    for i in range(0,nph):
        Start = Phrase[i]   #Looks through the data file for the phrases
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
    Phrase = ["<mu>","<k>","<Jm>"]; nph = len(Phrase); NsM = []
    for i in range(0,nph):
        Start = Phrase[i]   #Looks through the data file for the phrases
        ns = 0
        for line in lines: 
          index = 0   
          ns += 1
          while index < len(line): 
            index = line.find(Start, index)
            if index == -1: 
              break        
            NsM = np.append(NsM,int(ns-1))
            index += len(Start) 

    return lines, Ns, NsM










def LoadMesh(params,NsR):
    """
    Objective
        Read the .inp file produced by running the Cubit journal file and 
        identify the location of the Nodal Matrix,Connectivity Matrix, and 
        sidesets that are used to apply the boundary conditions. This 
        information is then reworked into the FEBio .feb syntax.
        
        This function could be improved, if the reader is daring.
        
    Parameters
    ----------
    params : (dictionary)
        A dictionary that stores parameters that define the simulation as well
        as a few paths.
    NsR : (array)
        Stores the locaion of the Nodal Matrix,Connectivity Matrix, boundary 
        conditions
    Returns
    -------
    lines : (list)
        .inp file
    Nodes : (list)
        Nodal Matrix, .feb syntax
    Elems : (list)
        Connectivity Matrix, .feb syntax
    BCx : (list)
        Symmetry boundary conditions, .feb syntax
    BCy : (list)
        Symmetry boundary conditions, .feb syntax
    BCz : (list)
        Symmetry boundary conditions, .feb syntax
    BCs : (list)
        Contact boundary conditions, .feb syntax

    """
    # print('Loading Mesh File')
    c = 0
    while c == 0: 
        lines = []   # load .inp file
        with open('%s' %(params['mesh_filename']), 'rt') as file: 
            for line in file: 
                lines.append(line)
                
        Phrase = ["*NODE", "*ELEMENT", "S I D E S E T S", "*ELSET, ELSET=SS1", "*ELSET, ELSET=SS2", "*ELSET, ELSET=SS3", "*ELSET, ELSET=SS4","*SURFACE"]; nph = len(Phrase); 
        Ns = []; Ns_bcx = []; Ns_bcz = []; Ns_bcy = []; Ns_bcs = []; Ns_Surf = []
        for i in range(0,nph):
            Start = Phrase[i]   #Looks through the data file for the phrases
            ns = 0
            for line in lines: 
              index = 0   
              ns += 1
              while index < len(line): 
                index = line.find(Start, index)
                if index == -1: 
                  break        
                Ns = np.append(Ns,int(ns-1))
                if i == 3:   Ns_bcx = np.append(Ns_bcx,int(ns-1))
                if i == 4:   Ns_bcz = np.append(Ns_bcz,int(ns-1))
                if i == 5:   Ns_bcy = np.append(Ns_bcy,int(ns-1))
                if i == 6:   Ns_bcs = np.append(Ns_bcs,int(ns-1))
                if i == 7:   Ns_Surf = np.append(Ns_Surf,int(ns-1))

                index += len(Start) 
        if len(Ns_bcs) >= 1: c =1
        else: print('Loading Mesh')

    # Determines how many nodes and elements there are in the sample
    Nn = int(Ns[1]-2)-int(Ns[0]+1); Ne = int(Ns[2]-1)-int(Ns[1]+1)
    Nn_ind = int(NsR[1]-NsR[0]-1)
    
    print('Number of Elements: %i' %(Ne))
    ElemType = len(np.array(re.split(',|\n',lines[i + int(Ns[1])+1] )[:-1]))
    
    # Stores the nodal positions and connectivity matrix into an array
    Nodes_sec = np.zeros((Nn, 4))
    Elem_sec = np.zeros((Ne, ElemType))
    
    for i in range(0,Nn):
        Nodes_sec[i,:] = np.array(re.split(',|\n',lines[i + int(Ns[0])+1] )[:-1])

    for i in range(0,Ne):
        Elem_sec[i,:] = np.array(re.split(',|\n',lines[i + int(Ns[1])+1] )[:-1])
 
    # First column numbers the node
    Nodes_sec[:,0] = np.linspace(int(NsR[1]-NsR[0]),int(Ns[1]-Ns[0]-4) + int(NsR[1]-NsR[0]),int(Ns[1]-Ns[0]-4) +1)
    Elem_sec[:,0] = np.linspace(int(NsR[5]-NsR[4]),int(Ns[2]-Ns[1]-4) + int(NsR[5]-NsR[4]),int(Ns[2]-Ns[1]-3) +1)
    Elem_sec[:,1:] = Elem_sec[:,1:] + int(NsR[1]-NsR[0]) - 1
    
    # Formats the nodal matrix and connectivity matrix into FEBio syntax
    Nodes = list()
    Elems = list()
    for i in range(len(Nodes_sec[:,0])):
        nstr = ('%s%i%s%f%s%f%s%f%s' %('<node id="',Nodes_sec[i,0],'">',Nodes_sec[i,1],',',Nodes_sec[i,2],',',Nodes_sec[i,3],'</node>'))
        Nodes = np.append(Nodes,nstr)
        
    if ElemType == 21:
        for i in range(len(Elem_sec[:,0])):
            estr = ('%s%i%s%i%s%i%s%i%s%i%s%i%s%i%s%i%s%i%s%i%s%i%s%i%s%i%s%i%s%i%s%i%s%i%s%i%s%i%s%i%s%i%s' %('<elem id="',Elem_sec[i,0],'">',Elem_sec[i,1],',',Elem_sec[i,2],',',Elem_sec[i,3],',',Elem_sec[i,4],',',Elem_sec[i,5],',',Elem_sec[i,6],',',Elem_sec[i,7],',',Elem_sec[i,8],',',Elem_sec[i,9],',',Elem_sec[i,10],',',Elem_sec[i,11],',',Elem_sec[i,12],',',Elem_sec[i,13],',',Elem_sec[i,14],',',Elem_sec[i,15],',',Elem_sec[i,16],',',Elem_sec[i,17],',',Elem_sec[i,18],',',Elem_sec[i,19],',',Elem_sec[i,20],'</elem>'))
            Elems = np.append(Elems,estr)
    elif ElemType == 9:
        for i in range(len(Elem_sec[:,0])):
            estr = ('%s%i%s%i%s%i%s%i%s%i%s%i%s%i%s%i%s%i%s' %('<elem id="',Elem_sec[i,0],'">',Elem_sec[i,1],',',Elem_sec[i,2],',',Elem_sec[i,3],',',Elem_sec[i,4],',',Elem_sec[i,5],',',Elem_sec[i,6],',',Elem_sec[i,7],',',Elem_sec[i,8],'</elem>'))
            Elems = np.append(Elems,estr)




    # Extracts information about the boundary conditions from the sidesets in 
    # the .inp file and stores it into the below lists
    BCx_list = []; BCy_list = []; BCz_list = [];  BCs_list = []; 

    kill = 0; c = 0
    while kill == 0:
        while c != len(Ns_bcx)-1:
            st = int(Ns_bcx[c]+1); ed = int(Ns_bcx[c+1])

            for i in range(st,ed):
                qu = np.array(re.split(',|\n',lines[i] )[:-1])
                BCx_list = np.append(BCx_list, qu)
            c+=1
        
        st = int(Ns_bcx[c]+1); ed = int(Ns_Surf[0])
        for i in range(st,ed):
            qu = np.array(re.split(',|\n',lines[i] )[:-1])
            BCx_list = np.append(BCx_list, qu)

        kill = 1

    BCx_List = BCx_list[BCx_list != '']; BCx_arr = np.zeros(len(BCx_List))
    for i in range(0,len(BCx_List)):
        BCx_arr[i] = int(BCx_List[i])
        
        
        
    kill = 0; c = 0
    while kill == 0:
        while c != len(Ns_bcz)-1:
            st = int(Ns_bcz[c]+1); ed = int(Ns_bcz[c+1])

            for i in range(st,ed):
                qu = np.array(re.split(',|\n',lines[i] )[:-1])
                BCz_list = np.append(BCz_list, qu)
            c+=1
        
        st = int(Ns_bcz[c]+1); ed = int(Ns_Surf[1])
        for i in range(st,ed):
            qu = np.array(re.split(',|\n',lines[i] )[:-1])
            BCz_list = np.append(BCz_list, qu)

        kill = 1

    BCz_List = BCz_list[BCz_list != '']; BCz_arr = np.zeros(len(BCz_List))
    for i in range(0,len(BCz_List)):
        BCz_arr[i] = int(BCz_List[i])



    kill = 0; c = 0
    while kill == 0:
        while c != len(Ns_bcy)-1:
            st = int(Ns_bcy[c]+1); ed = int(Ns_bcy[c+1])

            for i in range(st,ed):
                qu = np.array(re.split(',|\n',lines[i] )[:-1])
                BCy_list = np.append(BCy_list, qu)
            c+=1
        
        st = int(Ns_bcy[c]+1); ed = int(Ns_Surf[2])
        for i in range(st,ed):
            qu = np.array(re.split(',|\n',lines[i] )[:-1])
            BCy_list = np.append(BCy_list, qu)

        kill = 1

    BCy_List = BCy_list[BCy_list != '']; BCy_arr = np.zeros(len(BCy_List))
    for i in range(0,len(BCy_List)):
        BCy_arr[i] = int(BCy_List[i])



    kill = 0; c = 0
    while kill == 0:
        while c != len(Ns_bcs)-1:
            st = int(Ns_bcs[c]+1); ed = int(Ns_bcs[c+1])

            for i in range(st,ed):
                qu = np.array(re.split(',|\n',lines[i] )[:-1])
                BCs_list = np.append(BCs_list, qu)
            c+=1
        
        st = int(Ns_bcs[c]+1); ed = int(Ns_Surf[3])
        for i in range(st,ed):
            qu = np.array(re.split(',|\n',lines[i] )[:-1])
            BCs_list = np.append(BCs_list, qu)

        kill = 1

    BCs_List = BCs_list[BCs_list != '']; BCs_arr = np.zeros(len(BCs_List))
    for i in range(0,len(BCs_List)):
        BCs_arr[i] = int(BCs_List[i])

    All_BC = np.copy(BCx_arr)
    All_BC = np.append(All_BC,BCz_arr)
    All_BC = np.append(All_BC,BCy_arr)
    All_BC = np.append(All_BC,BCs_arr)




    # Finds which nodes of the elements in the above sidesets belongs to which
    # boundary condition. Reorders into a surface element
    BCx_Arr = [];  BCy_Arr = [];  BCz_Arr = [];  BCs_Arr = []
    if ElemType == 21: 
        Nn_bc = 8
    else:  
        Nn_bc = 4

    # print('Extracting BC Data')
    # for ne in range(0,len(Elem_sec[:,0])):
    for NE in range(0,len(All_BC)):
        ne = int(All_BC[NE]-1)
        
        # if ne%100==0: print(ne)
        quadx = []; quady = []; quadz = []; quads = []
        for nn in range(1,ElemType):
            if Nodes_sec[int(Elem_sec[ne,nn]-Nn_ind-1), 1] == 0:
                quadx = np.append(quadx, int(Nodes_sec[int(Elem_sec[ne,nn]-Nn_ind-1),0]))

            if Nodes_sec[int(Elem_sec[ne,nn]-Nn_ind-1), 3] == 0:
                quadz = np.append(quadz, int(Nodes_sec[int(Elem_sec[ne,nn]-Nn_ind-1),0]))

            if Nodes_sec[int(Elem_sec[ne,nn]-Nn_ind-1), 2] == min(Nodes_sec[:,2]):
                quady = np.append(quady, int(Nodes_sec[int(Elem_sec[ne,nn]-Nn_ind-1),0]))

            if Nodes_sec[int(Elem_sec[ne,nn]-Nn_ind-1), 2] == max(Nodes_sec[:,2]):
                quads = np.append(quads, int(Nodes_sec[int(Elem_sec[ne,nn]-Nn_ind-1),0]))

        if ElemType == 21:
            if len(quadx) == Nn_bc: 
                quadx_ord = [int(quadx[3]),int(quadx[2]),int(quadx[1]),int(quadx[0]),int(quadx[6]),int(quadx[5]),int(quadx[4]),int(quadx[7])]
                BCx_Arr = np.append(BCx_Arr, quadx_ord)
                
            if len(quadz) == Nn_bc: 
                quadz_ord = [int(quadz[0]),int(quadz[1]), int(quadz[3]),int(quadz[2]),int(quadz[4]),int(quadz[7]),int(quadz[5]),int(quadz[6])]
                BCz_Arr = np.append(BCz_Arr, quadz_ord)
    
            if len(quady) == Nn_bc: 
                quady_ord = [int(quady[0]),int(quady[1]),int(quady[3]),int(quady[2]), int(quady[4]),int(quady[7]), int(quady[5]),int(quady[6])]
                BCy_Arr = np.append(BCy_Arr, quady_ord)
    
            if len(quads) == Nn_bc: 
                quads_ord = [int(quads[0]),int(quads[1]),int(quads[3]),int(quads[2]),int(quads[4]),int(quads[7]), int(quads[5]),int(quads[6])]
                BCs_Arr = np.append(BCs_Arr, quads_ord)
        else:
            if len(quadx) == Nn_bc: 
                quadx_ord = [int(quadx[3]),int(quadx[2]),int(quadx[1]),int(quadx[0])]
                BCx_Arr = np.append(BCx_Arr, quadx_ord)
                
            if len(quadz) == Nn_bc: 
                quadz_ord = [int(quadz[0]),int(quadz[1]), int(quadz[3]),int(quadz[2])]
                BCz_Arr = np.append(BCz_Arr, quadz_ord)
    
            if len(quady) == Nn_bc: 
                quady_ord = [int(quady[0]),int(quady[1]),int(quady[3]),int(quady[2])]
                BCy_Arr = np.append(BCy_Arr, quady_ord)
    
            if len(quads) == Nn_bc: 
                quads_ord = [int(quads[0]),int(quads[1]),int(quads[3]),int(quads[2])]
                BCs_Arr = np.append(BCs_Arr, quads_ord)
                
    if ElemType == 21:
        BCx_Arr = np.reshape(BCx_Arr,((int(len(BCx_Arr)/8), 8)) )
        BCz_Arr = np.reshape(BCz_Arr,((int(len(BCz_Arr)/8), 8)) )
        BCy_Arr = np.reshape(BCy_Arr,((int(len(BCy_Arr)/8), 8)) )
        BCs_Arr = np.reshape(BCs_Arr,((int(len(BCs_Arr)/8), 8)) )
    else:
        BCx_Arr = np.reshape(BCx_Arr,((int(len(BCx_Arr)/4), 4)) )
        BCz_Arr = np.reshape(BCz_Arr,((int(len(BCz_Arr)/4), 4)) )
        BCy_Arr = np.reshape(BCy_Arr,((int(len(BCy_Arr)/4), 4)) )
        BCs_Arr = np.reshape(BCs_Arr,((int(len(BCs_Arr)/4), 4)) )
        
    BCx_arr = np.copy(BCx_Arr);BCz_arr = np.copy(BCz_Arr)
    BCy_arr = np.copy(BCy_Arr); BCs_arr = np.copy(BCs_Arr)
    
    
    # The boundary condition arrays are converted into FEBio syntax
    print('Formatting Mesh Data')
    BCx = list()
    BCz = list()
    BCy = list()
    BCs = list()
    
    if ElemType == 21:
        for i in range(len(BCx_arr[:,0])):
            bcx = ('%s%i%s%i%s%i%s%i%s%i%s%i%s%i%s%i%s%i%s' %('<quad8 id="',int(i+1),'">',BCx_arr[i,0],',',BCx_arr[i,1],',',BCx_arr[i,2],',',BCx_arr[i,3],',',BCx_arr[i,4],',',BCx_arr[i,5],',',BCx_arr[i,6],',',BCx_arr[i,7],'</quad8>' ))               
            BCx = np.append(BCx,bcx)
        for i in range(len(BCz_arr[:,0])):
            bcz = ('%s%i%s%i%s%i%s%i%s%i%s%i%s%i%s%i%s%i%s' %('<quad8 id="',int(i+1),'">',BCz_arr[i,0],',',BCz_arr[i,1],',',BCz_arr[i,2],',',BCz_arr[i,3],',',BCz_arr[i,4],',',BCz_arr[i,5],',',BCz_arr[i,6],',',BCz_arr[i,7],'</quad8>' ))               
            BCz = np.append(BCz,bcz)
        for i in range(len(BCy_arr[:,0])):
            bcy = ('%s%i%s%i%s%i%s%i%s%i%s%i%s%i%s%i%s%i%s' %('<quad8 id="',int(i+1),'">',BCy_arr[i,0],',',BCy_arr[i,1],',',BCy_arr[i,2],',',BCy_arr[i,3],',',BCy_arr[i,4],',',BCy_arr[i,5],',',BCy_arr[i,6],',',BCy_arr[i,7],'</quad8>' ))               
            BCy = np.append(BCy,bcy)
        for i in range(len(BCs_arr[:,0])):
            bcs = ('%s%i%s%i%s%i%s%i%s%i%s%i%s%i%s%i%s%i%s' %('<quad8 id="',int(i+1),'">',BCs_arr[i,0],',',BCs_arr[i,1],',',BCs_arr[i,2],',',BCs_arr[i,3],',',BCs_arr[i,4],',',BCs_arr[i,5],',',BCs_arr[i,6],',',BCs_arr[i,7],'</quad8>' ))               
            BCs = np.append(BCs,bcs)
    else:
        for i in range(len(BCx_arr[:,0])):
            bcx = ('%s%i%s%i%s%i%s%i%s%i%s' %('<quad4 id="',int(i+1),'">',BCx_arr[i,0],',',BCx_arr[i,1],',',BCx_arr[i,2],',',BCx_arr[i,3],'</quad4>' ))               
            BCx = np.append(BCx,bcx)
        for i in range(len(BCz_arr[:,0])):
            bcz = ('%s%i%s%i%s%i%s%i%s%i%s' %('<quad4 id="',int(i+1),'">',BCz_arr[i,0],',',BCz_arr[i,1],',',BCz_arr[i,2],',',BCz_arr[i,3],'</quad4>' ))               
            BCz = np.append(BCz,bcz)
        for i in range(len(BCy_arr[:,0])):
            bcy = ('%s%i%s%i%s%i%s%i%s%i%s' %('<quad4 id="',int(i+1),'">',BCy_arr[i,0],',',BCy_arr[i,1],',',BCy_arr[i,2],',',BCy_arr[i,3],'</quad4>' ))               
            BCy = np.append(BCy,bcy)
        for i in range(len(BCs_arr[:,0])):
            bcs = ('%s%i%s%i%s%i%s%i%s%i%s' %('<quad4 id="',int(i+1),'">',BCs_arr[i,0],',',BCs_arr[i,1],',',BCs_arr[i,2],',',BCs_arr[i,3],'</quad4>' ))               
            BCs = np.append(BCs,bcs)


    return lines, Nodes, Elems, BCx, BCy, BCz, BCs










def ModifyRunner(RunFile,NsR,Nodes,Elems, BCx, BCy, BCz, BCs,NsM,params):
    """
    Objective
        Modify the .feb file with the new mesh, boundary conditions and 
        material parameters
    Parameters
    ----------
    RunFile : (list)
        .feb file.
    NsR : (array)
        Stores the locaion of the Nodal Matrix,Connectivity Matrix, boundary 
        conditions
    Nodes : (list)
        Nodal Matrix, .feb syntax
    Elems : (list)
        Connectivity Matrix, .feb syntax
    BCx : (list)
        Symmetry boundary conditions, .feb syntax
    BCy : (list)
        Symmetry boundary conditions, .feb syntax
    BCz : (list)
        Symmetry boundary conditions, .feb syntax
    BCs : (list)
        Contact boundary conditions, .feb syntax
    NsM : (Array)
        Stores the location of material parameters
    params : (dictionary)
        A dictionary that stores parameters that define the simulation as well
        as a few paths.
    Returns
    -------
    working_file: (list)
        final .feb file.
    """
    
    print('Modifying Runner')

    dif = 0
    # Insert nodal matrix
    del RunFile[int(NsR[2]+1):int(NsR[3])]
    Nodes_List = np.ndarray.tolist(Nodes) 
    for elem in reversed(Nodes_List) :
        RunFile.insert(int(NsR[2]+1), elem)    
    dif += (int(NsR[3]) - int(NsR[2]+1)) - len(Nodes)

    # Insert element connectivity matrix
    del RunFile[int(NsR[6]+1)-dif:int(NsR[7])-dif]
    Elems_List = np.ndarray.tolist(Elems)
    for elem in reversed(Elems_List) :
        RunFile.insert(int(NsR[6]+1)-dif, elem)        
    dif += (int(NsR[7]) - int(NsR[6]+1)) - len(Elems)

    # Insert x-symmetry BC
    del RunFile[int(NsR[10]+1)-dif:int(NsR[11]-1)-dif]
    BCx_List = np.ndarray.tolist(BCx)
    for elem in reversed(BCx_List) :
        RunFile.insert(int(NsR[10]+1)-dif, elem)    
    dif += (int(NsR[11]-1) - int(NsR[10]+1)) - len(BCx)

    # Insert z-symmetry BC
    del RunFile[int(NsR[11]+1)-dif:int(NsR[12]-1)-dif]
    BCz_List = np.ndarray.tolist(BCz)
    for elem in reversed(BCz_List) :
        RunFile.insert(int(NsR[11]+1)-dif, elem)    
    dif += (int(NsR[12]-1) - int(NsR[11]+1)) - len(BCz)

    # Insert y-symmetry BC
    del RunFile[int(NsR[12]+1)-dif:int(NsR[13]-1)-dif]
    BCy_List = np.ndarray.tolist(BCy)
    for elem in reversed(BCy_List) :
        RunFile.insert(int(NsR[12]+1)-dif, elem)    
    dif += (int(NsR[13]-1) - int(NsR[12]+1)) - len(BCy)

    # Insert contact BC
    del RunFile[int(NsR[14]+1)-dif:int(NsR[15]-1)-dif]
    BCs_List = np.ndarray.tolist(BCs)
    for elem in reversed(BCs_List) :
        RunFile.insert(int(NsR[14]+1)-dif, elem)    
    dif += (int(NsR[15]-1) - int(NsR[14]+1)) - len(BCs)
    
    # Insert material parameters
    del RunFile[int(NsM[0])]
    MatC1 = '<mu>%s</mu>\n' %(params['mu_nondim'])
    RunFile.insert(int(NsM[0]), MatC1)          
    del RunFile[int(NsM[1])]
    MatC2 = '<k>%s</k>\n' %(params['kap_nondim'])
    RunFile.insert(int(NsM[1]), MatC2)  
    del RunFile[int(NsM[2])]
    MatC3 = '<Jm>%s</Jm>\n' %(params['JM_nondim'])
    RunFile.insert(int(NsM[2]), MatC3)  
    del RunFile[int(NsR[16]-dif)]
    Disp = '<value lc="1">%s</value>\n' %(params['deltamax']+params['overlap'])
    RunFile.insert(int(NsR[16]-dif), Disp)    

    # save
    working_file = ('%s%s'%(params['solver_filename'][:-4],'_working.feb') )
    np.savetxt(working_file, RunFile, fmt='%s')    
    
    
    return working_file











def RunMesher(command, args,params):
    """
    Objective
        Run Cubit with the new journal file

    Parameters
    ----------
    command : (string)
        Cubit executable
    args : (string)
        Cubit Input file.
    params : TYPE
        A dictionary that stores parameters that define the simulation as well
        as a few paths.
    Returns
    -------


    """
    print('Running Mesher')

    Results_file = ( params['mesh_filename'] )

    if exists(Results_file):    
        os.remove(Results_file)
    
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    subprocess.Popen([command] + args, startupinfo=startupinfo)#.wait()

    Pass = 0
    while Pass == 0:
        if exists(params['mesh_filename']):  Pass = 1
        else: time.sleep(0.5)
    
    
    return 









def RunSolver(command, args,solver_filename):
    """
    Objective
        Run FEBio with the new .feb file

    Parameters
    ----------
    command : (string)
        FEBio executable
    args : (string)
        FEBio Input file.
    solver_filename : TYPE
        DESCRIPTION.

    Returns
    -------


    """
    print('Running Solver')

    Results_file = ('%s%s'%(solver_filename[:-16],'Fy_output.txt') )
    
    if exists(Results_file):    
        os.remove(Results_file)
    
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    subprocess.Popen([command] + args, startupinfo=startupinfo)#.wait()

    Pass = 0
    while Pass == 0:
        if exists(Results_file):  Pass = 1
        else: time.sleep(0.1)

    return 









def LoadData(params):
    """
    Objective
        Wait for the solver to finish and then load the load-displacement data.
        
        This could be improved on.

    Parameters
    ----------
    params : (dictionary)
        A dictionary that stores parameters that define the simulation as well
        as a few paths.
    Returns
    -------
    Data : (array)
        Load-displacement data.

    """
    print('Writing Output')

    err = 1; n = 0; nmax = 600
    while ((err == 1)&(n<nmax)):
        # if n%10==0: print('Loading.....')
        time.sleep(1)

        # Load the FEBio output
        Fy_file = ('%s%s'%(params['solver_filename'][:-16],'Fy_output.txt') )
        uy_file = ('%s%s'%(params['solver_filename'][:-16],'uy_output.txt') )
    
        uy_data_raw = []   # read the text file for displacement of nodes
        with open('%s' %(uy_file), 'rt') as file: 
            for line in file: 
                uy_data_raw.append(line)
        Phrase = ["*Data"]; nph = len(Phrase); Ns = []
        for i in range(0,nph):
            Start = Phrase[i]   #Looks through the data file for the phrases
            ns = 0
            for line in uy_data_raw: 
              index = 0   
              ns += 1
              while index < len(line): 
                index = line.find(Start, index)
                if index == -1: 
                  break        
                Ns = np.append(Ns,int(ns-1))
                index += len(Start) 
            
        # Store the data into an array
        Fy_data_raw = []    # read the text file for rigid body load
        with open('%s' %(Fy_file), 'rt') as file: 
            for line in file: 
                Fy_data_raw.append(line)
    
        # store indentation data into an array
        uydata = np.zeros((len(Ns),2))
        for i in range(0,len(Ns)):
            ind = int(Ns[i] + 1)
            uydata[i,0] = float(uy_data_raw[ind].split()[0])
            uydata[i,1] = float(uy_data_raw[ind].split()[1])

        # Check to see if model is finished
        if len(uydata) == 0: 
            err = 1; 
            time.sleep(1)
        else: 
            if uydata[-1,1] == params['deltamax'] + params['overlap']: 
                err = abs(params['deltamax'] - uydata[-1,1])
            else:
                err = 1; 
                time.sleep(1)
        n+=1
        
    # store load data into an array
    Fydata = np.zeros((len(Ns),2))
    for i in range(1,len(Ns)+1):
        ind = int(4*i-1)
        Fydata[i-1,0] = float(Fy_data_raw[ind].split()[0])
        Fydata[i-1,1] = float(Fy_data_raw[ind].split()[1])

    Data = np.copy(uydata)
    Data[:,1] = -Fydata[:,1]
    Data[:,0] = -uydata[:,1] + params['overlap']
        
    return Data 


















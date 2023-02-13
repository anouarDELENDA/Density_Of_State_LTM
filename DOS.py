import numpy as np
from Tight_B import*
import matplotlib.pyplot as plt
import cProfile
import pstats
# read The k points: #The provided K_pts_list.txt 

print("step 1 - Reading the provided K points")

# The K points are represented by weigthed quantites u,v and w with out the b1,b2 and b3 basis
# So we need to get the real K points that describe the reciprocal space

recep_b = 2*pi*np.array([[-1, 1, 1],
                  [1, -1, 1],
                  [1, 1, -1]])


K_points = []

with open("k_pts_list.txt", "r") as f:
    
    for i, l in enumerate(f.readlines()):
        
        # here we evaluate K as : K = u*b1+u*b2+w*b3

        uvw = [float(x)*b for (x, b) in zip(l.split()[:3], recep_b)] # depending on the structure of the files we can custom this line.
                                                                     # mine have the following structure: u v w weight                   
        
        K_points.append(uvw)
        
        
    f.close()
 
# convert to numpy

K_points = np.array(K_points)

print("K point array shape", np.shape(K_points), "\n step 1  Done ...")

print("Step 2 - getting the Eigenvalues for the K points ...")

# initilize the tb method parameter for Ge:

Ge_parameters= [ [-6.78, 5.31, 2.62, 6.82],                        # Vpar=[Vss/V0, Vsp/V1, Vxx/V2, Vxy/V3]
                            [-5.88, -5.88, *[2.53 for _ in range(6)]] ] # diag_ele = [Esa, Esb, Epxa, Epya, Epza, Epxb, Epyb, Epzb]

# evaluate the g's function for more details see the ref: 
'''
    Chadi, D. J., and Marvin L. Cohen. "Tight-binding calculations of (111) surface densities of states of 
    Ge and GaAs." Solid State Communications 16.6 (1975): 691-694
'''

def get_g_k(K):
    ''' For earch K point in our path and for the neighboring Orbital Atomic position
    we evaluate the g's function (4 function) '''
    # this g function lies in the approximation of the first nighbors
    neighs = .25 * np.array( [[1 , 1, 1],
                                          [1 , -1, -1],
                                          [- 1, 1, -1],
                                          [- 1, -1, 1]])
    
    
    a, b, c, d = [ np.exp( 1j * np.sum( K @ neigh )) for neigh in neighs ]
    
    return [.25*(a + b + c + d),   .25*(a - b - c + d),   .25*(a - b + c - d),   .25*(a + b - c - d)]

# Get the eigen energies:

def Eigens(K):
    '''
        Return the Eigen-energies according to the given k point
        The Tight Binding method described here is according to "Chadi and Cohen Work"
    '''
    g0, g1, g2, g3 =  get_g_k(K=K)   # see the above function
    Vss, Vsp, Vxx, Vxy  = Ge_parameters[0] # fitted parameteres
    diag_ele = Ge_parameters[1] # fitted parameteres
    H   = np.array([
                 #Sa,     Sb,                   Pxa,                   Pya,                   Pza,    Pxb,    Pyb,    Pzb       
                  [0, Vss*g0,                     0,                     0,                     0, Vsp*g1, Vsp*g2, Vsp*g3],
                  [0,      0, -Vsp*np.conjugate(g1), -Vsp*np.conjugate(g2), -Vsp*np.conjugate(g3),      0,      0,      0],
                  [0,      0,                     0,                     0,                     0, Vxx*g0, Vxy*g3, Vxy*g2],
                  [0,      0,                     0,                     0,                     0, Vxy*g3, Vxx*g0, Vxy*g1],
                  [0,      0,                     0,                     0,                     0, Vxy*g2, Vxy*g1, Vxx*g0],
                  [0,      0,                     0,                     0,                     0,      0,      0,      0],
                  [0,      0,                     0,                     0,                     0,      0,      0,      0],
                  [0,      0,                     0,                     0,                     0,      0,      0,      0]
                ], dtype= 'complex_')
    # Diagonal elements
    for i in range(len(diag_ele)):
        H[i,i] = diag_ele[i]
    # complex conjugate:
    for j in range(8):
        for i in range(0, j):
            H[j, i] = np.conjugate(H[i, j]) 
    
    # diagonalize the Hamiltonian            
    return linalg.eigvalsh(H)


# Evaluating the eigenenergies for each K points:

Numb_K_points = np.shape(K_points)[0]


Energies = np.empty(shape=(Numb_K_points, 8)) # Why 8? Basically we will have 8 bands (8 by 8 matrix)

for (i, K) in enumerate(K_points):

    Energies[i,:] = Eigens(K=K) 

print("The shape of The Eigenvalues array :", np.shape(Energies), "\n step 2 Done ...")

print("Step 3- reading the Tetrahedrons file...")

Tetras = []

with open("tetra_list.txt", "r") as f:
    
    for l in f.readlines():

        t = [int(x) for x in l.split()[0:4]]
        
        if len(t) > 4:
            print(t)
            raise Exception(f"check the tetra data in the txt file. --{t}--")

        Tetras.append(t)
        
    f.close()
   

print("The shape of the Tetrahedrons array: ", np.shape(Tetras), "\n Done ... ")

print("Step 3- Density of State.")


# This is the 1D array of energies we will sacn over it to deal with DOS:
E_trial = np.linspace(np.min(Energies[:, 0]) - 4, np.max(Energies[:, -1]) + 4, 100) # just 100 pts for the sake of time consuming
 

def get_dos(Tetra, Eners):

    '''
        This function evaluate the density of state DOS 
    '''
    DOS = np.zeros_like(E_trial) # DOS full of zeros  -initilizing- has same shape as E_trial

    # loop over the energies

    for (i,E) in enumerate(E_trial):

        # initialize dos
        
        # Now we are working under one tetra ... T
        dos = 0
        for T in Tetra:
            
           # I have four index labeling the corner (verticies) of one tetra

            E_list = [ Eners[int(x)-1, :] for x in T ]
            E1, E2, E3, E4 = E_list # @ eatch corner we have 8 bands
            
            # each E1 ... E4 contain 8 values corresponding to the 8 bands
            
            # verification if E lies within the E1 ... E4

            if (E >= np.min(E_list)) and (E <= np.max(E_list)):

                # loop over bands 
                for n in range(8): # n = band index
                
                    e_s = [ E1[n], E2[n], E3[n], E4[n] ]

                    # sort the e_s
                    e_s = sorted(e_s)                                 

                    if (e_s[0] <= E) and ( E <= e_s[-1] ):

                        e21 = e_s[1] - e_s[0]                
                        e31 = e_s[2] - e_s[0]
                        e41 = e_s[3] - e_s[0]
                        e32 = e_s[2] - e_s[1]
                        e43 = e_s[3] - e_s[2]
                        e42 = e_s[3] - e_s[1]  
        

                    if (E >= e_s[0]) and ( E < e_s[1]):

                        dos += 3*( E - e_s[0] )**2 / ( e21 * e31 * e41 ) 
                        
                    elif (E >= e_s[1]) and ( E < e_s[2]):

                        dos += ( 3 * e21 + 6 * ( E - e_s[1]) - 3 * ( ( (e31+e42)*(E-e_s[1])**2 ) / (e32*e42) ) ) / (e31*e41) 
                        
                    elif (E >= e_s[2]) and ( E < e_s[3]):

                        dos += 3 * ( e_s[3] - E )**2 / ( e41*e42*e43 ) 
        
            # Here we finished looping over 8 bands of single tetrahedron

        # Here we finished looping over all tetrahedrons
        DOS[i] = dos
            
    return DOS 


print("Profiling the code:")

# since the code do need time to be executed we thought it will be nice to estimate the totale time spent with DOS calculations
cProfile.run("get_dos(Tetras[0:9], Energies)", "Out_put_profile.dat") # Estimate the time for the firts 10 Tetra

with open("Profile_time.txt", "w") as f:
    p = pstats.Stats("Out_put_profile.dat", stream=f)
    p.sort_stats("time").print_stats()
    f.close()

fp = open("Profile_time.txt", "r")#open("Profile_time.txt", "r")
t = round( float(fp.readlines()[2].split(" ")[-2])*(len(Tetras))/(600), 2)
fp.close()

print("the totale time is estimated to be ", t, "minutes")

print("DOS calculation ... ")

DOS = get_dos(Tetra=Tetras, Eners=Energies, )

plt.plot(E_trial, (DOS/np.max(DOS)))
plt.xlabel("$DOS$")
plt.title("$Germanium Density of State$")
plt.ylabel("$E\ (eV)$")
plt.grid(True)
plt.show()

print("Done")
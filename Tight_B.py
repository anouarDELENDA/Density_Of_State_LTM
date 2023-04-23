import numpy as np 
from scipy import linalg
import matplotlib.pyplot as plt
pi = np.pi


class Tight_B:

    def __init__(self, element, N_K_points = 100,
                 neighs = .25 * np.array( [[1 , 1, 1],
                                          [1 , -1, -1],
                                          [- 1, 1, -1],
                                          [- 1, -1, 1]] ),
                 higher_E_O = False, short_path = False ) :
                
        self.neighs = neighs  # the neighbors position in atomic system
        self.element = element # the element "species" we are interesting-in "string" eg. Si, C ... etc
        self.N_K_points = N_K_points # sampling qver k points  
        self.higher_E_O = higher_E_O # inclusion of higher enregy atomic orbital     
        self.short_path = short_path # in case we need some sample data to woek with

    def get_path(self):

        ''' We provide the following path:

            L to Gamma then Gamma to X to U/K then we return to Gamma

        '''    
        # Introduce the High symetric point in Reciprical space:
        # The point are in the {Kx, Ky, Kz} basis
        G = 2 * pi * np.array([0, 0, 0])
        L = 2 * pi * np.array([1/2, 1/2, 1/2])
        K = 2 * pi * np.array([3/4, 3/4, 0])
        X = 2 * pi * np.array([0, 1, 0])
        W = 2 * pi * np.array([1/2, 1, 0])
        U = 2 * pi * np.array([1/4, 1, 1/4])

        # ..... You Can adjuts here your prefred path according to the special point .....
        LG = np.linspace(L, G, self.N_K_points)
        
        GX = np.linspace(G, X, self.N_K_points)

        XU = np.linspace(X, U, self.N_K_points )      

        UW = np.linspace(U, W, self.N_K_points )      

        WK = np.linspace(W, K, self.N_K_points )      

        KG = np.linspace(K, G, self.N_K_points )      

        if self.short_path:
            my_path1 = np.concatenate([LG, GX, XU, KG])    
        else:    
            my_path1 = np.concatenate([LG, GX, XU, UW, WK, KG])
        
        return my_path1

    def elements(self):

        ''' Return The fitted parameters as well as the hopping one for Si, Ge, C and GaAs.
            the Function return 2D array, where the first sub-array containe : 
                Vpar=[Vss/V0, Vsp/V1, Vxx/V2, Vxy/V3]
            the second contain : 
                diag_ele = [Esa, Esb, Epxa, Epya, Epza, Epxb, Epyb, Epzb]
        '''        
        list_ele = {
                    "C":  [ [-15.2, 10.25, 3, 8.30],#[-22.725, 15.2206, 3.84, 11.67],  # Vpar=[Vss/V0, Vsp/V1, Vxx/V2, Vxy/V3]
                            [-4.545, -4.545, *[2.855 for _ in range(6)]]], # diag_ele = [Esa, Esb, Epxa, Epya, Epza, Epxb, Epyb, Epzb]4.545
                    "Si": [ [-8.13, 5.88, 3.17, 7.51],#[-8.3, 5.7292, 1.715, 4.575],  # Vpar=[Vss/V0, Vsp/V1, Vxx/V2, Vxy/V3]
                            [-4.2, -4.2, *[3 for _ in range(6)]]], # diag_ele = [Esa, Esb, Epxa, Epya, Epza, Epxb, Epyb, Epzb]       
                    "Ge": [ [-6.78, 5.31, 2.62, 6.82],#[-6.78, 5.4649, 1.61, 4.9],  # Vpar=[Vss/V0, Vsp/V1, Vxx/V2, Vxy/V3]
                            [-5.88, -5.88, *[2.53 for _ in range(6)]]], # diag_ele = [Esa, Esb, Epxa, Epya, Epza, Epxb, Epyb, Epzb]       
                    "GaAs":[[-6.4513, 4.48, 1.9546, 5.0779],  # Vpar=[Vss/V0, Vsp/V1, Vxx/V2, Vxy/V3]
                            [-8.3431, -2.6569, *[1.0414 for _ in range(3)], *[3.6686 for _ in range(3)]]]#[-6.01, -4.79, *[0.19 for _ in range(3)], *[4.59 for _ in range(3)] ]]#[-8.3431, -2.6569, *[1.0414 for _ in range(3)], *[3.6686 for _ in range(3)]]] # diag_ele = [Esa, Esb, Epxa, Epya, Epza, Epxb, Epyb, Epzb]               
                    }        # diag_ele = [Esa, Esb, Epxa, Epya, Epza, Epxb, Epyb, Epzb]  
        list_ele_higher = {
                    "C":  [ [8.2109, 8.2109],  # Vpar=[VsH_p, Vp_sH]
                            [11.3700, 11.3700]], # diag_ele = [EsHa, EsHb]
                    "Si": [ [5.3749, 5.3749],  # Vpar=[VsH_p, Vp_sH]
                            [6.6850, 6.6850]], # diag_ele = [EsHa, EsHb]
                    "Ge": [ [5.2191, 5.2191],  # Vpar=[VsH_p, Vp_sH]
                            [6.3900, 6.3900]], # diag_ele = [EsHa, EsHb]
                    "GaAs":[[4.8422, 4.8077],  # Vpar=[VsH_p, Vp_sH]
                            [8.5914, 6.7386]] # diag_ele = [EsHa, EsHb]
                    }                    
        ## Original paper for these param: SEMI-EMPIRICAL TIGHT-BINDING THEORY
        ## OF THE ELECTRONIC STRUCTURE
        ## OF SEMICONDUCTORS?
        ## P. VOGL
        ## Institut fiir Theoretische Physik, Universitlt Graz, Graz, Austria            
        if self.element in list_ele.keys():
            
            if self.higher_E_O :
                for (x, y) in zip(list_ele_higher[self.element][0], list_ele_higher[self.element][0]): 
                    list_ele[self.element][0].append( x )
                    list_ele[self.element][1].append( y )

            return list_ele[self.element]            
        else:
            raise ValueError("you need to specify one of the following compound\
                Si, Ge, C or GaAs")    

    def get_g_k(self, K) :

       ''' For earch K point in our path and for the neighboring Orbital Atomic position
       we evaluate the g's function (4 function) '''
       # this g function lies in the approximation of the first nighbors

       a, b, c, d = [ np.exp( 1j * np.sum( K @ neigh )) for neigh in self.neighs ]

       return [.25*(a + b + c + d),   .25*(a - b - c + d),   .25*(a - b + c - d),   .25*(a + b - c - d)]

    def get_bands(self, g, params):

        ''' We return the eigenvalues after diagonalizing the Hamiltonian for a certain hoping pararmeters 
        (Vpar) and fitted parameters (diag_ele) and also for a g's functions '''

        g0, g1, g2, g3 = g    
        Vss, Vsp, Vxx, Vxy  = params[0][0:4] # eg. [-6.78, 5.4649, 1.61, 4.9] 
        diag_ele = params[1]
        ## Acoording to Chadi papper
        ## The basis: {Sa, Sb, Pxa, Pya, Pza, Pxb, Pyb, Pzb}

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
        if self.higher_E_O :
            
            VsH_p, Vp_sH = params[0][4::]  # mainly VsH_p == Vp_sH   

            SH_a = np.array([0, 0, 0, 0, 0, *[VsH_p*g for g in (g1, g2, g3) ], 0, 0] , dtype= "complex")
            SH_b = np.array([0, 0, *[-Vp_sH*g for g in (g1, g2, g3) ], 0, 0, 0, 0, 0] , dtype= "complex")

            # expend dim (adding two rows at the bottom)
            H = np.concatenate( (H, np.zeros(shape=(1, 8), dtype= "complex") )  , axis=0)
            H = np.concatenate( (H, np.zeros(shape=(1, 8), dtype= "complex") ), axis=0)

            # adding the two new columns expressin the interaction with higher Orbital Energy
            H = np.concatenate((H, SH_a.reshape(len(SH_a), 1)), axis=1)
            H = np.concatenate((H, SH_b.reshape(len(SH_b), 1)), axis=1)
            
        
        for i in range(len(diag_ele)):
            H[i,i] = diag_ele[i]
        #diag = np.eye(len(diag_ele)) * diag_ele # eg. [-5.55, -5.88, 1.61, 1.61, 1.61, 1.61, 1.61, 1.61,]
        
        #H = np.matmul(H, diag) 

        dim = np.shape(H)[0]

        for j in range(dim):
            for i in range(0, j):
                H[j, i] = np.conjugate(H[i, j])        

        
        return linalg.eigvalsh(H) # we use "eigvalsh" instead of "eigvals" to avoid issue of complex eigenvalues ...   

    def get_energies(self, k_point):
        
        ''' Re-evaluate the Eigenvalues for each K point '''

        fitted_pars =  self.elements() 
        hop_g = self.get_g_k(K = k_point)

        return self.get_bands(g=hop_g, params=fitted_pars) 

    def draw(self, Energies, fill_b = True):

        ''' This function show the bands structure for the selected elements if the energies are provided. 
            
            Energies: 2D array of eigenenergies             
            
        '''

        if Energies != []:   

            fig, ax = plt.subplots(nrows=1, ncols=1, sharey=True)

            for i in range( np.shape(Energies)[1] ):
                l = f"$ E_{i} $"
                ax.plot(Energies[:, i], label=l)

            #ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            if self.short_path:
                ticks  = [tick for tick in ["L", "$\Gamma$", "X", "U/K", "$\Gamma$"]]
                
            else:
                ticks  = [tick for tick in ["L", "$\Gamma$", "X", "U", "W", "K", "$\Gamma$"]]
            
            if fill_b:
                Ev = np.max(Energies[:, 3])*np.ones(shape=(np.shape(Energies)[0],))
                
                Ec = np.min(Energies[:, 4])*np.ones(shape=(np.shape(Energies)[0],))
                
                ax.fill_between(np.linspace(0, (len(ticks)-1)*self.N_K_points,np.shape(Energies)[0]),  Ev, Ec, where=Ec>Ev, color= "gray", alpha=.3)#, color="gray", alpha=.3)

            ax.set_xticks( [i*self.N_K_points for i in range(len(ticks))] )   
            ax.set_xticklabels(ticks)
            

            ax.set_xticks( [i*self.N_K_points for i in range(len(ticks))] )   
            ax.set_xticklabels(ticks)
            
            #ax.set_yticks([0])
            #ax.set_yticklabels(["$ \mathcal{E}_{n} \ (eV)$"])
            
            ax.set_xlim(left=0, right=(len(ticks)-1)*self.N_K_points)
            ax.set_xlabel("$K$")
            ax.set_ylabel("$E\ (eV)$")
            fig.suptitle(f' Band Structure of {self.element}.', font="serif")
            ax.grid()

            plt.show()

        else:

            raise ValueError("Try first to compute the eigenenrgies.\
                Hint: use -get_energies()- method.")    

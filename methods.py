import numpy as np
import numpy.random as rand
import copy
import scipy.integrate
import matplotlib.pyplot as plt
import scipy.stats

def gen_matrices_obc(N, alpha):

    M = np.zeros([N,N])

    for kk in range(0,N):
        
        for hh in range(0,N):
        
            if hh != kk:

                M[kk,hh] = np.abs(kk - hh)**(-alpha)

    return M


def gen_matrices_pbc(N, alpha):

    M = np.zeros([N,N])

    for kk in range(0,N):
        
        for hh in range(0,N):
        
            if hh != kk:
            
                if abs(kk-hh) > int(N/2):

                    M[kk,hh] = np.abs(N-np.abs(kk - hh))**(-alpha)

                else:

                    M[kk,hh] = np.abs(kk - hh)**(-alpha)

    return M

def gen_matrices_2D_obc(N, alpha):

    L = int(np.sqrt(N))


    M = np.zeros([N,N])

    for kk_x in range(0,L):

        for kk_y in range(0,L):

            # index of spin ii (kx'th row, ky'th column)
            ii =  kk_x * L + kk_y 

            for hh_x in range(0,L):

                for hh_y in range(0,L):

                    jj =  hh_x * L + hh_y 

                    if ii != jj:

                        M[ii,jj] = np.sqrt((kk_x - hh_x)**2 + (kk_y - hh_y)**2)**(-alpha)

    return M


def gen_matrices_2D_pbc(N, alpha):

    L = int(np.sqrt(N))


    M = np.zeros([N,N])

    for kk_x in range(0,L):

        for kk_y in range(0,L):

            # index of spin ii (kx'th row, ky'th column)
            ii =  kk_x * L + kk_y 

            for hh_x in range(0,L):

                for hh_y in range(0,L):

                    jj =  hh_x * L + hh_y 

                    if ii != jj:
                        
                         
                        if abs(kk_x-hh_x) > int(L/2):
 
                            xshift = L - abs(kk_x-hh_x)

                        else:

                            xshift = abs(kk_x-hh_x)


                        if abs(kk_y-hh_y) > int(L/2):
 
                            yshift = L - abs(kk_y-hh_y)

                        else:

                            yshift = abs(kk_y-hh_y)


                        M[ii,jj] = np.sqrt((xshift)**2 + (yshift)**2)**(-alpha)

    return M


def gen_matrices_2D_pbc_vacancy(N, alpha, positions):

    L = int(np.sqrt(N))

    M = np.zeros([N,N])

    xx_cord = []
    yy_cord = []

    for kk_x in range(0,L):

        for kk_y in range(0,L):

            # index of spin ii (kx'th row, ky'th column)
            ii =  kk_x * L + kk_y 

            if positions[ii] == 1:
                xx_cord += [kk_x]
                yy_cord += [kk_y]


            for hh_x in range(0,L):

                for hh_y in range(0,L):

                    jj =  hh_x * L + hh_y 

                    if ii != jj:
                        
                         
                        if abs(kk_x-hh_x) > int(L/2):
 
                            xshift = L - abs(kk_x-hh_x)

                        else:

                            xshift = abs(kk_x-hh_x)


                        if abs(kk_y-hh_y) > int(L/2):
 
                            yshift = L - abs(kk_y-hh_y)

                        else:

                            yshift = abs(kk_y-hh_y)			

                        if positions[ii] == 1 and positions[jj] == 1: #only add interaction if both spins are accepted

                            M[ii,jj] = np.sqrt((xshift)**2 + (yshift)**2)**(-alpha)

    # remove all rows and columns with no interactions
    rows_to_keep = np.where(np.sum(M, axis=1) > 0)[0]

    M = M[rows_to_keep][:, rows_to_keep]

    return M, xx_cord, yy_cord




def integration_Euler(S_init_ss, timesteps, dt, N, Jx_mat, Jy_mat, Jz_mat, hX_mat, hYmat, hZmat):

    Sx = np.zeros([N,timesteps+1])
    Sy = np.zeros([N,timesteps+1])
    Sz = np.zeros([N,timesteps+1])

    Sx[:,0] = S_init_ss[:,0]
    Sy[:,0] = S_init_ss[:,1]
    Sz[:,0] = S_init_ss[:,2]

    dSx_dt = np.zeros(N)
    dSy_dt = np.zeros(N)
    dSz_dt = np.zeros(N)

    dSx_dt_heun = np.zeros(N)
    dSy_dt_heun = np.zeros(N)
    dSz_dt_heun = np.zeros(N)


    for tt in range(0,timesteps):

        # Euler step

        dSx_dt[:] = - 2 * Sy[:,tt] * np.dot(Jz_mat, Sz[:,tt]) + 2 * Sz[:,tt] * np.dot(Jy_mat, Sy[:,tt]) + 2 * Sz[:,tt] * hY_mat[:] - 2 * Sy[:,tt] * hZ_mat[:] 
        
        dSy_dt[:] =   2 * Sx[:,tt] * np.dot(Jz_mat, Sz[:,tt]) - 2 * Sz[:,tt] * np.dot(Jx_mat, Sx[:,tt]) - 2 * Sz[:,tt] * hX_mat[:] + 2 * Sx[:,tt] * hZ_mat[:] 

        dSz_dt[:] = - 2 * Sx[:,tt] * np.dot(Jy_mat, Sy[:,tt]) + 2 * Sy[:,tt] * np.dot(Jx_mat, Sx[:,tt]) + 2 * Sy[:,tt] * hX_mat[:] - 2 * Sx[:,tt] * hY_mat[:]
        

        Sx[:,tt+1] = Sx[:,tt] + dSx_dt[:] * dt

        Sy[:,tt+1] = Sy[:,tt] + dSy_dt[:] * dt

        Sz[:,tt+1] = Sz[:,tt] + dSz_dt[:] * dt


        # Heun correction 
        dSx_dt_heun[:] = - 2 * Sy[:,tt+1] * np.dot(Jz_mat, Sz[:,tt+1]) + 2 * Sz[:,tt+1] * np.dot(Jy_mat, Sy[:,tt+1]) + 2 * Sz[:,tt+1] * hY_mat[:] - 2 * Sy[:,tt+1] * hZ_mat[:] 
        
        dSy_dt_heun[:] =   2 * Sx[:,tt+1] * np.dot(Jz_mat, Sz[:,tt+1]) - 2 * Sz[:,tt+1] * np.dot(Jx_mat, Sx[:,tt+1]) - 2 * Sz[:,tt+1] * hX_mat[:] + 2 * Sx[:,tt+1] * hZ_mat[:]

        dSz_dt_heun[:] = - 2 * Sx[:,tt+1] * np.dot(Jy_mat, Sy[:,tt+1]) + 2 * Sy[:,tt+1] * np.dot(Jx_mat, Sx[:,tt+1]) + 2 * Sy[:,tt+1] * hX_mat[:] - 2 * Sx[:,tt+1] * hY_mat[:]


        Sx[:,tt+1] = Sx[:,tt] + (dSx_dt[:] + dSx_dt_heun[:]) / 2.0 * dt

        Sy[:,tt+1] = Sy[:,tt] + (dSy_dt[:] + dSy_dt_heun[:]) / 2.0 * dt

        Sz[:,tt+1] = Sz[:,tt] + (dSz_dt[:] + dSz_dt_heun[:]) / 2.0 * dt


    return Sx, Sy, Sz


def integration_schemes(S_init_ss, timevec, N, Jx_mat, Jy_mat, Jz_mat, hX_mat, hY_mat, hZ_mat):

    y_init = np.zeros([3 * N])

    y_init[0:N] = S_init_ss[:,0]
    y_init[N:(2*N)]  = S_init_ss[:,1]
    y_init[(2*N):(3*N)] = S_init_ss[:,2]
    
    def func(t, y):
        
        dy = np.zeros([3 * N])

        dy[0:N] = - 2 * y[N:(2*N)] * np.dot(Jz_mat, y[(2*N):(3*N)]) + 2 * y[(2*N):(3*N)] * np.dot(Jy_mat, y[N:(2*N)] ) + 2 * y[(2*N):(3*N)] * hY_mat[:] - 2 * y[N:(2*N)] * hZ_mat[:] 
        
        dy[N:(2*N)]  =   2 * y[0:(N)] * np.dot(Jz_mat, y[(2*N):(3*N)]) - 2 * y[(2*N):(3*N)] * np.dot(Jx_mat, y[0:(N)]) - 2 * y[(2*N):(3*N)] * hX_mat[:] + 2 * y[0:(N)] * hZ_mat[:]

        dy[(2*N):(3*N)] = - 2 * y[0:(N)] * np.dot(Jy_mat, y[N:(2*N)] ) + 2 * y[N:(2*N)] * np.dot(Jx_mat, y[0:(N)]) + 2 * y[N:(2*N)] * hX_mat[:] - 2 * y[0:(N)] * hY_mat[:]

        return dy

    #y = scipy.integrate.odeint(func, y_init, timevec) # func(y,t) order needed, and need transpose below and y = sol
    sol = scipy.integrate.solve_ivp(func, [0, timevec[-1]], y_init, method = 'RK45', t_eval = timevec, rtol = 1e-6, atol = 1e-9)

    y = sol.y

    Sx = y[0:N,:]
    Sy = y[N:(2*N),:]
    Sz = y[(2*N):(3*N),:]
 
    return Sx, Sy, Sz




def dtwa(S_init, bb, ss, samples, timevec, N, Jx_mat, Jy_mat, Jz_mat, hX_mat, hY_mat, hZ_mat):

    np.random.seed(bb * samples + ss)   
    
    random_numbers = rand.randint(0,2,[N, 3]) # generate random integers in range [0,1] for each sample and site in 3 directions (x,y,z)

    S_sample_init = np.zeros([N,3])

    # draw random numbers for initial state if IC not aligned with axis 
    # (assumes translationally invariance) 

    if S_init[0,0] == 0.0:

        S_sample_init[:,0] = 2 * (random_numbers[:,0] - 0.5 * np.ones([N]))

    else:

        S_sample_init[:,0] = S_init[:,0] * np.ones([N])

    if S_init[0,1] == 0.0:

        S_sample_init[:,1] = 2 * (random_numbers[:,1] - 0.5 * np.ones([N]))

    else:
            
        S_sample_init[:,1] = S_init[:,1] * np.ones([N])

    if S_init[0,2] == 0.0:

        S_sample_init[:,2] = 2 * (random_numbers[:,2] - 0.5 * np.ones([N]))

    else:

        S_sample_init[:,2] = S_init[:,2] * np.ones([N])


    output = integration_schemes(S_sample_init, timevec, N, Jx_mat, Jy_mat, Jz_mat, hX_mat, hY_mat, hZ_mat)
   
    
    return output



def dtwa_sc(S_init, bb, ss, samples, timevec, N, Jx_mat, Jy_mat, Jz_mat, hX_mat, hY_mat, hZ_mat, save_loc, rtol, atol):

    np.random.seed(bb * samples + ss)   
    
    random_numbers = rand.randint(0,2,[N, 3]) # generate random integers in range [0,1] for each sample and site in 3 directions (x,y,z)

    S_sample_init = np.zeros([N,3])

    # draw random numbers for initial state if IC not aligned with axis 
    # (assumes translationally invariance) 

    if S_init[0,0] == 0.0:

        S_sample_init[:,0] = 2 * (random_numbers[:,0] - 0.5 * np.ones([N]))

    else:

        S_sample_init[:,0] = S_init[:,0] * np.ones([N])

    if S_init[0,1] == 0.0:

        S_sample_init[:,1] = 2 * (random_numbers[:,1] - 0.5 * np.ones([N]))

    else:
            
        S_sample_init[:,1] = S_init[:,1] * np.ones([N])

    if S_init[0,2] == 0.0:

        S_sample_init[:,2] = 2 * (random_numbers[:,2] - 0.5 * np.ones([N]))

    else:

        S_sample_init[:,2] = S_init[:,2] * np.ones([N])


    integration_schemes_sc(S_sample_init, timevec, N, Jx_mat, Jy_mat, Jz_mat, hX_mat, hY_mat, hZ_mat, save_loc, ss,rtol, atol)
   
    
def integration_schemes_sc(S_init_ss, timevec, N, Jx_mat, Jy_mat, Jz_mat, hX_mat, hY_mat, hZ_mat,save_loc,ss,rtol, atol):

    y_init = np.zeros([3 * N])
    y_init[0:N] = S_init_ss[:,0]
    y_init[N:(2*N)]  = S_init_ss[:,1]
    y_init[(2*N):(3*N)] = S_init_ss[:,2]
    
    def func(t, y):
        
        dy = np.zeros([3 * N])

        dy[0:N] = - 2 * y[N:(2*N)] * np.dot(Jz_mat, y[(2*N):(3*N)]) + 2 * y[(2*N):(3*N)] * np.dot(Jy_mat, y[N:(2*N)] ) + 2 * y[(2*N):(3*N)] * hY_mat[:] - 2 * y[N:(2*N)] * hZ_mat[:] 
        
        dy[N:(2*N)]  =   2 * y[0:(N)] * np.dot(Jz_mat, y[(2*N):(3*N)]) - 2 * y[(2*N):(3*N)] * np.dot(Jx_mat, y[0:(N)]) - 2 * y[(2*N):(3*N)] * hX_mat[:] + 2 * y[0:(N)] * hZ_mat[:]

        dy[(2*N):(3*N)] = - 2 * y[0:(N)] * np.dot(Jy_mat, y[N:(2*N)] ) + 2 * y[N:(2*N)] * np.dot(Jx_mat, y[0:(N)]) + 2 * y[N:(2*N)] * hX_mat[:] - 2 * y[0:(N)] * hY_mat[:]

        return dy

    #y = scipy.integrate.odeint(func, y_init, timevec) # func(y,t) order needed, and need transpose below and y = sol
    sol = scipy.integrate.solve_ivp(func, [0, timevec[-1]], y_init, method = 'RK45', t_eval = timevec, rtol = rtol, atol = atol)

    y = sol.y

    Sx = y[0:N,:]
    Sy = y[N:(2*N),:]
    Sz = y[(2*N):(3*N),:]

    
    np.save(save_loc + "Sx_sample_" + str(ss) + ".npy", Sx) 
    np.save(save_loc + "Sy_sample_" + str(ss) + ".npy", Sy) 
    np.save(save_loc + "Sz_sample_" + str(ss) + ".npy", Sz) 



def poisson_filter(lambda0, xMax, yMax, lower_threshold, upper_threshold):

    #Simulation window parameters
    xMin=0
    yMin=0
    plotting = "True"  # Set to "True" to enable plotting
    xDelta=xMax-xMin;yDelta=yMax-yMin; #rectangle dimensions
    areaTotal=xDelta*yDelta;
    
    #Point process parameters
    # lambda0 is intensity (ie mean density) of the Poisson process
    
    #Simulate Poisson point process
    numbPoints = scipy.stats.poisson( lambda0*areaTotal ).rvs()#Poisson number of points
    xx = xDelta*scipy.stats.uniform.rvs(0,1,((numbPoints,1)))+xMin#x coordinates of Poisson points
    yy = yDelta*scipy.stats.uniform.rvs(0,1,((numbPoints,1)))+yMin#y coordinates of Poisson points

    # Flatten coordinate arrays to ensure they're 1D
    x_coords = xx.flatten()
    y_coords = yy.flatten()
    # Calculate distance matrix between all pairs of points
    distances = np.sqrt((x_coords[:, None] - x_coords[None, :])**2 + (y_coords[:, None] - y_coords[None, :])**2)
    # Set diagonal to np.inf or large to ignore self-distances (distance from point to itself)
    np.fill_diagonal(distances, 10000) 
    # Calculate the distance to the nearest neighbor for each point
    nearest_distances = np.min(distances, axis=1)

    # summation over distances 
    J_i = np.sum(1/distances**3, axis=1)  # J_i is the sum of inverse distances for each point
    # effective mean field distance
    reff = J_i**(-1/3)  # effective mean field distance
    reff_mean = np.mean(reff)  # average effective mean field distance

    # Filter the spins (dimers) below some cut-off distance
    distances_filt = np.copy(distances)  
    for ii in range(0,np.size(reff)):
        if reff[ii] < reff_mean * lower_threshold:  # lower_threshold is a fraction of the mean reff
            distances_filt[ii,:] = 10000 * np.ones(np.size(distances_filt[ii,:]))  # Set distances to infinity if reff is less than 0.2
    nearest_distance_filt = np.min(distances_filt, axis=1)

    # Filter the spins below some cut-off distance
    distances_filt_upper = np.copy(distances_filt) 
    for ii in range(0,np.size(reff)):
        if reff[ii] > reff_mean * upper_threshold:  # upper_threshold is a fraction of the mean reff
            distances_filt_upper[ii,:] = 10000 * np.ones(np.size(distances_filt_upper[ii,:]))  # Set distances to infinity if reff is less than 0.2
    nearest_distance_filt_alt = np.min(distances_filt_upper, axis=1)

    ######## Calculate effective distance
    Jalt_i = np.sum(1/distances_filt_upper**3, axis=1)  # J_i is the sum of inverse distances for each point
    # effective mean field distance
    reff_alt = Jalt_i**(-1/3)  # effective mean field distance

    ####### Create an interaction matrix with only the filtered sites
    distances_filt = np.copy(distances)  # Initialize Jeff with J_i values
    mat_ind = []
    #select all valid indices (remaining spins)
    for ii in range(0,np.size(reff)):
        if np.mean(distances_filt_upper[ii,:]) < 10000:
            mat_ind += [ii]

    distances_new = np.zeros([np.size(mat_ind),np.size(mat_ind)])  
    for ii in range(0,np.size(reff)):
        for jj in range(0,np.size(reff)):
            if ii in mat_ind and jj in mat_ind:
                #find position of ii in mat_ind
                ind_ii = mat_ind.index(ii)
                ind_jj = mat_ind.index(jj)
                distances_new[ind_ii,ind_jj] = distances_filt_upper[ii,jj] 

    xx_new = xx[mat_ind] 
    yy_new = yy[mat_ind] 

    J_new = 1.0/distances_new**3

    ####### Plotting the results
    if plotting == "True":

        plt.scatter(xx,yy, edgecolor='b', facecolor='none', alpha=0.5 )
        plt.scatter(xx_new,yy_new, marker="x", color='r',  alpha=0.5 )
        plt.xlabel("x"); plt.ylabel("y")
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.hist(reff, bins=30, density=True, alpha=0.6, color ='blue', edgecolor='black', label='No removal of sites')
        plt.hist(reff_alt[reff_alt<10], bins=20, density=True, alpha=0.6, color ='red', edgecolor='black', label=r'Filtered above and below')
        plt.title(r"Distribution of Effective Nearest Neighbour Distances: $1/r_{i ,\rm eff}^3= \sum_j 1/r_{ij}^3$")
        plt.xlabel(r"Effective Distance $r_{\rm eff}$")
        plt.ylabel("Density (arb. units)")
        plt.legend()
        plt.grid()
        plt.xlim(0,1.3)
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.hist(nearest_distances, bins=20, density=True, alpha=0.6, color='blue', edgecolor='black', label='No removal of sites')
        plt.hist(nearest_distance_filt_alt[nearest_distance_filt_alt<10], bins=10, density=True, alpha=0.6, color='red', edgecolor='black', label=r'Filtered above and below')
        plt.title("Distribution of Actual Nearest Neighbor Distances")
        plt.xlabel("Distance to Nearest Neighbor")
        plt.ylabel("Density")

        # For a Poisson process, nearest neighbor distances follow a Rayleigh-like distribution
        # The theoretical density is: f(r) = 2πλr * exp(-πλr²)
        # where λ is the intensity
        r_theory = np.linspace(0, 2, 1000)
        theoretical_density = 2 * np.pi * lambda0 * r_theory * np.exp(-np.pi * lambda0 * r_theory**2)
        plt.plot(r_theory, theoretical_density, 'r-', linewidth=2, label=f'Theoretical (λ={lambda0})')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.xlim(0, 1.5)
        plt.show()

        Jdist = np.sum(1.0/distances_new**3,1)

        #plot histogram of Jdist
        plt.figure(figsize=(10, 6))
        plt.hist(Jdist, bins=30, alpha=0.6, color='blue', edgecolor='black', label='J filtered')         
        plt.hist(J_i, bins=300, alpha=0.6, color='red', edgecolor='black', label='J original')       
        plt.title("Distribution of Jdist")
        plt.xlabel("Jdist")
        plt.ylabel("Density")               
        plt.legend()
        plt.grid()
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.hist(Jdist, bins=30, alpha=0.6, color='blue', edgecolor='black', label='J filtered')         
        plt.hist(J_i[J_i < 1000], bins=300, alpha=0.6, color='red', edgecolor='black', label='J original')       
        plt.title("Distribution of J")
        plt.xlabel("J")
        plt.ylabel("P(J) number of spins with given J")               
        plt.legend()
        plt.grid()
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.hist(Jdist, bins=15, alpha=0.6, color='blue', edgecolor='black', label='J filtered, num spins =' + str(np.size(Jdist)))         
        plt.hist(J_i[J_i < 1000], bins=300, alpha=0.6, color='red', edgecolor='black', label='J original, num spins =' + str(np.size(J_i)))       
        plt.title("Distribution of J")
        plt.xlabel("J")
        plt.ylabel("P(J) number of spins with given J")      
        plt.xlim(0,200)         
        plt.legend()
        plt.grid()
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.hist(Jdist, bins=15, alpha=0.6, color='blue', edgecolor='black', label='J filtered, num spins =' + str(np.size(Jdist)),log = True)         
        plt.hist(J_i[J_i < 1000], bins=30, alpha=0.6, color='red', edgecolor='black', label='J original, num spins =' + str(np.size(J_i)),log = True)       
        plt.title("Distribution of J")
        plt.xlabel("J")
        plt.ylabel("log P(J) number of spins with given J")      
        plt.xlim(0,1000)         
        plt.legend()
        plt.grid()
        plt.show()

    return J_new, distances_new, distances, xx_new, yy_new, xx, yy


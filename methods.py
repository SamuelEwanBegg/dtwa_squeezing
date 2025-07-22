import numpy as np
import numpy.random as rand
import copy
import scipy.integrate

def gen_matrices(N, alpha):

    M = np.zeros([N,N])

    for kk in range(0,N):
        
        for hh in range(0,N):
        
            if hh != kk:

                M[kk,hh] = np.abs(kk - hh)**(-alpha)

    return M



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



def dtwa_sc(S_init, bb, ss, samples, timevec, N, Jx_mat, Jy_mat, Jz_mat, hX_mat, hY_mat, hZ_mat, save_loc):

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


    integration_schemes_sc(S_sample_init, timevec, N, Jx_mat, Jy_mat, Jz_mat, hX_mat, hY_mat, hZ_mat, save_loc, ss)
   
    
def integration_schemes_sc(S_init_ss, timevec, N, Jx_mat, Jy_mat, Jz_mat, hX_mat, hY_mat, hZ_mat,save_loc,ss):

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

    
    np.save(save_loc + "Sx_sample_" + str(ss) + ".npy", Sx) 
    np.save(save_loc + "Sy_sample_" + str(ss) + ".npy", Sy) 
    np.save(save_loc + "Sz_sample_" + str(ss) + ".npy", Sz) 




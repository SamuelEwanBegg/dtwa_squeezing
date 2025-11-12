print(str(startTime),'Commence code')

#save file location
loc = "/Users/samuelbegg/Documents/Projects/EP_sensing/sc_results/result/"
temp_save_loc = loc

import numpy as np
import methods 
import numpy.random as rand
from joblib import Parallel, delayed, dump, load
import copy
import scipy.integrate
from datetime import datetime
startTime = datetime.now()

import os
print(f"SLURM_CPUS_PER_TASK: {os.environ.get('SLURM_CPUS_PER_TASK')}")
print(f"SLURM_JOB_CPUS_PER_NODE: {os.environ.get('SLURM_JOB_CPUS_PER_NODE')}")

print(str(startTime),'Commence code')

#save file location
loc = "/home/dal993192/dtwa_squeezing/dtwa_squeezing/results/bilayer/LOCATION1/LOCATION2/"
temp_save_loc = "/home/dal993192/scratch/bilayer/LOCATION1/LOCATION2/"

Jx_in = 1.0 # interaction terms
Jy_in = 1.0
Jz_in = 1.0
Jx_out = 1.0
Jy_out = 1.0
Jz_out = 0.0
hX = 0.0 # magnetic fields are in Pauli matrix convention (double compared to spin convention)
hY = 0.0
delta_hZ = YY # fractional magnetic field perturbation (see definition below)
alpha = 3.0 # power-law
az = 10.0 # interlayer distance
nu = 0.0 # frame/state angle 

# Simulation parameters
L = XX
Nval = L**2
samples = 32  #samples per batch
batches = 20   #int(total_samples / samples)
total_samples = samples * batches
timesteps = 1000
rtol = 10**(-3)
atol = 10**(-5)
num_cores = -1


# layer resolved field signs
hX_A = 0.0
hX_B = 0.0
hY_A = 0.0
hY_B = 0.0
#determine hZ by the average interactions (perturbed around)

# Generate interaction matrices for in-plane interactions
Jx_mat_in = Jx_in * methods.gen_matrices_2D_pbc(Nval, alpha)
Jy_mat_in = Jy_in * methods.gen_matrices_2D_pbc(Nval, alpha)
Jz_mat_in = Jz_in * methods.gen_matrices_2D_pbc(Nval, alpha)

# Generate interaction matrices for out-plane interactions
Jx_mat_out = Jx_out * methods.gen_matrices_2D_pbc_bilayer(Nval, alpha, az)
Jy_mat_out = Jy_out * methods.gen_matrices_2D_pbc_bilayer(Nval, alpha, az)
Jz_mat_out = Jz_out * methods.gen_matrices_2D_pbc_bilayer(Nval, alpha, az)

av_inter = np.mean(Jx_mat_out[0,:]) 
hZ =   (0.5 * Nval * av_inter)  * (1.0 + delta_hZ)  # factor of 1/2 is due to Pauli definition
hZ_A = hZ 
hZ_B = - hZ

print("av interaction", np.round(av_inter,5), "delta hZ", np.round(delta_hZ,5))

# Construct a unique ID based on simulation parameters
param_id = f"N{Nval}_alpha{alpha:.2f}_Jx_in{Jx_in:.2f}_Jy_in{Jy_in:.2f}_Jz_in{Jz_in:.2f}_Jx_out{Jx_out:.2f}_Jy_out{Jy_out:.2f}_Jz_out{Jz_out:.2f}_hX{hX:.2f}_hY{hY:.2f}_hZ{hZ:.5f}_delta_hZ{delta_hZ:.5f}_aZ{az:.3f}_total_samples{total_samples}"
print(param_id)

Vavg = 1.0/Nval**2 * np.sum(Jx_mat_out)  
total_time = 10.0 / (Vavg * Nval) 
dt = total_time / float(timesteps)

Jx_mat = np.zeros([2*Nval, 2*Nval])
Jy_mat = np.zeros([2*Nval, 2*Nval])
Jz_mat = np.zeros([2*Nval, 2*Nval])

Jx_mat[0:Nval, 0:Nval] = 0.5 * Jx_mat_in
Jx_mat[Nval:, Nval:]   = 0.5 * Jx_mat_in
Jx_mat[0:Nval, Nval:]  = 0.5 * Jx_mat_out
Jx_mat[Nval:, 0:Nval]  = 0.5 * Jx_mat_out

Jy_mat[0:Nval, 0:Nval] = 0.5 * Jy_mat_in
Jy_mat[Nval:, Nval:]  =  0.5 * Jy_mat_in
Jy_mat[0:Nval, Nval:]  = 0.5 * Jy_mat_out
Jy_mat[Nval:, 0:Nval]  = 0.5 * Jy_mat_out

Jz_mat[0:Nval, 0:Nval] = 0.5 * Jz_mat_in
Jz_mat[Nval:, Nval:]   = 0.5 * Jz_mat_in
Jz_mat[0:Nval, Nval:]  = 0.5 * Jz_mat_out
Jz_mat[Nval:, 0:Nval]  = 0.5 * Jz_mat_out


hX_mat_upper =  hX_A * np.ones(Nval)
hY_mat_upper =  hY_A * np.ones(Nval)
hZ_mat_upper =  hZ_A * np.ones(Nval)

hX_mat_lower = hX_B * np.ones(Nval)
hY_mat_lower = hY_B * np.ones(Nval)
hZ_mat_lower = hZ_B * np.ones(Nval)

hX_mat = np.ones(2*Nval)
hY_mat = np.ones(2*Nval)
hZ_mat = np.ones(2*Nval)

hX_mat[0:Nval] = hX_mat_upper
hX_mat[Nval:] = hX_mat_lower

hY_mat[0:Nval] = hY_mat_upper
hY_mat[Nval:] = hY_mat_lower

hZ_mat[0:Nval] = hZ_mat_upper
hZ_mat[Nval:] = hZ_mat_lower


# File paths with unique suffix
Jx_path = f"{temp_save_loc}/pkl_store/Jx_mat_{param_id}.pkl"
Jy_path = f"{temp_save_loc}/pkl_store/Jy_mat_{param_id}.pkl"
Jz_path = f"{temp_save_loc}/pkl_store/Jz_mat_{param_id}.pkl"
hX_path = f"{temp_save_loc}/pkl_store/hX_mat_{param_id}.pkl"
hY_path = f"{temp_save_loc}/pkl_store/hY_mat_{param_id}.pkl"
hZ_path = f"{temp_save_loc}/pkl_store/hZ_mat_{param_id}.pkl"

# Save
def always_dump(obj, filename):
    dump(obj, filename)

always_dump(Jx_mat, Jx_path)
always_dump(Jy_mat, Jy_path)
always_dump(Jz_mat, Jz_path)
always_dump(hX_mat, hX_path)
always_dump(hY_mat, hY_path)
always_dump(hZ_mat, hZ_path)

# Load using mmap_mode for shared access
Jx_mat = load(Jx_path, mmap_mode='r')
Jy_mat = load(Jy_path, mmap_mode='r')
Jz_mat = load(Jz_path, mmap_mode='r')
hX_mat = load(hX_path, mmap_mode='r')
hY_mat = load(hY_path, mmap_mode='r')
hZ_mat = load(hZ_path, mmap_mode='r')

# define for later use
timevec = dt * np.arange(0,timesteps+1)

# Classical position of spin. Note that this is the initial state prior to rotation by nu
S_init = np.zeros([2*Nval,3])

#Upper layer
S_init[0:Nval,0] = 0.0    # X position
S_init[0:Nval,1] = 0.0    # Y position
S_init[0:Nval,2] = 1.0    # Z position

#Lower layer
S_init[Nval:,0] = 0.0     # X position
S_init[Nval:,1] = 0.0     # Y position
S_init[Nval:,2] = - 1.0   # Z position


# initialize layer A and B varianbles for running mean and variance
Sx_mean_A =  np.zeros([timesteps+1])
Sy_mean_A =  np.zeros([timesteps+1])
Sz_mean_A =  np.zeros([timesteps+1])

Sx_M2_A =  np.zeros([timesteps+1])
Sy_M2_A =  np.zeros([timesteps+1])
Sz_M2_A =  np.zeros([timesteps+1])

nSx_A = 0
nSy_A = 0
nSz_A = 0

Sx_mean_B =  np.zeros([timesteps+1])
Sy_mean_B =  np.zeros([timesteps+1])
Sz_mean_B =  np.zeros([timesteps+1])

Sx_M2_B =  np.zeros([timesteps+1])
Sy_M2_B =  np.zeros([timesteps+1])
Sz_M2_B =  np.zeros([timesteps+1])

nSx_B = 0
nSy_B = 0
nSz_B = 0

# initialize Signal = Sz_A - Sz_B
Signal_mean = np.zeros([timesteps+1])
Signal_M2 = np.zeros([timesteps+1])
nSignal = 0

# initialize correlators
nCorrX_A = 0
nCorrY_B = 0
nCorrXY_AB = 0
nVar = 0

CorrX_A_mean =  np.zeros([timesteps+1])
CorrY_B_mean =  np.zeros([timesteps+1]) 
CorrXY_AB_mean = np.zeros([timesteps+1])
Var_mean = np.zeros([timesteps+1])

CorrX_A_M2 =  np.zeros(timesteps+1)
CorrY_B_M2 =  np.zeros(timesteps+1) 
CorrXY_AB_M2 = np.zeros(timesteps+1)
Var_M2 = np.zeros(timesteps+1)

# loop over batches
for bb in range(0,batches):

    print(bb)

    Parallel(n_jobs=num_cores)(delayed(methods.dtwa_sc_bilayer_EP)(S_init, bb, ss, samples, timevec, 2 * Nval, Jx_mat, Jy_mat, Jz_mat, hX_mat, hY_mat, hZ_mat, nu, temp_save_loc, rtol, atol) for ss in range(0,samples)) 

    if bb == batches - 1:
        os.remove(Jx_path)
        os.remove(Jy_path)
        os.remove(Jz_path)
        os.remove(hX_path)
        os.remove(hY_path)
        os.remove(hZ_path)

    # initialize matrices 
    Sx_av_A =  np.zeros(timesteps+1)
    Sy_av_A =  np.zeros(timesteps+1)
    Sz_av_A =  np.zeros(timesteps+1)
    Sx_av_B =  np.zeros(timesteps+1)
    Sy_av_B =  np.zeros(timesteps+1)
    Sz_av_B =  np.zeros(timesteps+1)
    Signal_av = np.zeros(timesteps+1)
    CorrX_A_av =  np.zeros(timesteps+1)
    CorrY_B_av =  np.zeros(timesteps+1) 
    CorrXY_AB_av = np.zeros(timesteps+1)

    for ss in range(0,samples):
        sx_sample = np.load(temp_save_loc + "Sx_sample_" + str(ss) + ".npy")
        sy_sample = np.load(temp_save_loc + "Sy_sample_" + str(ss) + ".npy")
        sz_sample = np.load(temp_save_loc + "Sz_sample_" + str(ss) + ".npy")

        Sx_sum_t_A = np.sum(sx_sample[0:Nval],0)    
        Sy_sum_t_A = np.sum(sy_sample[0:Nval],0)    
        Sz_sum_t_A = np.sum(sz_sample[0:Nval],0)   

        Sx_av_A += 1.0/ samples * Sx_sum_t_A
        Sy_av_A += 1.0/ samples * Sy_sum_t_A
        Sz_av_A += 1.0/ samples * Sz_sum_t_A

        Sx_sum_t_B = np.sum(sx_sample[Nval:],0)    
        Sy_sum_t_B = np.sum(sy_sample[Nval:],0)    
        Sz_sum_t_B = np.sum(sz_sample[Nval:],0)   

        Sx_av_B += 1.0/ samples * Sx_sum_t_B
        Sy_av_B += 1.0/ samples * Sy_sum_t_B
        Sz_av_B += 1.0/ samples * Sz_sum_t_B

        CorrX_A_av  += 1.0 / samples * Sx_sum_t_A**2
        CorrY_B_av  += 1.0 / samples * Sy_sum_t_B**2  
        CorrXY_AB_av += 1.0 / samples * Sx_sum_t_A * Sy_sum_t_B

        Signal_av += 1.0 / samples * (Sz_sum_t_A - Sz_sum_t_B)

        del sx_sample, sy_sample, sz_sample
        
        if bb == batches - 1:
            os.remove(temp_save_loc + "Sx_sample_" + str(ss) + ".npy")
            os.remove(temp_save_loc + "Sy_sample_" + str(ss) + ".npy")
            os.remove(temp_save_loc + "Sz_sample_" + str(ss) + ".npy")


    # Calculate the running mean and variance for magnetization and correlators
    nSx_A, Sx_mean_A, Sx_std_A, Sx_M2_A = methods.update_stats(Sx_av_A, nSx_A, Sx_mean_A, Sx_M2_A) 
    nSy_A, Sy_mean_A, Sy_std_A, Sy_M2_A = methods.update_stats(Sy_av_A, nSy_A, Sy_mean_A, Sy_M2_A) 
    nSz_A, Sz_mean_A, Sz_std_A, Sz_M2_A = methods.update_stats(Sz_av_A, nSz_A, Sz_mean_A, Sz_M2_A) 

    nSx_B, Sx_mean_B, Sx_std_B, Sx_M2_B = methods.update_stats(Sx_av_B, nSx_B, Sx_mean_B, Sx_M2_B) 
    nSy_B, Sy_mean_B, Sy_std_B, Sy_M2_B = methods.update_stats(Sy_av_B, nSy_B, Sy_mean_B, Sy_M2_B) 
    nSz_B, Sz_mean_B, Sz_std_B, Sz_M2_B = methods.update_stats(Sz_av_B, nSz_B, Sz_mean_B, Sz_M2_B) 

    nCorrX_A, CorrX_A_mean, CorrX_A_std, CorrX_A_M2 = methods.update_stats(CorrX_A_av, nCorrX_A, CorrX_A_mean, CorrX_A_M2) 
    nCorrY_B, CorrY_B_mean, CorrY_B_std, CorrY_B_M2 = methods.update_stats(CorrY_B_av, nCorrY_B, CorrY_B_mean, CorrY_B_M2) 
    nCorrXY_AB, CorrXY_AB_mean, CorrXY_AB_std, CorrXY_AB_M2 = methods.update_stats(CorrXY_AB_av, nCorrXY_AB, CorrXY_AB_mean, CorrXY_AB_M2)
    nSignal, Signal_mean, Signal_std, Signal_M2 = methods.update_stats(Signal_av, nSignal, Signal_mean, Signal_M2)

    Var_av = CorrX_A_mean + CorrY_B_mean - 2 * CorrXY_AB_mean - (Sx_mean_A**2 + Sy_mean_B**2 - 2 * Sx_mean_A * Sy_mean_B)
    nVar, Var_mean, Var_std, Var_M2 = methods.update_stats(Var_av, nVar, Var_mean, Var_M2)


endTime = datetime.now()
print(str(endTime - startTime),'Run time')

Signal = Sz_mean_A - Sz_mean_B

# save A variables
np.save(loc + "Sx_A.npy",Sx_mean_A)
np.save(loc + "Sy_A.npy",Sy_mean_A)
np.save(loc + "Sz_A.npy",Sz_mean_A)
np.save(loc + "Sx_A_std.npy",Sx_std_A)
np.save(loc + "Sy_A_std.npy",Sy_std_A)
np.save(loc + "Sz_A_std.npy",Sz_std_A)

# save B variables
np.save(loc + "Sx_B.npy",Sx_mean_B)
np.save(loc + "Sy_B.npy",Sy_mean_B)
np.save(loc + "Sz_B.npy",Sz_mean_B)
np.save(loc + "Sx_B_std.npy",Sx_std_B)
np.save(loc + "Sy_B_std.npy",Sy_std_B)
np.save(loc + "Sz_B_std.npy",Sz_std_B)

# save signal variable
np.save(loc + "SzA_minus_SzB.npy",Signal_mean)

# save correlators
np.save(loc + "CorrX_A.npy",CorrX_A_mean)
np.save(loc + "CorrX_A_std.npy",CorrX_A_std)
np.save(loc + "CorrY_B.npy",CorrY_B_mean)
np.save(loc + "CorrY_B_std.npy",CorrY_B_std)
np.save(loc + "CorrXY_AB.npy",CorrXY_AB_mean)
np.save(loc + "CorrXY_AB_std.npy",CorrXY_AB_std)
np.save(loc + "Vmin.npy",Var_mean)
np.save(loc + "Vmin_std.npy",Var_std)

# save time vector and Vavg
np.save(loc + "Nval.npy",Nval)
np.save(loc + "timevec.npy",timevec)
np.save(loc + "Vavg.npy",Vavg)



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 02:10:13 2024

@author: manuelfranciscocatacorarios
"""

import numpy as np
from typing import Callable
from free_solution import phi_free, analytical_phi_0
from cc_constants import CC_Constants
from frescox_waves_cc import Frescox_Inelastic_Wrapper
import math
from rescaling_functions_cc import rescaling_function_factor, find_smallest_nonzero, Gamow_factor
from utility_functions_cc import generate_second_derivative_matrix_9point

class Basis_CC(Frescox_Inelastic_Wrapper):
    def __init__(self, 
                 mass_t: float,
                 charge_t : float,
                 spin_t_gs : float,
                 spin_t_ex : float,
                 mass_p : float, 
                 charge_p : float,
                 spin_p : float,
                 E_lab : float,
                 J_tot_max : float,
                 J_tot_min : float,
                 coulomb_r : float,
                 reaction_name : str,
                 real_def : float,
                 n_basis_RBM : int,
                 E_states : np.array,
                 xgrid : np.array,
                 imag_def : float = 0,
                 ):
        super().__init__(mass_t, charge_t, spin_t_gs, spin_t_ex, mass_p, charge_p, spin_p, E_lab,
                         J_tot_max, J_tot_min, coulomb_r, reaction_name, real_def, E_states, xgrid, imag_def)
        
        self.l_max = math.ceil(J_tot_max + spin_p + max(spin_t_ex,spin_t_gs))
        constants = CC_Constants(mass_t, mass_p, E_lab, E_states)
        self.momentum_ar = constants.k()        
        self.n_basis_RBM = n_basis_RBM
        
    
    
    def free_waves(self):
        free_waves_rescaled = []
        xgrid_copy = self.xgrid.copy()
        if xgrid_copy[0] == 0:
            xgrid_copy[0] = 1e-12
            
        for l in range(self.l_max+1):
            wave = phi_free(self.momentum_ar[0]*xgrid_copy, l)
            scaled, factor = rescaling_function_factor(wave, l, xgrid_copy)
            free_waves_rescaled.append(scaled)
        return free_waves_rescaled
    
   
    def SVD_extract_frescox_asymp(self, 
                                  training_array_RBM : np.array):
        
        D2 = generate_second_derivative_matrix_9point(self.xgrid)
        wave_dict, grouped_keys_dict = self.frescox_run_inelastic_rescaled_waves(training_array_RBM)
        phi0_arr = self.free_waves()
        run_num = len(training_array_RBM)
        
        SVD_per_channel = {}
        SVD_per_channel_d2 = {}
        phi0_dict = {}
        phi0_d2_dict = {}
        singular_values = []
        
        for m in (grouped_keys_dict):
            subset_channels = grouped_keys_dict.get(m)
            
            for j in range(len(subset_channels)):
                l = int(subset_channels[j][2])
                wave_set = []
                
                if (subset_channels[j][1])==1.0:
                    for i in range(run_num):
                        wave_scaled = wave_dict.get(subset_channels[j])[i]
                        wave_set.append(wave_scaled)
                    phi0 = phi0_arr[l]
                    np_wave_set = (np.array(wave_set) - phi0).T
                    #print(np.shape(np_wave_set))
                    U, S, Vt = np.linalg.svd(np_wave_set, full_matrices=False)
                    basis = U[:, :self.n_basis_RBM]
                    singular_values.append(S[:self.n_basis_RBM])
                    d2 = D2 @ basis
                    #d20 = D2 @ phi0
                    
                    SVD_per_channel[subset_channels[j]] = basis
                    SVD_per_channel_d2[subset_channels[j]] = d2
                    
                    phi0_d2_dict[(l)] = D2 @ phi0_arr[l]
                    phi0_dict[(l)] = phi0_arr[l]
                    print(np.shape(basis),'el')
               
                else:
                    for i in range(run_num):
                        wave_scaled = wave_dict.get(subset_channels[j])[i]
                        wave_set.append(wave_scaled)
                    np_wave_set = (np.array(wave_set)).T
                   # print(np.shape(np_wave_set))
                    U, S, Vt = np.linalg.svd(np_wave_set, full_matrices=False)
                    basis = U[:, :self.n_basis_RBM]
                    singular_values.append(S[:self.n_basis_RBM])
                    d2 = D2 @ basis
                    SVD_per_channel[subset_channels[j]] = basis
                    SVD_per_channel_d2[subset_channels[j]] = d2
                    print(np.shape(basis),'in')
                    
        return SVD_per_channel, SVD_per_channel_d2, phi0_dict, phi0_d2_dict, grouped_keys_dict, D2, singular_values
                    
                    
                    

                    
                

        
    
        
    
    




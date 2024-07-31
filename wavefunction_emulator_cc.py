#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 19:54:59 2024

@author: manuelfranciscocatacorarios
"""
import numpy as np
from basis import Basis_CC
from cc_constants import CC_Constants
from utility_functions_cc import centrifugal
from interaction_eim_cc import InteractionEIM_CC
from typing import Callable
from cc_matrix import CC_matrix
from ws_radial_functions import WS_deformed_real_interaction, WS_spherical_interaction


class Wavefunction_Emulator_CC(Basis_CC):
    drop = 4
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
                 n_basis_sph_EIM: int,                                                  # number of basis in EIM
                 n_basis_def_EIM: int,
                 n_alpha_sph: int,                                                          # number of parameters alpha in interaction
                 n_alpha_def: int,
                 training_array_RBM : np.array,
                 training_array_EIM: np.array,# training array info
                 E_states : np.array,
                 xgrid : np.array,
                 delta_def : float,
                 r_array : np.array,
                 coordinate_space_potential_spherical: Callable[[float, np.array],float],    # radial potential function V(r,alpha)
                 coordinate_space_potential_deformed: Callable[[float, np.array],float],
                 deformed_complex: bool = False,
                 is_complex: bool = False,
                 imag_def : float = 0,
                 match_points: np.array = None):
        
        
        super().__init__(mass_t, charge_t, spin_t_gs, spin_t_ex, mass_p, charge_p, spin_p, E_lab,
                         J_tot_max, J_tot_min, coulomb_r, reaction_name, real_def, n_basis_RBM, E_states, xgrid, imag_def)
        
        self.training_array_RBM = training_array_RBM
        self.SVD_grouped_dict, self.SVD_grouped_d2_dict, self.phi0_dict, self.phi0_d2_dict, self.grouped_keys_dict, self.D2, self.singular_values = self.SVD_extract_frescox_asymp(self.training_array_RBM)
        
        constants = CC_Constants(mass_t, mass_p, E_lab, E_states)
        self.momentum_ar = constants.k() 
        self.hbar2_2mu = constants.hbar2_reduced()
        
        self.centrifugal_arr = [centrifugal(l, self.xgrid) for l in range(self.l_max+1)]
        self.E_gs, self.E_ex = constants.E_lab_to_COM()
        
        self.EIM_WS = InteractionEIM_CC(coordinate_space_potential_spherical, coordinate_space_potential_deformed, n_alpha_sph, n_alpha_def, delta_def, training_array_EIM, r_array, deformed_complex, is_complex, n_basis_sph_EIM, n_basis_def_EIM, match_points)
        self.EIM_basis_sph, self.EIM_basis_def = self.EIM_WS.basis_functions(self.xgrid)
        
        self.integrated_subchannels_right, self.integrated_subchannels_left = self.compute_all_integrals()
        
        
        
    
        
    def phi0_integrals(self,
                       subsets_arr : np.array,
                       matrix_subset : np.array):
    
        '''
    
        '''
    
        psiv_V_def_channel = []
        l_0 = int(subsets_arr[0][2])
        phi0 = self.phi0_dict.get(l_0)[self.drop:-self.drop]
        d20 = self.phi0_d2_dict.get(l_0)[self.drop:-self.drop]
        centri = (self.centrifugal_arr[l_0])[self.drop:-self.drop]
        
        F_0 = -self.hbar2_2mu * d20 + self.hbar2_2mu*centri*phi0 - self.E_gs*phi0
        psi1_F_0 = self.SVD_grouped_dict.get(subsets_arr[0])[self.drop:-self.drop].T @ F_0
        
        F_V0 = self.EIM_basis_sph[self.drop:-self.drop].T*phi0
        psi1_V_sph =  self.SVD_grouped_dict.get(subsets_arr[0])[self.drop:-self.drop].T @ F_V0.T
        
        for v in range(len(subsets_arr)):
            F_V_v = self.EIM_basis_def[self.drop:-self.drop].T*phi0
            psiv_V_def =  matrix_subset[v][0]*self.SVD_grouped_dict.get(subsets_arr[v])[self.drop:-self.drop].T @ F_V_v.T
            psiv_V_def_channel.append(psiv_V_def)
                
        return psi1_F_0, psi1_V_sph, psiv_V_def_channel
    
    def psi_F0_psi_integrals(self,
                             SVD_vect: np.array,
                             SVD_d2_vect : np.array,
                             l_val : int,
                             E_state):
        
        '''
        
        '''
        
        F_0 = -self.hbar2_2mu * SVD_d2_vect.T + self.hbar2_2mu*self.centrifugal_arr[l_val]*SVD_vect.T - E_state*SVD_vect.T
        
        #print(np.shape(self.centrifugal_arr[l_val]*SVD_vect.T))
        
        psi_F0_psi = SVD_vect[self.drop:-self.drop].T @ F_0.T[self.drop:-self.drop]
        
        return psi_F0_psi
    
    def psi_Vsph_psi_integrals(self,
                            SVD_vect: np.array):
        
        '''
        
        '''
        
        SVD_axis = SVD_vect[:, :, np.newaxis]
        EIM_sph_basis_axis = self.EIM_basis_sph[:,np.newaxis , : ]
        
        outer_prod = EIM_sph_basis_axis * SVD_axis
        
        psi_Vsph_psi = np.einsum('ij,jlm->ilm', SVD_vect[self.drop:-self.drop].T, outer_prod[self.drop:-self.drop])
        
        return psi_Vsph_psi
        
    
    def psi_Vdef_psi_integrals(self,
                            SVD_vect1: np.array,
                            SVD_vect2: np.array ):
        
        '''
        
        '''
        
        SVD_axis = SVD_vect1[:, :, np.newaxis]
        EIM_def_basis_axis = self.EIM_basis_def[:,np.newaxis , : ]
        
        #print(np.shape(SVD_vect1))
        #print(np.shape(self.EIM_basis_def))
        
        #appended_arr = []
        #for i in range(len(SVD_vect1.T)):
         #   multiply = self.EIM_basis_def.T*SVD_vect1.T[i]
         #   appended_arr.append(multiply)
        #print(np.array(appended_arr).T,'a') 
        #print(np.shape(np.array(appended_arr).T))            
        
        outer_prod = EIM_def_basis_axis * SVD_axis
       # print(outer_prod,'b')
        #print(np.shape(outer_prod))
        
        #print(np.shape(SVD_vect2.T), np.shape(outer_prod))

        
        psi_Vdef_psi = np.einsum('ij,jlm->ilm', SVD_vect2[self.drop:-self.drop].T, outer_prod[self.drop:-self.drop])
        
        return psi_Vdef_psi
    
    def vector_channel_sum(self, 
                           v1 : np.array,
                           v2 : np.array,
                           subsets_arr : np.array):
        '''
          Given two vectors, in particular the concatenated Galerkin coefficients and the concatenated
        Galerkin bases, split them into len(subsets) vectors, the number of channels in a coupled channel set
        and do a dot product of each individual splitted vector to obtain the emulated solution. 
        For the elastic channel, add the free solutions. This function is used once for each coupled channel set, 
        after solving the constraint equations.
        '''
        split_array = []
    
        emulated_channels = {}
        galerkin_coeff = {}
    
        split_v1 = np.split(v1,len(subsets_arr))
        split_v2 = np.split(v2,len(subsets_arr))
        #print(len(subsets))
        #print(np.shape(split_v2))
        #print(np.shape(split_v2))


    
        for h in range(len(subsets_arr)):
            l_scaling = int(subsets_arr[h][2])
        
#            if int(subsets_arr[h][1]) == 1:
#                phi0_res = self.phi0_dict.get(l_scaling)[self.drop:-self.drop]
#                emu_split = np.sum(split_v1[h]*split_v2[h].T,axis=1)
#                #emu_scaled, factor = rescaling_function_factor(emu_split,l_scaling)
#                emu = (emu_split + 1*phi0_res)
            if h == 0:
                phi0_res = self.phi0_dict.get(l_scaling)[self.drop:-self.drop]
                emu_split = np.sum(split_v1[h]*split_v2[h].T,axis=1)
             #emu_scaled, factor = rescaling_function_factor(emu_split,l_scaling)
                emu = (emu_split + 1*phi0_res)
             
            else:
                emu_split = np.sum(split_v1[h]*split_v2[h].T,axis=1)
                #emu_scaled, factor = rescaling_function_factor(np.sum(split_v1[0]*split_v2[0].T,axis=1),int(subsets[0][2]))
                emu = (np.array(emu_split))
        
            emulated_channels[subsets_arr[h]] = emu
            galerkin_coeff[subsets_arr[h]] = split_v1[h]
            #print(subsets[h])
        
        return emulated_channels, galerkin_coeff
    
    
    def compute_all_integrals(self
                              ):
        '''
        Solve the Galerkin equations after the off-line stage. Only dictionary inputs are used. The interaction is 
        obtained from Frescox and each coupled channel set is solved simultaneously by concatenating each 
        set of bases in the coupled channel set. The vector_channel_sum function is then called to split and obtain
        solutions for each individual channel in the coupled channel set.
        '''
        r_array = self.xgrid[self.drop:-self.drop]
        matrix_elements = CC_matrix(self.grouped_keys_dict, self.spin_p, self.spin_t_ex, self.spin_t_gs, self.spin_t_ex)
        #print(matrix_elements)
        integrated_subchannels_right = {}
        integrated_subchannels_left = {}
    
    

    
        for m in self.grouped_keys_dict:
            #free_solution_right = []
            basis_operators_tot = []
            psi = []
            
            subsets = self.grouped_keys_dict.get(m)
            matrix_subset = matrix_elements.get(m)
            
            #dimensions (nbasis), (nbasis,EIM_basis), (nchannels, nbasis, EIM_basis)
            # pass coefficients, add the first two and nchannels=0, concatenate the rest
            psi1_F_0, psi1_V_sph, psiv_V_def_channel = self.phi0_integrals(subsets, matrix_subset)
            
            #### save the integrals for each channel!!!

            for j in range(len(subsets)):
                def_operator_per_wave = []
                wave1 = self.SVD_grouped_dict.get(subsets[j])
                wave1_d2 = self.SVD_grouped_d2_dict.get(subsets[j])
                l = int(subsets[j][2])
                if int(subsets[j][1]) == 1:
                    energy = self.E_gs 
                else:
                    energy = self.E_ex
                    
                # array of dimension (nbasis, nbasis)
                psi_F0_psi = self.psi_F0_psi_integrals(wave1, wave1_d2, l, energy)   ## pass coefficients and add these two
                
                #array of dimension (nbasis, nbasis, EIM_basis)
                psi_Vsph_psi = self.psi_Vsph_psi_integrals(wave1)
                
                    
                
                for k in range(len(subsets)):
                    wave2 = self.SVD_grouped_dict.get(subsets[k])
                    if k != j:
                        ## array of dimension (nbasis,nbasis,EIM_basis)
                        psi_Vdef_psi = matrix_subset[j][k]*self.psi_Vdef_psi_integrals(wave1, wave2)
                        def_operator_per_wave.append(psi_Vdef_psi)
                    else:                        
                        ## array of dimension (nbasis,nbasis,EIM_basis) indexes are [wave2_index][wave1_index] <wave2|F * wave1>
                        psi_Vdef_psi = matrix_subset[j][k]*self.psi_Vdef_psi_integrals(wave1, wave2)
                        
                        ## def_operator_per_wave array has dimensions (waves_per_channel, nbasis, nbasis, EIM_basis)
                        def_operator_per_wave.append(psi_Vdef_psi)
                        ## pass coefficients and add psi_F0_psi, psi_Vsph_psi, and first element of def_operator_per_wave

                
                integrated_subchannels_left[(m,j)] = [psi_F0_psi, psi_Vsph_psi, def_operator_per_wave ]
            integrated_subchannels_right[(m)] = [psi1_F_0, psi1_V_sph, psiv_V_def_channel]
            
            
        return  integrated_subchannels_right, integrated_subchannels_left
    
    def Wavefunctions_Emulated(self, 
                               solving_array : np.array):
        
        total_channels = []
        
        for alpha in solving_array:
            sph_coeff, def_coeff = self.EIM_WS.coefficients(alpha, self.real_def)
            channels_per_run = []
            
            #sph_potential = np.einsum('j,ij -> i', sph_coeff, self.EIM_basis_sph)
            #print(sph_coeff, np.shape(self.EIM_basis_sph))
            #exact_potential = WS_spherical_interaction(self.xgrid, alpha)
            
            #print(sph_potential-exact_potential)
            

            
            for m in self.grouped_keys_dict:
                free_solution_right = []
                basis_operators_tot = []
                psi = []
                
                subsets = self.grouped_keys_dict.get(m)
                
                psi1_F_phi0, psi1_Vsph_phi0, psiv_Vdef_phi0 = self.integrated_subchannels_right.get(m)
                #print(np.shape(psi1_F_phi0), 'psi1_F_phi0')
                #print(np.shape(psi1_Vsph_phi0), 'psi1_Vsph_phi0')
               # print(np.shape(psiv_Vdef_phi0), 'psiv_Vdef_phi0')
                
                psi1_Vsph_phi0_coeff = np.einsum('ij,j->i', psi1_Vsph_phi0, sph_coeff)
               # print( psi1_Vsph_phi0_coeff, ' psi1_Vsph_phi0_coeff')
                
                for j in range(len(subsets)):
                    operators = []
                    
                    psi_F0_psi, psi_Vsph_psi, psi_Vdef_psi = self.integrated_subchannels_left.get((m,j))
                    psi_Vsph_psi_coeff = np.einsum('ijk,k->ij', psi_Vsph_psi, sph_coeff)
                    
                    wave = self.SVD_grouped_dict.get(subsets[j])[self.drop:-self.drop].T
                    
                    #wave_test = self.SVD_grouped_dict.get(subsets[j])
                    
                    #test1 = wave_test[self.drop:-self.drop].T @ (WS_spherical_interaction(self.xgrid[self.drop:-self.drop], alpha)[:, np.newaxis] * wave_test[self.drop:-self.drop])
                    #test2 = np.einsum('ilj, j -> il', self.psi_Vsph_psi_integrals(wave_test), sph_coeff)
                    #print(test1 - test2)
                    
                    if j == 0:
                        psiv_Vdef_phi0_coeff = np.einsum('ij,j->i', psiv_Vdef_phi0[j], def_coeff) + psi1_F_phi0 + psi1_Vsph_phi0_coeff
                    else:
                        psiv_Vdef_phi0_coeff = np.einsum('ij,j->i', psiv_Vdef_phi0[j], def_coeff)
                        

                    for k in range(len(subsets)):
                    
                        if k == j:
                            psi_Vdef_psi_coeff = np.einsum('ijk,k->ij', psi_Vdef_psi[k], def_coeff) + psi_F0_psi + psi_Vsph_psi_coeff
                            
                        else: 
                            psi_Vdef_psi_coeff = np.einsum('ijk,k->ij', psi_Vdef_psi[k], def_coeff)
                            
                        operators.append(psi_Vdef_psi_coeff)
                    free_solution_right.append(psiv_Vdef_phi0_coeff)
                    all_operators = np.concatenate(operators, axis=0)
                    basis_operators_tot.append(all_operators)
                    psi.append(wave)
                
                all_psi = np.concatenate(psi, axis=0)
               # print(np.shape(all_psi))
                #print(np.shape(free_solution_right))

                full_galerkin_matrix = np.concatenate(np.array(basis_operators_tot), axis=1)
                b = - np.concatenate(free_solution_right, axis=0)
                A = full_galerkin_matrix
                #print(A)
                a = np.linalg.solve(A, b)
                
                emulated_channels, galerkin_coeff = self.vector_channel_sum(a, all_psi, subsets)
                channels_per_run.append(emulated_channels)
            total_channels.append(channels_per_run)
            
        return total_channels, self.grouped_keys_dict


                
                

                    
                        
                        
            
            
    

                
    
    

        
        
    
    
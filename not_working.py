#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 16:50:18 2024

@author: manuelfranciscocatacorarios
"""

from typing import Callable
import numpy as np

from sampling import Sampling
from interaction_cc import Interaction_CC, InteractionSpace_CC
from cc_constants import CC_Constants


def max_vol(basis, indxGuess):
    r'''basis looks like a long matrix, the columns are the "pillars" V_i(x):
        [   V_1(x)
            V_2(x)
            .
            .
            .
        ]
        indxGuess is a first guess of where we should "measure", or ask the questions

    '''
    nbases = basis.shape[1]
    interpBasis = np.copy(basis)

    for ij in range(len(indxGuess)):
        interpBasis[[ij,indxGuess[ij]],:] = interpBasis[[indxGuess[ij],ij],:]
    indexing = np.array(range(len(interpBasis)))

    for ij in range(len(indxGuess)):
        indexing[[ ij,indxGuess[ij] ]] = indexing[[ indxGuess[ij],ij ]]
    
    for iIn in range(1, 100):
        B = np.dot(interpBasis, np.linalg.inv(interpBasis[:nbases]))
        b = np.max(B)
        if b > 1:
            
            p1, p2 = np.where(B == b)[0][0], np.where(B == b)[1][0]
            interpBasis[[p1,p2],:] = interpBasis[[p2,p1],:]
            indexing[[p1,p2]] = indexing[[p2,p1]]
        else:
            break
        #this thing returns the indexes of where we should measure
    return np.sort(indexing[:nbases])

class InteractionEIM_CC(Interaction_CC):
    def __init__(self, coordinate_space_potential_spherical: Callable[[float, np.array],float],    # radial potential function V(r,alpha)
                 coordinate_space_potential_deformed: Callable[[float, np.array],float],
                 n_alpha_sph: int,                                                          # number of parameters alpha in interaction
                 n_alpha_def: int,
                 training_array: np.array,# training array info
                 deformed_complex: bool = False,
                 is_complex: bool = False,
                 n_basis_sph: int = None,                                                  # number of basis in EIM
                 n_basis_def: int = None,
                 #explicit_training: bool = False,                                      # used defined array, if necessary
                 #n_train: int = 1000,                                                  # number of training points  # full radial array
                 r_array = np.array,
                 match_points: np.array = None):                                        # radial points at which matching is done for the EIM:
        
        super().__init__(coordinate_space_potential_spherical, 
                         coordinate_space_potential_deformed,
                         n_alpha_sph, n_alpha_def, is_complex=is_complex)
            
        self.r_mesh = r_array.copy()
        self.scale_training, self.samples, *self.central_val = training_info
        print(self.central_val)
            ## generate a basis used to approximate the potential
            ## did the user specify training points?
        initialize_train = Sampling(self.scale_training, self.samples, self.central_val)
        self.training_parameters = initialize_train.latinhypercube()
        
        if deformed_complex:
            snapshots_sph = np.array([self.radial_potential_sph(self.r_mesh, theta) for theta in training_parameters]).T
            
            training_parameters_def = np.array([*training_parameters.T[:3],*training_parameters.T[6:9]]).T
            snapshots_def = np.array([self.radial_potential_def(self.r_mesh, theta) for theta in training_parameters_def]).T
        
        else: 
            snapshots_sph = np.array([self.radial_potential_sph(self.r_mesh, theta) for theta in training_parameters]).T
            
            training_parameters_def = training_parameters.T[:3].T
            snapshots_def = np.array([self.radial_potential_def(self.r_mesh, theta) for theta in training_parameters_def]).T
        
        U_sph, S_sph, _ = np.linalg.svd(snapshots_sph, full_matrices=False)
        self.singular_values_sph = np.copy(S_sph)
        U_def, S_def, _ = np.linalg.svd(snapshots_def, full_matrices=False)
        self.singular_values_def = np.copy(S_def)
        
        if match_points is None:
            if n_basis_sph is None:
                n_basis_sph = n_alpha_sph
            if n_basis_def is None:
                n_basis_def = n_alpha_def
                
            self.snapshots_sph = np.copy(U_sph[:, :n_basis_sph])
            self.snapshots_def = np.copy(U_def[:, :n_basis_def])
            
            # random r points between 0 and 2π fm
            i_max_sph = self.snapshots_sph.shape[0] // 2
            i_max_def = self.snapshots_def.shape[0] // 2
            
            di_sph = i_max_sph // (n_basis_sph - 1)
            di_def = i_max_def // (n_basis_def - 1)
            
            i_init_sph = np.arange(0, i_max_sph + 1, di_sph)
            i_init_def = np.arange(0, i_max_def + 1, di_def)
            
            self.match_indices_sph = max_vol(self.snapshots_sph, i_init_sph)
            self.match_indices_def = max_vol(self.snapshots_def, i_init_def)
                # np.random.randint(0, self.snapshots.shape[0], size=self.snapshots.shape[1]))
            self.match_points_sph = r_array[self.match_indices_sph]
            self.match_points_def = r_array[self.match_indices_def]

            self.r_i_sph = np.copy(self.match_points_sph)
            self.r_i_def = np.copy(self.match_points_def)
        else:
            n_basis_sph = match_points.size
            n_basis_def = match_points.size
            
            self.snapshots_sph = np.copy(U_sph[:, :n_basis_sph])
            self.snapshots_def = np.copy(U_def[:, :n_basis_def])

            self.match_points_sph = np.copy(match_points)
            self.match_points_def = np.copy(match_points)

            self.match_indices_sph = np.array([np.argmin(np.abs(r_array - ri)) for ri in self.match_points_sph])
            self.match_indices_def = np.array([np.argmin(np.abs(r_array - ri)) for ri in self.match_points_def])

            
            self.r_i_sph = r_array[self.match_indices_sph]
            self.r_i_def = r_array[self.match_indices_def]


        self.Ainv_sph = np.linalg.inv(self.snapshots_sph[self.match_indices_sph])
        self.Ainv_def = np.linalg.inv(self.snapshots_def[self.match_indices_def])


        
        
    def coefficients(self,
        alpha: np.array,
        deformed_complex: bool = False,
    ):
        r'''Computes the EIM expansion coefficients.

        Parameters:
            alpha (ndarray): interaction parameters
        
        Returns:
            coefficients (ndarray): EIM expansion coefficients

        '''
        if deformed_complex:
            alpha_def = np.array([*alpha[:3],*alpha[6:9]])
            u_true_sph = self.radial_potential_sph(self.r_i_sph, alpha)
            u_true_def = self.radial_potential_def(self.r_i_def, alpha_def)
            coeff_sph = self.Ainv_sph @ u_true_sph
            coeff_def = self.Ainv_def @ u_true_def
        
        else:
            alpha_def = alpha[:3]
            u_true_sph = self.radial_potential_sph(self.r_i_sph, alpha)
            u_true_def = self.radial_potential_def(self.r_i_def, alpha_def)
            coeff_sph = self.Ainv_sph @ u_true_sph
            coeff_def = self.Ainv_def @ u_true_def
        
        return coeff_sph, coeff_def
    
    
    def radial_potential_emu(self,
        alpha: np.array,
        deformed_complex: bool = False,
    ):
        r'''Emulated interaction = $\hat{U}(s, \alpha, E)$

        Parameters:
            alpha (ndarray): interaction parameters
        
        Returns:
            u_hat (ndarray): emulated interaction

        '''
        a_sph, a_def = self.coefficients(alpha, deformed_complex)
        
        emu_sph = np.sum(a_sph * self.snapshots_sph, axis=1)
        emu_def = np.sum(a_def * self.snapshots_def, axis=1)
        
        return emu_sph, emu_def
    
    
    def basis_functions(self, r_array: np.array):
       r'''$u_j$ in $\tilde{U} \approx \hat{U} \equiv \sum_j \beta_j(\alpha) u_j$

       Parameters:
           s_mesh (ndarray): $s$ mesh points
       
       Returns:
           u_j (ndarray): "pillars" (MxN matrix; M = number of mesh points; N = number of pillars)

       '''
       return np.copy(self.snapshots_sph), np.copy(self.snapshots_def)
        
        
        
class InteractionEIMSpace_CC(InteractionSpace_CC):
    def __init__(self, coordinate_space_potential_spherical: Callable[[float, np.array],float],    # radial potential function V(r,alpha)
                 coordinate_space_potential_deformed: Callable[[float, np.array],float],
                 n_alpha_sph: int,                                                          # number of parameters alpha in interaction
                 n_alpha_def: int,
                 training_info: tuple,# training array info
                 deformed_complex: bool = False,
                 is_complex: bool = False,
                 n_basis_sph: int = None,                                                  # number of basis in EIM
                 n_basis_def: int = None,
                 #explicit_training: bool = False,                                      # used defined array, if necessary
                 #n_train: int = 1000,                                                  # number of training points  # full radial array
                 r_array = np.array,
                 match_points: np.array = None):                                        # radial points at which matching is done for the EIM:
        
        self.interactions_emu_tot = []
        self.interactions_emu_tot.append(
            [InteractionEIM_CC(coordinate_space_potential_spherical, 
                                                            coordinate_space_potential_deformed, 
                                                            n_alpha_sph,
                                                            n_alpha_def,
                                                            training_info,
                                                            is_complex=is_complex,
                                                            n_basis_sph= n_basis_sph,
                                                            n_basis_def= n_basis_def,
                                                            r_array= r_array,
                                                            match_points= match_points)])
        
        

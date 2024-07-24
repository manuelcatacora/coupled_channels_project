#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 15:56:00 2024

@author: manuelfranciscocatacorarios
"""

from typing import Callable
import numpy as np
from cc_constants import CC_Constants


class Interaction_CC:
    '''
    defines a nuclear interaction, which needs to be affine
    '''
    def __init__(self, coordinate_space_potential_spherical: Callable[[float, np.array], float], #potential radial function V(r,alpha)
                 coordinate_space_potential_deformed: Callable[[float, np.array, float], float],
                 n_alpha_sph : int,                                                    # number of parameters in interaction potential
                 n_alpha_def : int,
                 delta_def : float,
                 is_complex : bool = False ):
        
        
        self.v_r_sph = coordinate_space_potential_spherical
        self.v_r_def = coordinate_space_potential_deformed
        self.n_alpha_sph = n_alpha_sph
        self.n_alpha_def = n_alpha_def
        self.is_complex = is_complex
        self.def_parameter = delta_def
        
        
    def radial_potential_sph(self, 
                         r: float,            # radial value
                         alpha: np.array      # parameter array
                         ):     
        '''
        takes as input the radial values and array of parameters and outputs the radial potential at the given "r" with alpha
        parameters.
        '''
        
        vr = self.v_r_sph(r, alpha)
        
        return vr
    
    def radial_potential_def(self, 
                         r: float,            # radial value
                         alpha: np.array,      # parameter array
                         delta_def):     
        '''
        takes as input the radial values and array of parameters and outputs the radial potential at the given "r" with alpha
        parameters.
        '''
        #print(alpha,'a')
        vr_def = self.v_r_def(r, alpha, delta_def)
        
        return vr_def
    
    def basis_functions_sph(self, 
                        r_array: np.array     #radial array
                        ):
        '''
        takes in the radial array at which potential is to be calculated 
        and outputs an array for which all the affine parameters are assumed to be 1
        '''
        return np.array([
            self.radial_potential_sph(r_array, row) for row in np.eye(self.n_alpha_sph)
                         ]).T
   
    
    
    def basis_functions_def(self, 
                        r_array: np.array,     #radial array
                        delta_def):
        '''
        takes in the radial array at which potential is to be calculated 
        and outputs an array for which all the affine parameters are assumed to be 1
        '''
        return np.array([
            self.radial_potential_def(r_array, row, delta_def) for row in np.eye(self.n_alpha_def)
                         ]).T
            
    
    
    def coefficients(self,
                     alpha: np.array  # interaction parameters
                     ):
        '''
        the coefficients that will be passed to the basis potential vector elements
        '''
        
        return alpha
    

class InteractionSpace_CC:
    
    '''
    defines an array, whose elements are the interaction classes. This is structure this way because  the
    original ROSE single-channel code would define an interaction for each partial waves and pass through the couplings for
    the l*s spin-orbit interaction.
    '''
    
    def __init__(self, coordinate_space_potential_spherical: Callable[[float, np.array], float], #potential radial function V(r,alpha)
                 coordinate_space_potential_deformed: Callable[[float, np.array, float], float],
                 n_alpha_sph : int,                                                    # number of parameters in interaction potential
                 n_alpha_def : int,
                 delta_def : float,
                 is_complex: bool = False ):
        
        self.interactions_tot = []
        self.interactions_tot.append(
            [Interaction_CC(coordinate_space_potential_spherical,
                            coordinate_space_potential_deformed,
                                             n_alpha_sph,
                                             n_alpha_def,
                                             delta_def,
                                             is_complex=is_complex,
                                             )])
        
        
        
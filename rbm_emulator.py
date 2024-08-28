#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 02:10:13 2024

@author: manuelfranciscocatacorarios
"""

import numpy as np
from typing import Callable
from free_solution import phi_free
from cc_constants import CC_Constants

class Reduced_Basis_CC:
    def __init__(self, 
                 mass_t: float,
                 charge_t: float,
                 spin_t_gs: float,
                 spin_t_ex: float,
                 mass_p: float, 
                 charge_p: float,
                 spin_p: float,
                 E_lab: float,
                 J_tot_max: float,
                 J_tot_min: float,
                 coulomb_r: float,
                 reaction_name: str,
                 real_def: float,
                 E_states: np.array,
                 xgrid: np.array,
                 coordinate_space_potential_spherical: Callable[[float, np.array], float],    # radial potential function V(r,alpha)
                 coordinate_space_potential_deformed: Callable[[float, np.array], float],
                 n_alpha_sph: int,                                                          # number of parameters alpha in interaction
                 n_alpha_def: int,
                 training_array_EIM: np.array,  # training array info
                 training_array_RBM: np.array,
                 deformed_complex: bool = False,
                 is_complex: bool = False,
                 n_basis_sph_EIM: int = None,                                                # number of basis in EIM
                 n_basis_def_EIM: int = None,
                 n_basis_RBM: int = None,
                 match_points: np.array = None,
                 imag_def: float = 0):
        self.mass_t = mass_t
        self.charge_t = charge_t
        self.spin_t_gs = spin_t_gs
        self.spin_t_ex = spin_t_ex
        self.mass_p = mass_p
        self.charge_p = charge_p
        self.spin_p = spin_p
        self.E_lab = E_lab
        self.J_tot_max = J_tot_max
        self.J_tot_min = J_tot_min
        self.coulomb_r = coulomb_r
        self.reaction_name = reaction_name
        self.real_def = real_def
        self.E_states = E_states
        self.xgrid = xgrid
        self.coordinate_space_potential_spherical = coordinate_space_potential_spherical
        self.coordinate_space_potential_deformed = coordinate_space_potential_deformed
        self.n_alpha_sph = n_alpha_sph
        self.n_alpha_def = n_alpha_def
        self.training_array_EIM = training_array_EIM
        self.training_array_RBM = training_array_RBM
        self.deformed_complex = deformed_complex
        self.is_complex = is_complex
        self.n_basis_sph_EIM = n_basis_sph_EIM
        self.n_basis_def_EIM = n_basis_def_EIM
        self.n_basis_RBM = n_basis_RBM
        self.match_points = match_points
        self.imag_def = imag_def
        self.max_l = J_tot_max + spin_p + max(spin_t_ex,spin_t_gs)
        a = CC_Constants(mass_t, mass_p, E_lab, E_states)
        self.k_i = a.k()

    




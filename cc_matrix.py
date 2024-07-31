#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 16:48:10 2024

@author: manuelfranciscocatacorarios
"""

import numpy as np
from sympy.physics.wigner import clebsch_gordan
from sympy.physics.wigner import wigner_6j


# CURRENTLY THIS IMPLEMENTATION IS NOT EQUIPPED TO HANDLE S > 0.

def hat(a):
    '''
    Just a function to compute normalzation coefficients of the Wigner-Eckhart theorem
    '''
    return np.sqrt(2*a+1)



def coupling_matrix_elements(I_i, l_i, j_i, I_f, l_f, j_f, s, J, Q):
    '''
    Given the quantum numbers of one specific choice of initial and final channels calculate the coupling
    coefficient.
    '''
    K_f = 0
    K_i = 0
#     coeff = (-1)**(Q+s+K_f+(2*j_i)+J)*hat(l_i)*hat(l_f)*hat(j_i)*hat(j_f)*hat(I_i)*hat(I_f)*hat(Q) \
#           * wigner_6j(j_f,j_i,Q,I_i,I_f,J).n(12)*wigner_6j(j_f,j_i,Q,l_i,l_f,s).n(12) \
#             *wigner_3j(l_i,l_f,Q,0,0,0).n(12)*wigner_3j(I_i,I_f,Q,K_i,-K_f,0).n(12)
    if l_i==l_f:
        coeff = (1j**(l_i-l_f))*(-1)**(Q+J+l_i+I_f)*hat(I_i)*hat(l_i)*hat(Q) \
               * wigner_6j(l_i,I_i,J,I_f,l_f,Q).n(8) *clebsch_gordan(l_i,Q,l_f,0,0,0).n(8)\
               *clebsch_gordan(I_i,Q,I_f,K_i,0,K_f).n(8)
    else:
        coeff = (1j**(l_i-l_f))*(-1)**(Q+J+l_i+I_f)*hat(I_i)*hat(l_i)*hat(Q) \
               * wigner_6j(l_i,I_i,J,I_f,l_f,Q).n(8) *clebsch_gordan(l_i,Q,l_f,0,0,0).n(8)\
               *clebsch_gordan(I_i,Q,I_f,K_i,0,K_f).n(8)
    # in reality this equation should be updated to include a term i^(l_i-l_f) to account for the +1 in the secon equation
    
    return coeff


def CC_matrix(grouped_dict, spin, Q, spin_t_gs, spin_t_ex):
    '''
    Given a specific "grouped" coupled channel set, projectile spin and coupling potential order,
    calculate the coupling matrix elements between all channels in the coupled channel set. 
    '''
    group_coeff = {}
    
    for m in grouped_dict:
        sub_channel = grouped_dict.get(m)
        couplings = np.zeros((len(sub_channel), len(sub_channel)))
        
        for i in range(len(sub_channel)):
            I_initial = spin_t_gs if sub_channel[i][1] == 1 else spin_t_ex
            
            for j in range(len(sub_channel)):
                I_final = spin_t_gs if sub_channel[j][1] == 1 else spin_t_ex
                l_initial = sub_channel[i][2]
                j_initial = sub_channel[i][3]
                J_tot = sub_channel[i][4]
                l_final = sub_channel[j][2]
                j_final = sub_channel[j][3]
                
                coeff = coupling_matrix_elements(I_initial, l_initial, j_initial, I_final, l_final, j_final, spin, J_tot, Q)
                couplings[i][j] = coeff
            
        group_coeff[m] = couplings
            
    return group_coeff


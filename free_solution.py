#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 02:03:52 2024

@author: manuelfranciscocatacorarios
"""
from scipy.special import spherical_jn, spherical_yn
import numpy as np
from utility_functions_cc import find_closest_array_value, generate_first_derivative_function



def F(rho, ell):
    '''
    Bessel function of the first kind.
    '''
    return rho*spherical_jn(ell, rho)


def G(rho, ell):
    '''
    Bessel function of the second kind.
    '''
    return rho*spherical_yn(ell, rho)



def H_plus(rho, ell):
    '''
    Hankel function of the first kind.
    '''
    return G(rho, ell) + 1j*F(rho, ell)


def H_minus(rho, ell):
    '''
    Hankel function of the second kind.
    '''
    return G(rho, ell) - 1j*F(rho, ell)

def phi_free(rho, ell):
    '''
    Solution to the "free" (V = 0) radial Schrödinger equation.
    '''
    return (-H_plus(rho, ell) + H_minus(rho, ell))

def analytical_phi_0(rho):
    
    return 1j*np.sin(rho)

def CC_H_matrix(grouped_dictionary, r_match, r_array, momentum_array):
    
    '''
    For a given grouped dictionary (giving the quantum numbers of the reaction channel of interest), 
    a user-defined matching radius, a radial array, and the momentum array of the reaction, 
    generate the diagonal H matrices and their derivatives for each coupled channel. 
    '''
    
    r_closest, r_index = find_closest_array_value(r_match, r_array)
    H_plus_matrix = {}
    H_minus_matrix = {}
    d_dr = generate_first_derivative_function(r_array)
    
    for m in grouped_dictionary:
        subsets = grouped_dictionary.get(m)
        
        h_vect_plus_matched = []
        h_vect_minus_matched = []
        
        der_h_vect_plus_matched = []
        der_h_vect_minus_matched = []
        
        for j in range(len(subsets)):
            l = int(subsets[j][2])
            k_wavenum = momentum_array[int(subsets[j][1])-1]
            
            h_plus_l = H_plus(k_wavenum * r_array, l)
            h_minus_l = H_minus(k_wavenum * r_array, l)
            der_h_plus_l = d_dr(h_plus_l)
            der_h_minus_l = d_dr(h_minus_l)
            
            
            h_vect_plus_matched.append(h_plus_l[r_index])
            h_vect_minus_matched.append(h_minus_l[r_index])
            der_h_vect_plus_matched.append(der_h_plus_l[r_index])
            der_h_vect_minus_matched.append(der_h_minus_l[r_index])
            
            
        H_plus_matrix[(m)] = np.diag(h_vect_plus_matched)
        H_minus_matrix[(m)] = np.diag(h_vect_minus_matched)
        der_H_plus_matrix[(m)] = np.diag(der_h_vect_plus_matched)
        der_H_minus_matrix[(m)] = np.diag(der_h_vect_minus_matched)
        
    return H_plus_matrix, H_minus_matrix, der_H_plus_matrix, der_H_minus_matrix
        
    for m in grouped_dictionary:
    subsets = grouped_dictionary.get(m)
    h_vect_plus = []
    h_vect_minus = []
    for j in range(len(subsets)):
        l = int(subsets[j][2])
        h_plus_l = H_plus(k_1*r_array[r_match],l)
        h_minus_l = H_minus(k_2*r_array[r_match],l)
        h_vect_plus.append(h_plus_l)
        h_vect_minus.append(h_minus_l)
    H_plus_matrix[(m)] = np.diag(h_vect_plus)
    H_minus_matrix[(m)] = np.diag(h_vect_minus)

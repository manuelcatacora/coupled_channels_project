#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 00:06:25 2024

@author: manuelfranciscocatacorarios
"""
import numpy as np
from itertools import groupby



def group_channels(keys : dict):
    '''
    groups the keys by channel number to normalize and cast into galerkin formulation
    '''
    quantum_numbers = keys
    
    # Sort the list based on the first element of each tuple (channel set)
    quantum_numbers.sort(key=lambda x: x[0])
    # Group the tuples by the first element (channel set)
    g_quantum_numbers = {key: list(group) for key, group in groupby(quantum_numbers, key=lambda x: x[0])}
    
    return g_quantum_numbers

def Gamow_factor(l : int,
                 eta : float ):
    r'''This returns the... Gamow factor.
    See [Wikipedia](https://en.wikipedia.org/wiki/Gamow_factor).

    Parameters:
        l (int): angular momentum
        eta (float): Sommerfeld parameter (see
            [Wikipedia](https://en.wikipedia.org/wiki/Sommerfeld_parameter))
    
    Returns:
        C_l (float): Gamow factor

    '''
    if eta == 0.0:
        if l == 0:
            return 1
        else:
            return 1 / (2*l + 1) * Gamow_factor(l-1, 0)
    elif l == 0:
        return np.sqrt(2*np.pi*eta / (np.exp(2*np.pi*eta)-1))
    else:
        return np.sqrt(l**2 + eta**2) / (l*(2*l+1)) * Gamow_factor(l-1, eta)
    
def find_smallest_nonzero(wave: np.array,
                          l : int,
                          xgrid: np.array,
                          tolerance : float = 1e-5):
    '''
    given an array, it finds the smallest value matchin the r0 condition bellow starting from the beginning. 
    this will ensure we do not divide by 0.
    '''
    i = 0
    gamow = Gamow_factor(l,0)
    while np.abs(gamow*(xgrid[i])**(l+1)) <= tolerance: ##condition
        i += 1
        minimum = wave[i]
        #print(minimum)
        a = i
        #print(i)
    return minimum, a

def rescaling_function_free(wave: np.array,
                            keys: dict ,
                            l: int,
                            xgrid: np.array):
    '''
    normalizes the wave near zero in this manner: 
    psi' = C_l*psi*r0^(l+1)/psi(r0) for free solutions only!
    '''
    mini, i0 = find_smallest_nonzero(wave.get(keys[l]),l,xgrid)
    #wave_res = []
    w_keyed = Gamow_factor(l,0)*((xgrid[i0])**(l+1))*(wave.get(keys[l]))/(wave.get(keys[l]))[i0]
    return np.array(w_keyed)

def rescaling_function_factor(wave: np.array,
                              l: int,
                              xgrid: np.array):
    '''
    normalizes the wave near zero in this manner: 
    psi' = C_l*psi*r0^(l+1)/psi(r0)
    '''
    mini, i0 = find_smallest_nonzero(wave,l,xgrid)
    #wave_res = []
    
    w_keyed = (Gamow_factor(l,0)*((xgrid[i0])**(l+1))*(wave))/((wave)[i0])
    
    factor = (Gamow_factor(l,0)*((xgrid[i0])**(l+1)))/((wave)[i0])
    
    return np.array(w_keyed), factor
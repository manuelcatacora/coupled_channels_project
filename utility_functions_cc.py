#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 16:43:51 2024

@author: manuelfranciscocatacorarios
"""

import numpy as np
from findiff import FinDiff


def generate_second_derivative_matrix_9point(xgrid):
    '''
    For a given grid of points, compute the second derivative in 7 points
    '''
    d2_dx2 = FinDiff(0, xgrid[1]-xgrid[0], 2, acc=8)
    u = xgrid**2

    mat = d2_dx2.matrix(u.shape)  # this method returns a scipy sparse matrix
    
    D2 = mat.toarray()
    
    return D2

def generate_first_derivative_function(xgrid):
    '''
    for a given radial grid, return a first derivative function. 
    '''
    dx = xgrid[1]-xgrid[0]
    d_dx = FinDiff(0, dx, 1)
    return d_dx


def centrifugal(l,x):
    '''with a radia grid x, generate an array containing the centrifugal potential'''
        
    if x[0] == 0:
        x[0] = 1
        cent = (l*(l+1))/(x**2)
        cent[0] = 0.0
        x[0] = 0
        #print(cent)
    
    else:
        cent = (l*(l+1))/(x**2)
    
    return cent


def find_closest_array_value(value, array):
    '''
    for a user defined matching radius, find the radial array element 
    closest to it and then return the array element and its index.
    '''
    # Find the index of the closest value
    index = np.abs(array - value).argmin()
    # Return the closest value from the array
    return array[index], index



    
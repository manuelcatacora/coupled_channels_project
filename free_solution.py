#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 02:03:52 2024

@author: manuelfranciscocatacorarios
"""
from scipy.special import spherical_jn, spherical_yn
import numpy as np



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

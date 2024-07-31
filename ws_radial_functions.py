#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 16:18:02 2024

@author: manuelfranciscocatacorarios
"""
import numpy as np



def woods_saxon_base_interaction(R, depth, radius, difuseness):
    return -depth/(1+np.exp((R-((12**(1/3))*radius))/difuseness))

def der_woods_saxon_base_interaction(R, depth, radius, difuseness):
    exponential = np.exp((R-((12**(1/3))*radius))/difuseness)
    return -depth*exponential/(difuseness*(exponential+1)**2)


def WS_spherical_interaction(R,alpha):
    V, r, a, W_v, r_v, a_v, V_s, r_vs, a_vs, W_s, r_s, a_s = alpha
    full_interaction = woods_saxon_base_interaction(R, V, r, a) + 1j*woods_saxon_base_interaction(R, W_v, r_v, a_v) \
                        + der_woods_saxon_base_interaction(R, V_s, r_vs, a_vs) \
                        + 1j*der_woods_saxon_base_interaction(R, W_s, r_s, a_s)
    return full_interaction

def WS_deformed_real_interaction(R,alpha,lambda_def):
    V, r, a = alpha
    real_def_interaction = lambda_def*der_woods_saxon_base_interaction(R, V, r, a)/(np.sqrt(4*np.pi))
    return real_def_interaction

def WS_deformed_imaginary_interaction(R,alpha,lambda_def):
    V, r, a, W_s, r_s, a_s = alpha
    return None
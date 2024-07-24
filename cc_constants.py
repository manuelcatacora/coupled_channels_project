#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 10:35:49 2024

@author: manuelfranciscocatacorarios
"""
import numpy as np


class CC_Constants:
    amu =  931.49432                                                      #MeV
    hbarc = 197.32705
    finec = 137.03599
    def __init__(self, mass_t, mass_p, E_lab, E_states):
        self.mass_t = mass_t
        self.mass_p = mass_p
        self.E_lab = E_lab                                           #MeV Laboratory Energy
        self.E_states = tuple(E_states)                                     #MeV Excited States Energy Array (E_ground = 0)
        self.mu = (self.mass_t * self.mass_p * self.amu * self.amu)     \
                    /((self.mass_t + self.mass_p)*self.amu)
                    
        self.h2_mass = ((self.hbarc**2)/(2*self.mu))
        
    def reduced_mass(self):
        '''
        this class method calculates the reduced mass of the system  reduced_mass = (mass_t * mass_p)/(mass_t + mass_p)
        '''
        return self.mu
    
    def hbar2_reduced(self):
        '''
        this class method calculates the constant  hbarc^2/(2*reduced_mass) 
        '''
        
        return self.h2_mass

    def radial_m_coeff(self):
        
        '''
        this class method calculates the radial coefficient dependent on the mass_t of the system; mass_t^(1/3)
        '''
        mass_coeff = self.mass_t**(1/3)      
                    
        return mass_coeff
    
    def E_lab_to_COM(self):
        '''
        this class method calculates the center of mass energy from a given array of excited states energies, the array must
        be relative ot the ground state, e.g. E_ground = 0. 
        
        E_com = mass_t/(mass_p + mass_t)E_lab - State_Energy
        '''
        E_COM_array = []
        for energy in self.E_states:
            com_energy = (self.E_lab * ((self.mass_t)/(self.mass_t + self.mass_p))) - energy #E_com = mt/(mp+mt)
            E_COM_array.append(com_energy)
        return E_COM_array
    
    def k(self):
        '''
        this class method calculates an array whose elements are the wave-numbers of the appropriate energy in the COM frame;
        for the appropriate energy given the excitated states array
        k = sqrt(2*mu*E_COM/(hbarc**2))
        '''
        k_array = []
        for energy in self.E_states:
            k_com = np.sqrt((((self.E_lab) * ((self.mass_t)/(self.mass_t + self.mass_p)))- energy)\
                    /(self.h2_mass))
            k_array.append(k_com)
        return k_array
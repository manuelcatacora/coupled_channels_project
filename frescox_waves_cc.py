#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 17:36:11 2024

@author: manuelfranciscocatacorarios
"""
import numpy as np
import os
import re
import subprocess
from rescaling_functions_cc import (group_channels, Gamow_factor, find_smallest_nonzero, rescaling_function_factor) 



class Frescox_Inelastic_Wrapper:
    def __init__(self, 
                 mass_t: float,
                 charge_t : float,
                 spin_t_gs : float,
                 spin_t_ex : float,
                 mass_p : float, 
                 charge_p : float,
                 spin_p : float,
                 E_lab : float,
                 J_tot_max : float,
                 J_tot_min : float,
                 coulomb_r : float,
                 reaction_name : str,
                 real_def : float,
                 E_states : np.array,
                 xgrid : np.array,
                 imag_def : float = 0,
                 ):
        
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
        self.imag_def = imag_def
        self.xgrid = xgrid
        self.E_states = E_states
        

    def generate_template_inelastic(self):
        '''
        Given a parameter array, pass the parameters and rewrite the Frescox input file.
        '''
        file = 'frescox_full_template.in'

        # Read the template file
        with open(file) as f:
            content = f.readlines()
            
            # Iterate through the content and replace placeholders
        for idx, line in enumerate(content):
            if 'HEADER' in line:
                line_temp = line.replace("HEADER", str(self.reaction_name), 1)
                content[idx] = line_temp
            if 'J_TOT_MIN' in line and 'J_TOT_MAX' in line:
                line_temp = line.replace("J_TOT_MIN", str(self.J_tot_min), 1)
                line_temp = line_temp.replace("J_TOT_MAX", str(self.J_tot_max), 1)
                content[idx] = line_temp
            if 'E_LAB' in line:
                line_temp = line.replace("E_LAB", str(self.E_lab), 1)
                content[idx] = line_temp
            if 'MASS_P' in line and 'CHARGE_P' in line:
                line_temp = line.replace("MASS_P", str(self.mass_p), 1)
                line_temp = line_temp.replace("CHARGE_P", str(self.charge_p), 1)
                content[idx] = line_temp
            if 'MASS_T' in line and 'CHARGE_T' in line:
                line_temp = line.replace("MASS_T", str(self.mass_t), 1)
                line_temp = line_temp.replace("CHARGE_T", str(self.charge_t), 1)
                content[idx] = line_temp
            if 'S_PROJECTILE' in line and 'I_GROUND' in line:
                line_temp = line.replace("S_PROJECTILE", str(self.spin_p), 1)
                line_temp = line_temp.replace("I_GROUND", str(self.spin_t_gs), 1)
                line_temp = line_temp.replace("E_GROUND", str(self.E_states[0]), 1)
                content[idx] = line_temp
            if 'I_EXCITED' in line and 'E_EXCITED' in line:
                line_temp = line.replace("I_EXCITED", str(self.spin_t_ex), 1)
                line_temp = line_temp.replace("E_EXCITED", str(self.E_states[1]), 1)
                content[idx] = line_temp
            if 'MASS_T' in line and 'COULOMB_R' in line:
                line_temp = line.replace("MASS_T", str(self.mass_t), 1)
                line_temp = line_temp.replace("COULOMB_R", str(self.coulomb_r), 1)
                content[idx] = line_temp
            if 'REAL_DEFORM' in line:
                line_temp = line.replace("REAL_DEFORM", str(self.real_def), 1)
                content[idx] = line_temp
            if 'IMAGINARY_DEFORM' in line:
                line_temp = line.replace("IMAGINARY_DEFORM", str(self.imag_def), 1)
                content[idx] = line_temp

        # Write the modified content to a new file
        with open(f"{self.reaction_name}_input.in", "w") as f:
            f.writelines(content)

    
    def generate_input_file_inelastic(self,
                                      alpha:np.array):
        '''
        given a parameter array, pass the parameters and rewrite the Frescox input file
        '''
        #os.chdir("bandframework/software/Bfrescox/Tutorial_I/python_scripts")
        
        
        file = f'{self.reaction_name}_input.in'

        with open(file) as f:
            content = f.readlines()
        
        no_p = 0;
        for idx, line in enumerate(content):
            if 'XXXXX' in line:
                no_param = line.count('XXXXX')
                line_temp = line
                for i in range(no_param):
                    line_temp = line_temp.replace("XXXXX", str(alpha[no_p]), 1) 
                    no_p += 1
                    
                content[idx] = line_temp
    
        f = open("{self.reaction_name}_temp_input.in", "a")
        f.writelines(content)
        f.close()
        
        
        
    def extract_partial_waves(self, 
                              file_extract='fort.17'):
        '''
        this code extracts from the fort.17 file the partial waves and adds them as dictionary with the keys
        corresponding to the different quantum numbers of the coupled set of differential equations. The keys are
        broken down as follows {coupled channel set,channel number,l',j',J, l, j}.
        '''
        keys_array = []
    
        # Initialize a dictionary to store the values
        data_dict = {}

        # Initialize a counter for lines starting with "401"
        count_401 = 0

        # Specify the file path
        file_path = 'fort.17'

        # Regular expression pattern to match lines with the desired format
        pattern = re.compile(r'^\s*(\d+\s+\d+\s+\d+\.\d+\s+\d+\.\d+\s+\d+\s+\d+\.\d+)')
        
        pattern2 = re.compile(r'^\s*201\s+\d+\.\d+\s+\d+\.\d+\s+\d+\.\d+\s+-1\s+\d+\.\d+\s+\d+\.\d+\s+\d+\.\d+\s+\d+\.\d+')

        pattern3 = re.compile(r'^\s*-1\s+0\s+0\.0\s+0\.0\s+0\s+0\.0\s+0\.0{10}\s+0\.0{10}\s+0\.0{8}')

        # Open the file and read its contents
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        # Initialize a variable to hold the current key
        current_key = None

        # Iterate through the lines to find lines that match the pattern
        for i in range(len(lines)):
            line = lines[i].strip()
            match = pattern.match(line)
            if match:
                # Convert the first 6 values of the line to floats and use them as part of the key
                key_values = match.group(1).split()[:6]
                key_values = [float(value) for value in key_values]
                current_key = tuple([count_401] + key_values)
                # Initialize an empty list for the values associated with the key
                data_dict[current_key] = []
                i += 1  # Move to the next line to start reading values
                while i < len(lines)  and not lines[i].startswith("8001"):
                    # Split the current line into values
                    line2 = lines[i].strip()
                    match2 = pattern.match(line2)
                    line4 = lines[i].strip()
                    match4 = pattern3.match(line2)
                    if match2:
                        break
                    if match4:
                        i+=1
                    else:
                        values = lines[i].split()
                        # Convert values to floats and append them to the current key's list
                        data_dict[current_key].extend([float(value.replace('D', 'E')) for value in values])
                        i += 1
            elif lines[i].startswith("8001"):
                # Increment the counter for lines starting with "401"
                count_401 += 1
    
        keys_array = list(data_dict.keys())
  
    
        return data_dict, keys_array
    
    
    
    def wave_functions_format_imaginary(self, 
                                        file_extract='fort.17'):
        '''
        This function takes as input the fort.17 file and using the keys from the function extract_partial_waves
        then writes the wavefunction from the fresco output to \psi= real + imag
        '''
        waves_exact = {}
    
        data_dict, keys_array = self.extract_partial_waves()
    
        for key in keys_array:
            temp_real_array = []
            temp_imag_array = []
            temp_wave = data_dict.get(key)
            for i in range(len(np.array(temp_wave))):
                radial = temp_wave[i]
                if (i % 2 != 0):
                    temp_imag_array.append(radial)
                if (i % 2 == 0) :
                    temp_real_array.append(radial)
                
            waves_exact[key] = np.array(temp_real_array) + 1j* np.array(temp_imag_array)
        
        return waves_exact, keys_array
    
    
    
    
    def frescox_output_inelastic_wavefunctions(self, 
                                               input_file='{self.reaction_name}_temp_input.in',
                                               output_file='fort.17'):
        '''
        this function reads in the frescox input file formatted through the generated input file function
        runs it and extracts the partial wave functions with the respective keys corresponding to the quantum
        numbers.
        '''

        commands = '''
        cd ~/fewbody/CC_emulator
        mkdir frescox_outputs
        ~/fewbody/Frescoxex/frescox < {self.reaction_name}_temp_input.in > frescox_outputs/{self.reaction_name}_temp.out
        '''

        # Execute frescox
        bashresults = subprocess.run(commands, capture_output=True, shell=True)

        # Read outputs
        #os.chdir("bandframework/software/Bfrescox/Tutorial_I/python_scripts")
        
        waves, keys = self.wave_functions_format_imaginary(output_file)
    
        
        os.remove(input_file)
        os.remove(output_file)
    
        return waves, keys
    
    def frescox_run_inelastic_waves(self, 
                                    alpha_array : np.array):
        '''
        This function takes as an input a set of parameters and for each parameter set generates a frescox
        input using the template form, and then runs frescox and extracts the wave functions. the outputs 
        are an array whose elements are dictionaries of the different partial waves corresponding to
        each set of paramters in theta, and the keys to the dictionaries.
        '''
        initialize = self.generate_template_inelastic()
        theta_list = alpha_array.tolist()
        waves_array = []
        #    keys_array = []
        for para_obs in theta_list:
            #para_obs += [3.5,1,0.3]
            self.generate_input_file_inelastic(para_obs)
            waves_per_calc, keys_per_calc  = self.frescox_output_inelastic_wavefunctions()
            waves_array.append(waves_per_calc)
#           keys_array.append(keys_per_calc)

        return waves_array, keys_per_calc
    
    def frescox_run_inelastic_rescaled_waves(self, 
                                             alpha_array: np.array):
        '''
        This function takes as an input a set of parameters and for each parameter set generates a frescox
        input using the template form, and then runs frescox and extracts the wave functions. the outputs 
        are an array whose elements are dictionaries of the different partial waves corresponding to
        each set of paramters in theta, and the keys to the dictionaries. It then passes them through the appropriate
        rescaling functions above, this function normalizes the elastic and passes the same normalization
        to the inelastic channels in the same coupled channel set.
        '''
        waves, keys = self.frescox_run_inelastic_waves(alpha_array)
        grouped_channels = group_channels(keys)
    
        wave_scaled_dict = {}
        #phase_shift_dict = {}

    
        for m in (grouped_channels):
            subset_channels = grouped_channels.get(m)
            #print(subset_channels[0][2])
        
            for j in range(len(subset_channels)):
                l = int(subset_channels[j][2])
                wave_set = []
                #phase_shift_set = []
            
                
                if (subset_channels[j][1])==1.0:
                    for i in range(len(waves)):
                        wave_unscaled = waves[i].get(subset_channels[j])
                        wave_scaled, factor = rescaling_function_factor(wave_unscaled, l, self.xgrid)
                        #print(factor,i,l)
                        #phase_shift =  phase_shift_interp(u=wave_scaled,s=xgrid,ell=l,x0=x_phase_shift,dx=1e-6)
                        wave_set.append(wave_scaled) 
                        #phase_shift_set.append(phase_shift)
                
                
                else:
                    for i in range(len(waves)):
                        wave_unscaled = waves[i].get(subset_channels[j])
                        wave_scaled, factor = rescaling_function_factor(waves[i].get(subset_channels[0]), int(subset_channels[0][2]), self.xgrid)
                        wave_ineslastic= np.array(wave_unscaled)*factor
                        #phase_shift = phase_shift_interp_inelastic(u=wave_ineslastic, s=xgrid, ell=l, x0=x_renorm, x1=x_phase_shift, dx=1e-6)
                        wave_set.append(wave_ineslastic)
                        #phase_shift_set.append(phase_shift)

                #phase_shift_dict[subset_channels[j]] = np.array(phase_shift_set)
                wave_scaled_dict[subset_channels[j]] = np.array(wave_set)

            
        return wave_scaled_dict, grouped_channels
    





    
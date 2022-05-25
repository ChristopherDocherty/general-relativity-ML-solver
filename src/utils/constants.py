import tensorflow as tf
import numpy as np
import datetime
import math

### larger star ###
#scaling_factors = [100.0, 1e7, 2*np.pi, np.pi]
#M_sol = 1.5e3 
#scaling_factors = [100.0, 1e7, 2*np.pi, np.pi]

### Teeny star ###
scaling_factors = [100.0, 1e2, 2*np.pi, np.pi]
M_sol = 1.5e-2 
R_sol = 7 

#Constant density of star
rho = M_sol/(4/3 * math.pi * R_sol**3)

#If avoiding the discontinuity this is the half width of the 
#area around hte discontinuity that coordinates cannot be 
#sampled from
split_half_width = 5e-3


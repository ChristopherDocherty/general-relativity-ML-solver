import tensorflow as tf
import numpy as np
import datetime


from utils.constants import scaling_factors, M_sol, R_sol, rho

def get_mask_function(inside, r):

    if inside:
        return tf.cast(tf.math.less_equal(r,tf.convert_to_tensor(np.array(R_sol/scaling_factors[1]), dtype=tf.float32)), dtype=tf.float32) 
    else:
        return tf.cast(tf.math.greater_equal(r,tf.convert_to_tensor(np.array(R_sol/scaling_factors[1]), dtype=tf.float32)), dtype=tf.float32) 


def get_scaling_factor_correction_tensor(shape):

    correction_tensor = []


    for i in range(4):
        correction_tensor.append(tf.ones(shape=shape)  / scaling_factors[i])


    return tf.concat(correction_tensor, -1)



def timestamp_filename(f_str,path="/data/www.astro/2312403d/figs/"):
    return path + "_".join([f"{datetime.datetime.utcnow()}"[:-10].replace(" ", "_").replace(":", "-"), f_str])

import tensorflow as tf
import numpy as np
import datetime


from utils.constants import scaling_factors, M_sol, R_sol, rho, split_half_width

def get_mask_function(inside, r):
    '''
        This funciton is used to set elements of a tensor to zero based on whether they
        are inside or outside of the star's radius.

        This is used to enforce the discontinuity at the star's boundary found in the 
        metric tensor.
    '''


    if inside:
        return tf.cast(tf.math.less_equal(r,tf.convert_to_tensor(np.array(R_sol/scaling_factors[1]), dtype=tf.float32)), dtype=tf.float32) 
    else:
        return tf.cast(tf.math.greater_equal(r,tf.convert_to_tensor(np.array(R_sol/scaling_factors[1]), dtype=tf.float32)), dtype=tf.float32) 


def get_scaling_factor_correction_tensor(shape):
    '''
        Input cooridnates are normalised according to the scaling factors held in 
        utils.constants. 

        When tensorflow gradients are found, these scaling factors need to be reintroduced
        to give the correct derivatives which is when this funciton is used.
    '''

    correction_tensor = []


    for i in range(4):
        correction_tensor.append(tf.ones(shape=shape)  / scaling_factors[i])


    return tf.concat(correction_tensor, -1)



def timestamp_filename(f_str,path="/data/www.astro/2312403d/figs/"):
    '''
        Used to timestamp filenames so that multiple runs of data can be saved without overwriting
    '''
    return path + "_".join([f"{datetime.datetime.utcnow()}"[:-10].replace(" ", "_").replace(":", "-"), f_str])



def transform_metric(output, from_model=False):
    '''
        The metric needs to be transformed such that the dynamic range is large and the
        loss funciton can be effectively minimised


        This funciton is used on output from the network
    '''

    if from_model:
        return  1 - output/1000
    else:
        return 1000 * (1 - output)


def get_coords(size, fixed_dict, plotting):
    '''
    fixed_dict: determines which cooridnates will cover a range of value or take just one value

    plotting: determines whether linspace is used (in order for plotting) or random (for training)

     
    '''

    
    t,r,th,phi = None, None, None, None 

    if fixed_dict["t"]: 
        t = np.reshape(np.repeat(np.array(1),size),(-1,1))
    else:
        t = np.reshape(np.random.random(size),(-1,1))
    
    if fixed_dict["r"]:
        r = np.reshape(np.repeat(np.array(3e-2),size),(-1,1))
    else:
        if plotting:
            r = np.reshape(np.linspace(0,1,size) ,(-1,1)) #TODO: refactor this when changing values
        else:
            r = np.reshape(np.random.random(size),(-1,1))

        r = r*7e-2 + 4e-2
#        r = r + 1e-2

    if fixed_dict["th"]:
        th = np.reshape(np.repeat(np.array(0.635), size), (-1,1))
    else:
        th = np.reshape(np.random.random(size),(-1,1))

    if fixed_dict["phi"]:
        phi = np.reshape(np.repeat(np.array(0.873), size), (-1,1))
    else:
        phi = np.reshape(np.random.random(size),(-1,1))

    
    coords = np.concatenate((t,r,th,phi),1)
    coords = tf.convert_to_tensor(coords, dtype=tf.float32)
    return coords


def get_coords_avoiding_discontinuity(size, fixed_dict, plotting):
    '''
    Fixed_dict: determines which cooridnates will cover a range of value or take just one value

    plotting: determines whether linspace is used (in order for plotting) or random (for training)

    Choices about radii and other coordinate values are hard coded and must be changed manually.
    '''

    rescaled_R_sol = R_sol / scaling_factors[1]
    
    t,r,th,phi = None, None, None, None 

    if fixed_dict["t"]: 
        t = np.reshape(np.repeat(np.array(1),size),(-1,1))
    else:
        t = np.reshape(np.random.random(size),(-1,1))
    
    if fixed_dict["r"]:
        r = np.reshape(np.repeat(np.array(3e-2),size),(-1,1))
    else:
        if plotting:
            r1 = rescaled_R_sol - split_half_width - np.reshape(np.linspace(1e-2,0,size//2) ,(-1,1))
            r2 = rescaled_R_sol + split_half_width + np.reshape(np.linspace(0,2e-2,size//2) ,(-1,1))
            r = np.concatenate((r1,r2),0)

        else:
            r1 = rescaled_R_sol - split_half_width - np.reshape(np.random.random(size//2) * 1e-2 ,(-1,1))
            r2 = rescaled_R_sol + split_half_width + np.reshape(np.random.random(size//2) * 2e-2 ,(-1,1))
            r = np.concatenate((r1,r2),0)

    if fixed_dict["th"]:
        th = np.reshape(np.repeat(np.array(0.635), size), (-1,1))
    else:
        th = np.reshape(np.random.random(size),(-1,1))

    if fixed_dict["phi"]:
        phi = np.reshape(np.repeat(np.array(0.873), size), (-1,1))
    else:
        phi = np.reshape(np.random.random(size),(-1,1))

    
    coords = np.concatenate((t,r,th,phi),1)
    coords = tf.convert_to_tensor(coords, dtype=tf.float32)
    return coords


def generate_placeholder_coords(sample_cnt):
    '''
        Placeholder coordinates are reuqired as the tensorflow neural network requires
        cooridnates to be passed to its training method.

        As PINNs are an unsupervised learning algorithm, cooridnates are just generated every batch
        with the dimensions of these placeholder coordinates.
    '''

    t = np.reshape(np.zeros(sample_cnt),(-1,1))
    r = np.reshape(np.zeros(sample_cnt),(-1,1))
    th = np.reshape(np.zeros(sample_cnt),(-1,1))
    phi = np.reshape(np.zeros(sample_cnt),(-1,1))
    
    y = np.zeros(sample_cnt)
    
    placeholder_coords= np.concatenate((t,r,th,phi),1)

    return (tf.convert_to_tensor(placeholder_coords, dtype=tf.float32), y)

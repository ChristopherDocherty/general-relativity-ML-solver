import tensorflow as tf
import numpy as np
import datetime
import math

import data_visualisation


### larger star ###
#scaling_factors = [100.0, 1e7, 2*np.pi, np.pi]
#M_sol = 1.5e3 
#scaling_factors = [100.0, 1e7, 2*np.pi, np.pi]

### Teeny star ###
scaling_factors = [100.0, 1e2, 2*np.pi, np.pi]
M_sol = 1.5e-2 
R_sol = 7 


rho = M_sol/(4/3 * math.pi * R_sol**3)





def transform_metric(output, from_model=False):

    if from_model:
        return  1 - output/1000
    else:
        return 1000 * (1 - output)

#    if from_model:
#        return tf.math.log(output) 
#    else:
#        return tf.math.exp(output) 



def T(coords, batch_size):
    '''
    come back and simplify this spaghetti code
    '''

    inside_mask_tensor = get_mask_function(inside=True, r=coords[:,1])
    inside_mask_tensor = tf.expand_dims(inside_mask_tensor,-1)
    inside_mask_tensor = tf.expand_dims(inside_mask_tensor,-1)

    inside_T = tf.convert_to_tensor(np.array([
        [rho, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0]
    ]) 
    , dtype=tf.float32)

    inside_T = tf.expand_dims(inside_T,0)
    inside_T = tf.repeat(inside_T,batch_size, 0)

    return inside_mask_tensor * inside_T




def get_einstein_tensor(contr_ricci_tensor, g_inv, ricci_scalar, dims):

    ricci_scalar = tf.expand_dims(ricci_scalar, -1)
    ricci_scalar = tf.expand_dims(ricci_scalar, -1)
    ricci_scalar = tf.repeat(ricci_scalar,[dims],1)
    ricci_scalar = tf.repeat(ricci_scalar,[dims],2)


    return contr_ricci_tensor - 0.5 * ricci_scalar * g_inv 







##############################
## Metric finding functions ##
##############################

def build_g_from_g_rr(g_rr, coords):

    inside_mask_tensor = get_mask_function(inside=True, r=coords[:,1])
    outside_mask_tensor = get_mask_function(inside=False, r=coords[:,1]) 


    g_tt_inside = -(1.5 * tf.math.sqrt(1 - 2*M_sol/R_sol) - 0.5 * tf.math.sqrt(1 - 2*M_sol* (coords[:,1] * scaling_factors[1])**2 / R_sol**3))**2
    g_tt_outside = - (1 - (2*M_sol / (coords[:,1]*scaling_factors[1])))



    g_tt =  tf.expand_dims(inside_mask_tensor * g_tt_inside + outside_mask_tensor * g_tt_outside, 1)


    g_thth = tf.expand_dims((coords[:,1] * scaling_factors[1]) **2, 1)
    g_pp = g_thth * tf.math.sin(tf.expand_dims((coords[:,2] * scaling_factors[2]), 1))**2 


    g_0 = g_pp * 0 
    

    g_sp = tf.expand_dims(g_tt_inside,1)
    g_spp = tf.expand_dims(g_tt_outside,1)
 
    g_t = tf.concat([g_tt, g_0, g_0, g_0], 1)
    g_t = tf.expand_dims(g_t, 1)
    
 
    g_r = tf.concat([g_0, g_rr, g_0, g_0], 1)
    g_r = tf.expand_dims(g_r, 1)
    
 
    g_th = tf.concat([g_0, g_0, g_thth, g_0], 1)
    g_th = tf.expand_dims(g_th,1)
 
    g_p = tf.concat([g_0, g_0, g_0, g_pp],1)
    g_p = tf.expand_dims(g_p,1)
 
    return tf.concat([g_t, g_r, g_th, g_p],1)





def get_true_metric(coords):

    return get_sliced_true_metric(coords)
    
    inside_mask_tensor = get_mask_function(inside=True, r=coords[:,1])
    outside_mask_tensor = get_mask_function(inside=False, r=coords[:,1]) 
    g_rr_inside = 1 / (1 - 8/3 * math.pi * (coords[:,1] * scaling_factors[1])**2 * rho)
    g_rr_outside = 1 / (1 -2*M_sol / (coords[:,1] * scaling_factors[1])) 


    g_rr =  tf.expand_dims(inside_mask_tensor * g_rr_inside + outside_mask_tensor * g_rr_outside , 1)
    
    return build_g_from_g_rr(g_rr, coords) 








def get_spherical_minkowski_metric(coords):
    '''
        scaling_factors: holds the scaling factors implicit for each of the input coordinates

    '''
    g_thth = tf.expand_dims((coords[:,1] * scaling_factors[1]) **2, 1)

    g_pp = g_thth * tf.sin(tf.expand_dims(coords[:,2] * scaling_factors[2], 1))**2 
    g_0 = g_pp * 0
    
    g_tt = g_0 - tf.Variable([1.0])


    g_t = tf.concat([g_tt, g_0, g_0, g_0], 1)
    g_t = tf.expand_dims(g_t, 1)

    g_r = tf.concat([g_0, -g_tt, g_0, g_0], 1)
    g_r = tf.expand_dims(g_r, 1)
    

    g_th = tf.concat([g_0, g_0, g_thth, g_0], 1)
    g_th = tf.expand_dims(g_th,1)

    g_p = tf.concat([g_0, g_0, g_0, g_pp],1)
    g_p = tf.expand_dims(g_p,1)

    return  tf.concat([g_t, g_r, g_th, g_p],1)


def get_schwarzchild_metric(coords):
    
    r = coords[:,1] * scaling_factors[1]
    th = coords[:,2] * scaling_factors[2]

    g_tt = tf.expand_dims(- (1 - 2*M_sol/r) ,1)
    g_rr = tf.expand_dims(1/(1 - 2*M_sol/r) ,1)
    g_thth = tf.expand_dims(r**2 ,1)
    g_pp = tf.expand_dims(r**2 * tf.math.sin(th)**2 ,1)
    g_0 = g_pp * 0


    g_t = tf.concat([g_tt, g_0, g_0, g_0], 1)
    g_t = tf.expand_dims(g_t, 1)

    g_r = tf.concat([g_0, g_rr, g_0, g_0], 1)
    g_r = tf.expand_dims(g_r, 1)
    

    g_th = tf.concat([g_0, g_0, g_thth, g_0], 1)
    g_th = tf.expand_dims(g_th,1)

    g_p = tf.concat([g_0, g_0, g_0, g_pp],1)
    g_p = tf.expand_dims(g_p,1)

    return  tf.concat([g_t, g_r, g_th, g_p],1)



#########################
## testing functions   ##
#########################



def get_einstein_tensor_from_g(model, batch_size, use_model):

    with tf.GradientTape() as t2:
        with tf.GradientTape() as t1:
    
            
            fixed_dict = {
                    't': True,
                    'r': False,
                    'th': True, 
                    'phi': True 
                    }
            coords = get_coords(size = batch_size, fixed_dict=fixed_dict, plotting=True) 
            dims = coords.shape[1]
            
            
            t1.watch(coords)
            t2.watch(coords)

            g = None

            if use_model:
                g_linear = model(coords)
                g_linear = transform_metric(g_linear, True)
                g = build_g_from_g_rr(g_linear, coords)
            else:

                g = get_true_metric(coords)


        
        g_inv = tf.linalg.inv(g)

        grads = t1.batch_jacobian(g, coords) * get_scaling_factor_correction_tensor((batch_size,dims,dims,1))

    
        christoffel_tensor = 0.5 * (
            tf.einsum('ail,aklj->aikj', g_inv, grads) 
            + tf.einsum('ail,ajlk->aikj', g_inv, grads) 
            - tf.einsum('ail,ajkl->aikj', g_inv, grads)
        )
    
    christoffel_grads = t2.batch_jacobian(christoffel_tensor,coords) * get_scaling_factor_correction_tensor((batch_size,dims,dims,dims,1))

    
    
    reimann_tensor = (
        tf.einsum('amil,akmj->akijl',christoffel_tensor, christoffel_tensor)
        - tf.einsum('amij,akml->akijl',christoffel_tensor, christoffel_tensor)
        + tf.einsum('aijkl->aijlk', christoffel_grads)
        - christoffel_grads
        ) 
    


    ricci_tensor = tf.einsum('aijik', reimann_tensor)
    
    
    ricci_scalar = tf.einsum('aij,aij->a',g_inv, ricci_tensor)

    
    contr_ricci_tensor = tf.einsum('aiy,ajz,ayz->aij',g_inv, g_inv, ricci_tensor)

    
    return get_einstein_tensor(contr_ricci_tensor, g_inv, ricci_scalar, dims)




def test_metric_log_g_rr(model, test_sample_cnt): 

    return split_test_metric_log_g_rr(model, test_sample_cnt)
    fixed_dict = {
            't': True,
            'r': False,
            'th': True,
            'phi': True 
            }
    coords = get_coords(size = test_sample_cnt, fixed_dict=fixed_dict, plotting=True) 
        
    true_metric = get_true_metric(coords) 


    results = transform_metric(model(coords), True)
    results = build_g_from_g_rr(results, coords)


    data_visualisation.save_grr_plot(timestamp_filename("g_rr.jpg","/data/www.astro/2312403d/figs/"), coords, results, true_metric)

#    data_visualisation.save_4_4_tensor(timestamp_filename("full_g_sigmoid"),"g", coords, results)
#    data_visualisation.save_4_4_tensor("/data/www.astro/2312403d/figs/full_true_g","g", coords, true_metric)


#    einstein_tensor_predict = get_einstein_tensor_from_g(model, test_sample_cnt, True)
#    data_visualisation.save_4_4_tensor(timestamp_filename("full_G_sigmoid"),"G", coords, einstein_tensor_predict)
#
#    einstein_tensor_true = get_einstein_tensor_from_g(model, test_sample_cnt, False)
#    data_visualisation.save_4_4_tensor("/data/www.astro/2312403d/figs/full_true_G","G", coords, einstein_tensor_true)







###############
## general   ##
###############

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









def get_coords(size, fixed_dict, plotting, unsplit=False):
    '''
    Fixed_dict: determines which cooridnates will cover a range of value or take just one value

    plotting: determines whether linspace is used (in order for plotting) or random (for training)
    '''

    return get_split_coords(size,fixed_dict, plotting, unsplit)
    
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






###############
## split ver ##
###############

split_half_width = 2e-2

def get_sliced_true_metric(coords):


    inside_r = coords[:,1] + split_half_width 
    outside_r = coords[:,1] - split_half_width
    
    inside_mask_tensor = get_mask_function(inside=True, r=inside_r)
    outside_mask_tensor = get_mask_function(inside=False, r=outside_r) 
    g_rr_inside = 1 / (1 - 8/3 * math.pi * (inside_r * scaling_factors[1])**2 * rho)
    g_rr_outside = 1 / (1 -2*M_sol / (outside_r * scaling_factors[1])) 


    g_rr =  tf.expand_dims(inside_mask_tensor * g_rr_inside + outside_mask_tensor * g_rr_outside , 1)
    
    return build_g_from_g_rr(g_rr, coords) 

def split_test_metric_log_g_rr(model, test_sample_cnt): 

    fixed_dict = {
            't': True,
            'r': False,
            'th': True,
            'phi': True 
            }
    coords = get_coords(size = test_sample_cnt, fixed_dict=fixed_dict, plotting=True) 
        
    true_metric = get_true_metric(coords) 


    results = transform_metric(model(coords), True)
    results = build_g_from_g_rr(results, coords)

    coords = get_coords(size = test_sample_cnt, fixed_dict=fixed_dict, plotting=False, unsplit=True) 

    data_visualisation.save_grr_plot(timestamp_filename("g_rr.jpg","/data/www.astro/2312403d/figs/"), coords, results, true_metric)

def get_split_coords(size, fixed_dict, plotting, unsplit):
    '''
    Fixed_dict: determines which cooridnates will cover a range of value or take just one value

    plotting: determines whether linspace is used (in order for plotting) or random (for training)
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
        elif unsplit:
            r1 = rescaled_R_sol - np.reshape(np.linspace(1e-2,0,size//2) ,(-1,1))
            r2 = rescaled_R_sol + np.reshape(np.linspace(0,2e-2,size//2) ,(-1,1))
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


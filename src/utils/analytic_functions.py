import tensorflow as tf
import numpy as np
import math


from utils.constants import scaling_factors, M_sol, R_sol, rho
import utils.utils as utils


def T(coords, batch_size):
    '''
    Returns the energy-momentum tensor for a static star at each set of coords
    with the correct dimesnions to match a batch of training data
    '''

    inside_mask_tensor = utils.get_mask_function(inside=True, r=coords[:,1])
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


def build_g_from_g_rr(g_rr, coords):
    '''
        Takes a tensor containing values of the g_rr metric element and returns 
        the full set of metric elements at the given set of coords.

        THis funciton supplies the analytical solutions for all other elements.
    ''' 

    inside_mask_tensor = utils.get_mask_function(inside=True, r=coords[:,1])
    outside_mask_tensor = utils.get_mask_function(inside=False, r=coords[:,1]) 


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
    '''
        Returns a tensor containing the value of all elements of the true metric
        for each set of coordinates in coords.

        This is physically valid inside and outside of the star
    '''

    
    inside_mask_tensor = utils.get_mask_function(inside=True, r=coords[:,1])
    outside_mask_tensor = utils.get_mask_function(inside=False, r=coords[:,1]) 
    g_rr_inside = 1 / (1 - 8/3 * np.pi * (coords[:,1] * scaling_factors[1])**2 * rho)
    g_rr_outside = 1 / (1 -2*M_sol / (coords[:,1] * scaling_factors[1])) 


    g_rr =  tf.expand_dims(inside_mask_tensor * g_rr_inside + outside_mask_tensor * g_rr_outside , 1)
    
    return build_g_from_g_rr(g_rr, coords) 



def get_spherical_minkowski_metric(coords):
    '''
        Returns a tensor containing the value of all elements of the spherical Minkowski metric
        for each set of coordinates in coords

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
    '''
        Returns a tensor containing the value of all elements of the Schwarzchild metric
        for each set of coordinates in coords


        Only physically meaningful outside the star's radius
    '''
    
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





def get_einstein_tensor(contr_ricci_tensor, g_inv, ricci_scalar, dims):

    ricci_scalar = tf.expand_dims(ricci_scalar, -1)
    ricci_scalar = tf.expand_dims(ricci_scalar, -1)
    ricci_scalar = tf.repeat(ricci_scalar,[dims],1)
    ricci_scalar = tf.repeat(ricci_scalar,[dims],2)


    return contr_ricci_tensor - 0.5 * ricci_scalar * g_inv 







def get_true_metric_grads(coords, batch_size, dims):


    with tf.GradientTape() as t:
        t.watch(coords)

        g = get_true_metric(coords)

    
    return t.batch_jacobian(g, coords) * utils.get_scaling_factor_correction_tensor((batch_size,dims,dims,1))




def get_analytical_christoffel(coords):

    c = 8/3 * np.pi * rho
    b = 3/2 * math.sqrt(1- 2 * M_sol / R_sol)
    r_s = 2*M_sol

    r = coords[:,1] * scaling_factors[1]
    theta = coords[:,2] * scaling_factors[2]

    
    ins = utils.get_mask_function(inside=True, r=coords[:,1])
    out = utils.get_mask_function(inside=False, r=coords[:,1]) 



    c_rrr = tf.expand_dims( 
            ins * (c *r / (1 - c*r**2)) 
            - out * r_s / ( 2* r * (r - r_s))
                , 1)
              
    c_rtt = tf.expand_dims( 
            ins * 0.5 * (( b*c*r / tf.math.sqrt(1 - c*r**2) - 0.5*c*r) * (1 - c*r**2))
            + out * (r_s* (r - r_s) / (2*r**3)) 
            , 1)
              
    c_rthth = tf.expand_dims( 
            ins * (c*r**3 - r)
            - out * (r - r_s)
            , 1)
              
    c_rpp = tf.expand_dims( 
            ins * (c*r**3 - r) * tf.math.sin(theta)**2
            - out * (r-r_s) * tf.math.sin(theta)**2
            , 1)
              
    c_thpp = tf.expand_dims(  -tf.math.sin(theta) * tf.math.cos(theta), 1)
              

    c_ttr = tf.expand_dims( ins * 
                0.5 * 
                (c*r/tf.math.sqrt(1 - c*r**2)) *
                1/(b - 0.5 * tf.math.sqrt(1 - c*r**2))
            , 1)


    c_ththr = tf.expand_dims( 1/r, 1)
              
    c_ppr = tf.expand_dims( 1/r, 1)
              
    c_ppth = tf.expand_dims(  1/ tf.math.tan(theta), 1)

    c_000 = c_ppr * 0

    

    c_tt = tf.concat([c_000, c_ttr,c_000,c_000],1)
    c_tr = tf.concat([c_ttr,c_000,c_000,c_000],1)

    c_rt = tf.concat([c_rtt, c_000, c_000, c_000],1)
    c_rr = tf.concat([c_000, c_rrr, c_000, c_000],1)
    c_rth = tf.concat([c_000, c_000, c_rthth, c_000],1)
    c_rp = tf.concat([c_000, c_000, c_000, c_rpp],1)


    c_thr = tf.concat([c_000, c_000, c_ththr, c_000],1)
    c_thth = tf.concat([c_000, c_ththr, c_000, c_000],1)
    c_thp = tf.concat([c_000, c_000, c_000, c_thpp],1)


    c_pr = tf.concat([c_000,c_000, c_000, c_ppr],1)
    c_pth = tf.concat([c_000, c_000, c_000, c_ppth],1)
    c_pp = tf.concat([c_000, c_ppr, c_ppth, c_000],1)
    
    c_00 = tf.concat([c_000,c_000,c_000,c_000],1)




    c_t = [c_tt, c_tr,c_00,c_00]
    c_t = tf.concat([tf.expand_dims(x,1) for x in c_t], 1)

    c_r = [c_rt, c_rr, c_rth, c_rp]
    c_r = tf.concat([tf.expand_dims(x,1) for x in c_r], 1)


    c_th = [c_00, c_thr, c_thth, c_thp]
    c_th = tf.concat([tf.expand_dims(x,1) for x in c_th], 1)

    c_p = [c_00, c_pr, c_pth, c_pp]
    c_p = tf.concat([tf.expand_dims(x,1) for x in c_p], 1)

    c = [c_t, c_r, c_th, c_p]

    return tf.concat([tf.expand_dims(x,1) for x in c], 1)






def get_analytical_reimann(coords,batch_size, dims):

    with tf.GradientTape() as t2:
        with tf.GradientTape() as t1:
    

            t1.watch(coords)
            t2.watch(coords)
    
            g = get_true_metric(coords) 
    
        g_inv = tf.linalg.inv(g)

        grads = t1.batch_jacobian(g, coords) * utils.get_scaling_factor_correction_tensor((batch_size,dims,dims,1))

        christoffel_tensor = 0.5 * (
            tf.einsum('ail,aklj->aikj', g_inv, grads) 
            + tf.einsum('ail,ajlk->aikj', g_inv, grads) 
            - tf.einsum('ail,ajkl->aikj', g_inv, grads)
        )

    christoffel_grads = t2.batch_jacobian(christoffel_tensor,coords) * utils.get_scaling_factor_correction_tensor((batch_size,dims,dims,dims,1))

    
    reimann_tensor = (
        tf.einsum('amil,akmj->akijl',christoffel_tensor, christoffel_tensor)
        - tf.einsum('amij,akml->akijl',christoffel_tensor, christoffel_tensor)
        + tf.einsum('aijkl->aijlk', christoffel_grads)
        - christoffel_grads
        ) 

    return reimann_tensor

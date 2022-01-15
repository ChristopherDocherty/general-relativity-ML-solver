import tensorflow as tf
import numpy as np
import datetime
import math

import data_visualisation

one_tensor = tf.Variable([1.0])

scaling_factors = [100.0, 1e7, 2*np.pi, np.pi]


M_sol = 1.5e3
R_sol = 7e5

rho = M_sol/(4/3 * math.pi * R_sol**3)

def T(coords):

    return tf.constant([
        [rho, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0]
    ])



def logit(x):
    return tf.math.log(x / (1.0-x))




def get_einstein_tensor(contr_ricci_tensor, g_inv, ricci_scalar, dims):

    ricci_scalar = tf.expand_dims(ricci_scalar, -1)
    ricci_scalar = tf.expand_dims(ricci_scalar, -1)
    ricci_scalar = tf.repeat(ricci_scalar,[dims],1)
    ricci_scalar = tf.repeat(ricci_scalar,[dims],2)

    return contr_ricci_tensor - 0.5 * ricci_scalar * g_inv 


def get_contravariant_ricci_tensor(ricci_tensor, g_inv, dims):


    contr_ricci_tensor = []

    for i in range(dims):
        row = []

        for j in range(dims):

            contr_ricci_element = tf.zeros([1],tf.float32)

            for k in range(dims):
                for l in range(dims):

                    contr_ricci_element += g_inv[:,i,k] * g_inv[:,j,l] * ricci_tensor[:,k,l]

            contr_ricci_element = tf.expand_dims(contr_ricci_element, 1)
            row.append(contr_ricci_element)

        row = tf.concat(row, 1)
        row = tf.expand_dims(row, 1)
        contr_ricci_tensor.append(row)

    return tf.concat(contr_ricci_tensor, 1)


def get_ricci_scalar(ricci_tensor, g_inv, dims):


    ricci_scalar = tf.zeros([1], tf.float32)

    for i in range(dims):
        for j in range(dims):
            ricci_scalar += g_inv[:, i,j] * ricci_tensor[:, i,j]

    return ricci_scalar


def get_ricci_tensor(reimann_tensor, dims):

    matrix = []

    for i in range(dims):

        row = []

        for j in range(dims):

            ricci_element = tf.zeros([1],tf.float32)

            for k in range(dims):

                ricci_element += reimann_tensor[:,k,i,k,j]


            ricci_element = tf.expand_dims(ricci_element,1)
            row.append(ricci_element)

        row = tf.concat(row, 1)
        row = tf.expand_dims(row, 1)
        matrix.append(row)

    return tf.concat(matrix,1)


def get_reimann_tensor(christof_tensor, christof_grads, dims):


    big_tensor = []

    for i in range(dims):
        small_tensor = []

        for j in range(dims):

            matrix = []

            for k in range(dims):

                row = []

                for l in range(dims):

                    reimann_element = get_reimann_element((i,j,k,l), christof_tensor, christof_grads, dims)
                    reimann_element = tf.expand_dims(reimann_element, 1)
                    row.append(reimann_element) 

                row = tf.concat(row, 1)
                row = tf.expand_dims(row, 1)
                matrix.append(row)

            matrix = tf.concat(matrix, 1)
            matrix = tf.expand_dims(matrix, 1)
            small_tensor.append(matrix)

        small_tensor = tf.concat(small_tensor, 1)
        small_tensor = tf.expand_dims(small_tensor, 1)
        big_tensor.append(small_tensor)

    return tf.concat(big_tensor, 1)



def get_reimann_element(indices, christof_tensor, christof_grads, dims):
    '''
        indices according to R^\mu_{\alpha \beta \gamma}
    '''

    mu, alpha, beta, gamma = indices


    R = christof_grads[:, mu, alpha, gamma, beta]
    R -= christof_grads[:, mu, alpha, beta, gamma]    

    for sigma in range(dims):

        R += christof_tensor[:, sigma, alpha, gamma] * christof_tensor[:, mu, sigma, beta]
        R -= christof_tensor[:, sigma, alpha, beta] * christof_tensor[:, mu, sigma, gamma]

    return R



def get_christoffel_tensor(g_inv, grads, dims):

    tensor = []

    for i in range(dims):
        matrix = []

        for j in range(dims):

            row = []

            for k in range(dims):

                christoffel_symbol = get_christoffel_symbol((i,j,k), g_inv, grads, dims)

                christoffel_symbol = tf.expand_dims(christoffel_symbol, 1)
                row.append(christoffel_symbol) 

            row = tf.concat(row, 1)
            row = tf.expand_dims(row, 1)
            matrix.append(row)

        matrix = tf.concat(matrix, 1)
        matrix = tf.expand_dims(matrix, 1)
        tensor.append(matrix)


    return tf.concat(tensor,1)


def get_christoffel_symbol(indices, g_inv, grads, dims):
    '''
        indices specified as (mu, nu, sigma) <=> \Lambda^\mu_{\\nu\sigma}
    '''

    mu, nu, sigma = indices

    christoffel_symbol = tf.zeros([1], tf.float32)


    for i in range(dims):
        
            
        grad_sum =  grads[:,i, nu, sigma]  + grads[:, i, sigma, nu] - grads[:, sigma, nu, i]


        
        christoffel_symbol += 0.5 * g_inv[:,mu, i] * grad_sum

    return christoffel_symbol







def get_radius_test_coords(sample_cnt, full=False):

    t = np.reshape(np.repeat(np.array(1),sample_cnt),(-1,1))
    th = np.reshape(np.repeat(np.array(0.5), sample_cnt), (-1,1))
    phi = np.reshape(np.repeat(np.array(0.5), sample_cnt), (-1,1))

    r = None

    if full:
        r = np.reshape(np.linspace(0,1,sample_cnt)+3.01e-4,(-1,1))
    else:
        r = np.reshape(np.linspace(0,5e-4,sample_cnt)+3.01e-4,(-1,1))
    
    
    coords = np.concatenate((t,r,th,phi),1)

    return tf.convert_to_tensor(coords, dtype=tf.float32)
    

def get_theta_test_coords(sample_cnt):

    t = np.reshape(np.repeat(np.array(1),sample_cnt),(-1,1))
    r = np.reshape(np.repeat(np.array(1e8), sample_cnt), (-1,1))
    th = np.reshape(np.linspace(0,2*math.pi,sample_cnt),(-1,1))
    phi = np.reshape(np.repeat(np.array(0.8), sample_cnt), (-1,1))

    coords = np.concatenate((t,r,th,phi),1)

    return tf.convert_to_tensor(coords, dtype=tf.float32)


def get_phi_test_coords(sample_cnt):

    t = np.reshape(np.repeat(np.array(1),sample_cnt),(-1,1))
    r = np.reshape(np.repeat(np.array(1e8), sample_cnt), (-1,1))
    th = np.reshape(np.repeat(np.array(2.5), sample_cnt), (-1,1))
    phi = np.reshape(np.linspace(0,math.pi,sample_cnt), (-1,1))


    coords = np.concatenate((t,r,th,phi),1)

    return tf.convert_to_tensor(coords, dtype=tf.float32)


def timestamp_filename(f_str,path="/home/2312403d/masters/schwarzchild/saved_model_data/"):
    return path + "_".join([f"{datetime.datetime.utcnow()}"[:-10].replace(" ", "_").replace(":", "-"), f_str])


def get_true_metric(coords):

    g_tt = tf.expand_dims(- (1 - (2*M_sol / (coords[:,1]*scaling_factors[1]))), 1)
    g_rr = 1/(-g_tt)
    g_thth = tf.expand_dims((coords[:,1] * scaling_factors[1]) **2, 1)
    g_pp = g_thth * tf.math.sin(tf.expand_dims((coords[:,2] * scaling_factors[2]), 1))**2 

    
    return  tf.concat([g_tt, g_rr, g_thth, g_pp], 1)


def test_metric_logit(model, test_sample_cnt):
  

    radius_coords = get_radius_test_coords(test_sample_cnt)
        
    radius_true_metric = get_true_metric(radius_coords)

#    np.save(timestamp_filename("radius_true_metric"), radius_true_metric)

    radius_results = model(radius_coords)
    radius_results = build_g(radius_results, radius_coords)
    radius_results = logit(radius_results)

    np.save(timestamp_filename("radius_predicted"), radius_results)

    data_visualisation.save_grr_plot(timestamp_filename("radius_fig.jpg","/data/www.astro/2312403d/figs/"), radius_coords, radius_results, radius_true_metric)

    
def test_metric_log(model, test_sample_cnt):
  

    radius_coords = get_radius_test_coords(test_sample_cnt, True)
        
    radius_true_metric = get_true_metric(radius_coords)


    radius_results = model(radius_coords)
    radius_results = build_g(radius_results, radius_coords)
    radius_results = tf.math.exp(radius_results)


    data_visualisation.save_grr_plot(timestamp_filename("g_rr_in_full.jpg","/data/www.astro/2312403d/figs/"), radius_coords, radius_results, radius_true_metric)




    radius_coords = get_radius_test_coords(test_sample_cnt)
        
    radius_true_metric = get_true_metric(radius_coords)


    radius_results = model(radius_coords)
    radius_results = build_g(radius_results, radius_coords)
    radius_results = tf.math.exp(radius_results)


    data_visualisation.save_grr_plot(timestamp_filename("g_rr_ROI_focussed.jpg","/data/www.astro/2312403d/figs/"), radius_coords, radius_results, radius_true_metric)





def get_spherical_minkowski_metric(coords):
    '''
        scaling_factors: holds the scaling factors implicit for each of the input coordinates

    '''
    g_thth = tf.expand_dims((coords[:,1] * scaling_factors[1]) **2, 1)

    g_pp = g_thth * tf.sin(tf.expand_dims(coords[:,2] * scaling_factors[2], 1))**2 
    g_0 = g_pp * 0
    
    g_tt = g_0 - one_tensor


    g_t = tf.concat([g_tt, g_0, g_0, g_0], 1)
    g_t = tf.expand_dims(g_t, 1)

    g_r = tf.concat([g_0, -g_tt, g_0, g_0], 1)
    g_r = tf.expand_dims(g_r, 1)
    

    g_th = tf.concat([g_0, g_0, g_thth, g_0], 1)
    g_th = tf.expand_dims(g_th,1)

    g_p = tf.concat([g_0, g_0, g_0, g_pp],1)
    g_p = tf.expand_dims(g_p,1)

    return  tf.concat([g_t, g_r, g_th, g_p],1)



def build_g(g_rr, coords):


    g_tt = tf.expand_dims(- (1 - (2*M_sol / coords[:,1])), 1)
    g_thth = tf.expand_dims(coords[:,1]**2, 1)
    g_pp = g_thth * tf.math.sin(tf.expand_dims(coords[:,2], 1))**2 
    g_0 = g_pp * 0 
    
 
    g_t = tf.concat([g_tt, g_0, g_0, g_0], 1)
    g_t = tf.expand_dims(g_t, 1)
    
 
    g_r = tf.concat([g_0, g_rr, g_0, g_0], 1)
    g_r = tf.expand_dims(g_r, 1)
    
 
    g_th = tf.concat([g_0, g_0, g_thth, g_0], 1)
    g_th = tf.expand_dims(g_th,1)
 
    g_p = tf.concat([g_0, g_0, g_0, g_pp],1)
    g_p = tf.expand_dims(g_p,1)
 
    return tf.concat([g_t, g_r, g_th, g_p],1)

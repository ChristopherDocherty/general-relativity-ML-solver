import tensorflow as tf
import numpy as np


from visualisation import visualise

from utils.constants import scaling_factors, M_sol, R_sol, rho
from utils import analytic_functions, utils


def get_einstein_tensor_from_g(model, batch_size, use_model):
    '''
        Builds the Einstein tensor for a given metric.

        Can handle analytic metrics or metrics from trained models
    '''

    with tf.GradientTape() as t2:
        with tf.GradientTape() as t1:
    
            
            fixed_dict = {
                    't': True,
                    'r': False,
                    'th': True, 
                    'phi': True 
                    }
            coords = utils.get_coords(size = batch_size, fixed_dict=fixed_dict, plotting=True) 
            dims = coords.shape[1]
            
            
            t1.watch(coords)
            t2.watch(coords)

            g = None

            if use_model:
                g_linear = model(coords)
                g_linear = utils.transform_metric(g_linear, True)
                g = analytic_functions.build_g_from_g_rr(g_linear, coords)
            else:

                g = analytic_functions.get_true_metric(coords)


        
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
    


    ricci_tensor = tf.einsum('aijik', reimann_tensor)
    
    
    ricci_scalar = tf.einsum('aij,aij->a',g_inv, ricci_tensor)

    
    contr_ricci_tensor = tf.einsum('aiy,ajz,ayz->aij',g_inv, g_inv, ricci_tensor)

    
    return analytic_functions.get_einstein_tensor(contr_ricci_tensor, g_inv, ricci_scalar, dims)




def test_models_metric_prediction(model, test_sample_cnt): 
    '''
        Plots the predicted g_rr metric element and associated Einstein tensor from a 
        trained model as a function of radius for fixed t, \phi and \\theta.
    '''


    fixed_dict = {
            't': True,
            'r': False,
            'th': True,
            'phi': True 
            }

    coords = utils.get_coords(size = test_sample_cnt, fixed_dict=fixed_dict, plotting=True) 
        
    true_metric = analytic_functions.get_true_metric(coords) 


    results = utils.transform_metric(model(coords), True)
    results = analytic_functions.build_g_from_g_rr(results, coords)


    visualise.save_grr_plot(utils.timestamp_filename("g_rr.jpg","/data/www.astro/2312403d/figs/"), coords, results, true_metric)

    trueset_G = 8 * np.pi * analytic_functions.T(coords, test_sample_cnt)


    einstein_tensor_predict = get_einstein_tensor_from_g(model, test_sample_cnt, True)
    visualise.save_4_4_tensor_plot(utils.timestamp_filename("full_G_sigmoid,jpg"),"G", coords, einstein_tensor_predict, "", "PINN Einstein Tensor Prediction", trueset_G)

    einstein_tensor_true = get_einstein_tensor_from_g(model, test_sample_cnt, False)
    visualise.save_4_4_tensor_plot("/data/www.astro/2312403d/figs/full_true_G.jpg","G", coords, einstein_tensor_true, "", "Einstein Tensor Found From Analytical Metric", trueset_G)


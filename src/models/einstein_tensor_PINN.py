import os, math

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np

from utils import analytic_functions, utils



loss_tracker = tf.keras.metrics.Mean(name="myLoss")







class PINN_g_rr(tf.keras.Model):
    '''
    PINN class for calculating loss for Einsteins Equations
    '''

    def get_BC_loss(self, batch_size):
        '''
        Calculates the boundary condition loss by generating 
        coordinates at the boundary radius for various t, phi and theta
        '''


        BC_radii = [4e-2, 11e-2] 

        t = np.reshape(np.random.random(batch_size),(-1,1))
        th = np.reshape(np.random.random(batch_size),(-1,1))
        phi = np.reshape(np.random.random(batch_size),(-1,1))

        g_rr = []
        g_rr_true = []

        for radius in BC_radii:


            r = np.reshape(np.repeat(np.array(radius),batch_size),(-1,1))
            coords_BC = np.concatenate((t,r,th,phi),1)
            coords_BC = tf.convert_to_tensor(coords_BC, dtype=tf.float32)


            g_rr.append(self(coords_BC))

            g_rr_true.append(analytic_functions.get_true_metric(coords_BC)[:,1,1])


        
        g_rr = tf.concat(g_rr,0)
        g_rr_true= [ tf.expand_dims(utils.transform_metric(x, False), 1) for x in g_rr_true]
        g_rr_true = tf.concat(g_rr_true, 0)

        return tf.math.reduce_mean(tf.math.square(g_rr - g_rr_true)) 



    def get_PDE_loss(self, batch_size, dims):
        '''
            Find the PDE loss by converting the log output to the actual expected output,
            finding the associated Einstein tensor and comparing it to the Energy Momentum tensor defined for the system of interest
        '''

    
        with tf.GradientTape() as t2:
            with tf.GradientTape() as t1:
        

                fixed_dict = {
                        't': False,
                        'r': False,
                        'th': False,
                        'phi': False                        
                        }
        
                coords = utils.get_coords_avoiding_discontinuity(size=batch_size, fixed_dict=fixed_dict, plotting=False) 
                
                t1.watch(coords)
                t2.watch(coords)
        
        
        
                g_linear = self(coords)

                g_linear = utils.transform_metric(g_linear, True)
        
                g = analytic_functions.build_g_from_g_rr(g_linear, coords)
        
        
        
            g_inv = tf.linalg.inv(g)

            grads = t1.batch_jacobian(g, coords) * utils.get_scaling_factor_correction_tensor((batch_size,dims,dims,1))


        
            christoffel_tensor = 0.5 * (
                tf.einsum('ail,aklj->aikj', g_inv, grads) 
                + tf.einsum('ail,ajlk->aikj', g_inv, grads) 
                - tf.einsum('ail,ajkl->aikj', g_inv, grads)
            )

            true_christoffel_tensor = analytic_functions.get_analytical_christoffel(coords)

            return tf.math.reduce_mean(tf.math.square(christoffel_tensor - true_christoffel_tensor))

        christoffel_grads = t2.batch_jacobian(christoffel_tensor,coords) * utils.get_scaling_factor_correction_tensor((batch_size,dims,dims,dims,1))

        
        
        reimann_tensor = (
            tf.einsum('amil,akmj->akijl',christoffel_tensor, christoffel_tensor)
            - tf.einsum('amij,akml->akijl',christoffel_tensor, christoffel_tensor)
            + tf.einsum('aijkl->aijlk', christoffel_grads)
            - christoffel_grads
            ) 
        
#        true_reimann_tensor = utils.get_analytical_reimann(coords, batch_size, dims)
#
#
#        return tf.math.reduce_mean(tf.math.square(reimann_tensor - true_reimann_tensor))


        ricci_tensor = tf.einsum('aijik', reimann_tensor)
        
        ricci_scalar = tf.einsum('aij,aij->a',g_inv, ricci_tensor)

        contr_ricci_tensor = tf.einsum('aiy,ajz,ayz->aij',g_inv, g_inv, ricci_tensor)

        
        einstein_tensor = analytic_functions.get_einstein_tensor(contr_ricci_tensor, g_inv, ricci_scalar, dims)

        
        EE_loss = tf.math.reduce_mean(tf.math.square(einstein_tensor - 8*np.pi*utils.T(coords,batch_size)))

        
        return EE_loss



    def get_fixed_point_tracking_results(self):

        t = 0.5
        phi = 0.4
        theta = 0.3

        fixed_radii =[]

        fixed_points = [tf.convert_to_tensor(np.array([[t, r, phi, theta]]), dtype=tf.float32) for r in fixed_radii] 

       
        return {f"{r:.5f}" :  tf.math.reduce_mean(tf.math.exp(self(coords))) for r, coords in zip(fixed_radii, fixed_points)} 



    def train_step(self, data):
        '''
        Dummy coords are used as data can just be generated every iteration instead of using hte same set of data points
        each iteration

        '''

        placeholder_coords, _ = data

        dims = placeholder_coords.shape[1]
        batch_size = placeholder_coords.shape[0] 

        with tf.GradientTape() as t3:

            loss, PDE_loss, BC_loss  = 0,0,0

            PDE_loss = self.get_PDE_loss(batch_size, dims)

            PDE_loss = PDE_loss 
            BC_loss = self.get_BC_loss(batch_size)
            
            loss = self.PDE_factor * PDE_loss + BC_loss 


        gradients = t3.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        loss_tracker.update_state(loss)

        fixed_point_trackers = {}#self.get_fixed_point_tracking_results()

        return {"loss": loss_tracker.result(), "PDE_loss": PDE_loss, "BC_loss": BC_loss,  **fixed_point_trackers}

    @property
    def metrics(self):
        return [loss_tracker]





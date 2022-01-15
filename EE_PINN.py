import os, math

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np

import EE_utils



loss_tracker = tf.keras.metrics.Mean(name="myLoss")

class log_PINN(tf.keras.Model):
    '''
    PINN class for calculating loss for Einsteins Equations
    '''

    def get_BC_loss(self, batch_size):
        '''
        Calculates the boundary condition loss by generating 
        coordinates at the boundary radius for various t, phi and theta
        '''

        t = np.reshape(np.random.random(batch_size),(-1,1))
        r = np.reshape(np.repeat(np.array(1),batch_size),(-1,1))
        th = np.reshape(np.random.random(batch_size),(-1,1))
        phi = np.reshape(np.random.random(batch_size),(-1,1))


        coords_BC = np.concatenate((t,r,th,phi),1)
        coords_BC = tf.convert_to_tensor(coords_BC, dtype=tf.float32)


        g_rr= self(coords_BC)

        g_minkowski = EE_utils.get_spherical_minkowski_metric(coords_BC)


        return tf.reduce_mean(tf.math.abs(g_rr - tf.math.log(g_minkowski[:,1,1])))



    def get_PDE_loss(self, batch_size, dims):
        '''
            Find the PDE loss by converting the log output to the actual expected output,
            finding the associated Einstein tensor and comparing it to the Energy Momentum tensor defined for the system of interest

            r is shaped according to the scaling factor to give values in the region of interest (i.e. close to the star's Schwarzchild 
            radius) 
        '''

    
        with tf.GradientTape() as t2:
            with tf.GradientTape() as t1:
        
        
                
                t = np.reshape(np.random.random(batch_size),(-1,1))
                r = np.reshape(np.random.lognormal(0.0, 1, batch_size) * 1e-4 +3.01e-4,(-1,1))
                th = np.reshape(np.random.random(batch_size),(-1,1))
                phi = np.reshape(np.random.random(batch_size),(-1,1))
                
                coords = np.concatenate((t,r,th,phi),1)
                coords = tf.convert_to_tensor(coords, dtype=tf.float32)
                
                t1.watch(coords)
                t2.watch(coords)
        
        
        
                g_linear = tf.math.exp(self(coords))
        
                g = EE_utils.build_g(g_linear, coords)
        
        
        
            g_inv = tf.linalg.inv(g)
        
        
            grads = t1.batch_jacobian(g, coords)
            
        
            christofell_tensor = EE_utils.get_christoffel_tensor(g_inv, grads, dims)
        
        christofell_grads = t2.batch_jacobian(christofell_tensor,coords)
        
        
        reimann_tensor = EE_utils.get_reimann_tensor(christofell_tensor, christofell_grads, dims)
        
        ricci_tensor = EE_utils.get_ricci_tensor(reimann_tensor, dims)
        
        
        ricci_scalar = EE_utils.get_ricci_scalar(ricci_tensor, g_inv, dims)
        
        contr_ricci_tensor = EE_utils.get_contravariant_ricci_tensor(ricci_tensor, g_inv, dims)
        
        einstein_tensor = EE_utils.get_einstein_tensor(contr_ricci_tensor, g_inv, ricci_scalar, dims)
        
        
        return tf.reduce_mean(tf.math.abs(einstein_tensor - 8*np.pi*EE_utils.T(coords)))



    def get_fixed_point_tracking_results(self):

        t = 0.5
        phi = 0.4
        theta = 0.3

        fixed_radii =[1e-3,2e-3, 5e-3,1e-2,2e-2, 1e-1, 5e-1, 1]

        fixed_points = [tf.convert_to_tensor(np.array([[t, r, phi, theta]]), dtype=tf.float32) for r in fixed_radii] 

       
        return {f"{r:.5f}" :  tf.reduce_mean(tf.math.exp(self(coords))) for r, coords in zip(fixed_radii, fixed_points)} 



    def train_step(self, data):
        '''
        Dummy coords are used as data can just be generated every iteration instead of using hte same set of data points
        each iteration

        '''

        dummy_coords, g_true = data

        dims = dummy_coords.shape[1]
        batch_size = dummy_coords.shape[0] 

        with tf.GradientTape() as t3:

            PDE_loss = 1e-17 * self.get_PDE_loss(batch_size, dims)
            BC_loss = self.get_BC_loss(batch_size)

            loss = PDE_loss  + BC_loss


        trainable_vars = self.trainable_variables
        gradients = t3.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        loss_tracker.update_state(loss)

        fixed_point_trackers = self.get_fixed_point_tracking_results()

        return {"loss": loss_tracker.result(), "PDE_loss": PDE_loss, "BC_loss": BC_loss, **fixed_point_trackers}

    @property
    def metrics(self):
        return [loss_tracker]


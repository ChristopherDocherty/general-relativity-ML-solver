import os, math
import tensorflow as tf
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from utils import analytic_functions, utils


loss_tracker = tf.keras.metrics.Mean(name="myLoss")



class MetricTensorPINN(tf.keras.Model):
    '''
        Physics informed neural network (PINN) implemented with a custom loss function
        that represents how well Einstein's field equations are satsified by the 
        network prediction of the g_rr metric element. This onvolves satisfying 
        the PDE as well as boundary conditions that are set in this class

        Calling this model on a set of 4-dimensional coordinates will give the
        predicted values of the g_rr metric element for every set of coordinates
        in the input tensor
    '''

    def get_BC_loss(self, batch_size):
        '''
            When called, generates coordinates at the user defined boundaries with 
            the batch size given in the model.train() function.
           
            Returns the mean squared error of the model's prediction of the g_rr metric
            tensor element at the boundaries with the analytical value.

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
            When called, generates coordinates at the user defined boundaries with 
            the batch size given in the model.train() function.

            A g_rr prediction is then found using the model (i.e. self) from which a 
            metric tensor is built using the analytical values for the other elements
            at each set of coordinates.

            This is then passed through a pipeline that finds metric tensor
            associated with that metric.

            The mean squared error of this predicted metric tensor as compared
            to the analytical metric tensor is returned.
        '''

    

        fixed_dict = {
                't': False,
                'r': False,
                'th': False,
                'phi': False                        
                }

        coords = utils.get_coords_avoiding_discontinuity(size=batch_size, fixed_dict=fixed_dict, plotting=False) 

        g_linear = self(coords)

        g_linear = utils.transform_metric(g_linear, True)

        g = analytic_functions.build_g_from_g_rr(g_linear, coords)

        true_g = analytic_functions.get_true_metric(coords)


        return tf.math.reduce_mean(tf.math.square(g - true_g))



    def get_fixed_point_tracking_results(self):
        '''
            Optional function that tracks how the models prediction of the g_rr element of
            the metric changes every epoch.

            Can be used to understand if training is working correctly or not.
        '''

        t = 0.5
        phi = 0.4
        theta = 0.3

        #Populate the list below with radii to track
        fixed_radii =[]

        fixed_points = [tf.convert_to_tensor(np.array([[t, r, phi, theta]]), dtype=tf.float32) for r in fixed_radii] 

       
        return {f"{r:.5f}" :  tf.math.reduce_mean(tf.math.exp(self(coords))) for r, coords in zip(fixed_radii, fixed_points)} 



    def train_step(self, data):
        '''
            Overrides the train_step function of the tf.keras.Model class.

            This function determines the loss at each training step. 

            As the PINN algorithm is unsupervised learning, new coordinates
            are generated each training step and the placeholer coordinates
            in the variable "data" are only used to find the batch_size.

            The funciton returns  a dict of all variables to be tracked.

        '''

        placeholder_coords, _ = data

        dims = placeholder_coords.shape[1]
        batch_size = placeholder_coords.shape[0] 

        with tf.GradientTape() as t3:

            loss, PDE_loss, BC_loss  = 0,0,0

            PDE_loss = self.get_PDE_loss(batch_size, dims)

            BC_loss = self.get_BC_loss(batch_size)
           
            #the PDE factor is introduced so that callbacks 
            #can control whether the PDE_loss is shown to the 
            #optimiser. This allows the BC_loss to be satisfied
            #before the PDE_loss is "turned on"
            loss = self.PDE_factor * PDE_loss + BC_loss 


        gradients = t3.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        loss_tracker.update_state(loss)

        fixed_point_trackers = {}#self.get_fixed_point_tracking_results()

        return {"loss": loss_tracker.result(), "PDE_loss": PDE_loss, "BC_loss": BC_loss,  **fixed_point_trackers}

    @property
    def metrics(self):
        return [loss_tracker]





import tensorflow as tf
import numpy as np




class EaseInPDELoss(tf.keras.callbacks.Callback):
    '''
        This class is used to allwo the boundary conditions to be satisfied before the PDE loss is
        introduced. 

        Then, the learning rate is managed as the PDE_loss decreases.

    '''

    def __init__(self, model):
        super(EaseInPDELoss, self).__init__()
        model.PDE_factor = tf.Variable(0.0, trainable=False, name='PDE_factor', dtype=tf.float32) 

        

    def on_epoch_end(self, epoch, logs=None):


        if 'BC_loss' in logs and logs['BC_loss'] < 1e-4:
            tf.keras.backend.set_value(self.model.PDE_factor, tf.constant(1.0e11))


        if 'PDE_loss' in logs and logs['PDE_loss'] < 1e-12:
            tf.keras.backend.set_value(self.model.optimizer.lr, 1e-5)

        if 'PDE_loss' in logs and logs['PDE_loss'] < 6e-3:
            tf.keras.backend.set_value(self.model.optimizer.lr, 1e-4)

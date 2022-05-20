import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np

from utils import EE_utils 
from models import EE_PINN

gpu_id = 2 

os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = 'true'
gpus = tf.config.list_physical_devices('GPU')
print(f"This is the number of GPU's being used: {len(gpus)}")


np.random.seed(1234)
tf.random.set_seed(1234)



sample_cnt = 10000
batch_size = 1000


########################
##   Setup Functions  ##
########################
 

def generate_placeholder_coords(sample_cnt):
    '''
    These are the placehodler coordinates that I need to pass to the training method

    changing these will not affect the code whatsoever
    '''

    t = np.reshape(np.zeros(sample_cnt),(-1,1))
    r = np.reshape(np.zeros(sample_cnt),(-1,1))
    th = np.reshape(np.zeros(sample_cnt),(-1,1))
    phi = np.reshape(np.zeros(sample_cnt),(-1,1))
    
    y = np.zeros(sample_cnt)
    
    placeholder_coords= np.concatenate((t,r,th,phi),1)

    return (tf.convert_to_tensor(placeholder_coords, dtype=tf.float32), y)


class EaseInPDELoss(tf.keras.callbacks.Callback):

        def __init__(self, model):
            super(EaseInPDELoss, self).__init__()
            model.PDE_factor = tf.Variable(0.0, trainable=False, name='PDE_factor', dtype=tf.float32) 

            

        def on_epoch_end(self, epoch, logs=None):


            if 'BC_loss' in logs and logs['BC_loss'] < 1e-4:
                tf.keras.backend.set_value(self.model.PDE_factor, tf.constant(1.0e11))
#                tf.keras.backend.set_value(self.model.optimizer.lr, 1e-4)
#                tf.keras.backend.set_value(self.model.optimizer.lr, 1e-2)


#            if 'PDE_loss' in logs and logs['PDE_loss'] < 1e4:
#                tf.keras.backend.set_value(self.model.optimizer.lr, 1e-4)


            if 'PDE_loss' in logs and logs['PDE_loss'] < 1e-12:
                tf.keras.backend.set_value(self.model.optimizer.lr, 1e-5)

            if 'PDE_loss' in logs and logs['PDE_loss'] < 6e-3:
                tf.keras.backend.set_value(self.model.optimizer.lr, 1e-4)


########################
##       Model        ##
########################



initializer = tf.keras.initializers.GlorotUniform(
    seed = 1234
)

inputs = tf.keras.Input(shape=(4,), name="layer1")
outputs = tf.keras.layers.Dense(32, activation='sigmoid', kernel_initializer=initializer, name="layer2")(inputs)
outputs = tf.keras.layers.Dense(32,activation='sigmoid', kernel_initializer=initializer, name="layer3")(outputs)
outputs = tf.keras.layers.Dense(32,activation='sigmoid', kernel_initializer=initializer, name="layer4")(outputs)
outputs = tf.keras.layers.Dense(1, kernel_initializer=initializer, name="last")(outputs)


model = EE_PINN.PINN_g_rr(inputs, outputs)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1.0e-2))


placeholder_coords, y = generate_placeholder_coords(sample_cnt)
callback = EaseInPDELoss(model)

history = model.fit(placeholder_coords, y, batch_size=batch_size, epochs=10, callbacks=[callback])



EE_utils.test_metric_log_g_rr(model, 5000)


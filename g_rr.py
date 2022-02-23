import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np

import EE_utils, EE_PINN
import data_visualisation

gpu_id = 0 

os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = 'true'
gpus = tf.config.list_physical_devices('GPU')
print(f"This is the number of GPU's being used: {len(gpus)}")


np.random.seed(1234)
tf.random.set_seed(1234)



sample_cnt = 100000
batch_size = 10000


########################
##   Setup Functions  ##
########################
 

def generate_faux_coords(sample_cnt):
    '''
    These are the fake coordinates that I need to pass to the training method

    changing these will not affect the code whatsoever
    '''

    t = np.reshape(np.zeros(sample_cnt),(-1,1))
    r = np.reshape(np.zeros(sample_cnt),(-1,1))
    th = np.reshape(np.zeros(sample_cnt),(-1,1))
    phi = np.reshape(np.zeros(sample_cnt),(-1,1))
    
    y = np.zeros(sample_cnt)
    
    faux_coords= np.concatenate((t,r,th,phi),1)

    return (tf.convert_to_tensor(faux_coords, dtype=tf.float32), y)


class EaseInPDELoss(tf.keras.callbacks.Callback):

        def __init__(self, model):
            super(EaseInPDELoss, self).__init__()
            model.PDE_factor = tf.Variable(1.0, trainable=False, name='PDE_factor', dtype=tf.float32) 
            model.use_PINN = tf.Variable(1.0, trainable=False, name='use_PINN', dtype=tf.float32) 

            

        def on_epoch_end(self, epoch, logs=None):


            if 'loss' in logs and logs['loss'] < 1e-3:
#                tf.keras.backend.set_value(self.model.use_PINN, tf.Variable(1.0, trainable=False, dtype=tf.float32))
#                tf.keras.backend.set_value(self.model.PDE_factor, tf.constant(1.0e6))
                tf.keras.backend.set_value(self.model.optimizer.lr, 1e-3)

            if 'loss' in logs and logs['loss'] < 1e-4:
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

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1.0e-1))


faux_coords, y = generate_faux_coords(sample_cnt)
callback = EaseInPDELoss(model)

history = model.fit(faux_coords, y, batch_size=batch_size, epochs=500, callbacks=[callback])


########################
##      Plotting      ##
########################

data_visualisation.save_losses_plot(EE_utils.timestamp_filename("losses.jpg","/data/www.astro/2312403d/figs/"), history)
#data_visualisation.save_fixed_point_plots(EE_utils.timestamp_filename("fixed_point_results_1e-4.jpg","/data/www.astro/2312403d/figs/"), history)

EE_utils.test_metric_log_g_rr(model, 5000)


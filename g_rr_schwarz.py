import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np

import EE_utils, EE_PINN
import data_visualisation

gpu_id = 1 

os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = 'true'
gpus = tf.config.list_physical_devices('GPU')
print(f"This is the number of GPU's being used: {len(gpus)}")




sample_cnt = 1000
batch_size = 1000

def generate_faux_coords(sample_cnt):
    t = np.reshape(np.random.random(sample_cnt),(-1,1))
#    r = np.reshape(np.random.lognormal(-2, 0.1, sample_cnt)*1e-4+3e-4,(-1,1))
    r = np.reshape(np.linspace(0,5e-4,sample_cnt)+3.01e-4,(-1,1))
    th = np.reshape(np.random.random(sample_cnt),(-1,1))
    phi = np.reshape(np.random.random(sample_cnt),(-1,1))
    
#    print(np.mean(r), r.max(), r.min())
    y = np.linspace(0,1,sample_cnt)
    
    faux_coords= np.concatenate((t,r,th,phi),1)
    return (tf.convert_to_tensor(faux_coords, dtype=tf.float32), y)



initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.3)

inputs = tf.keras.Input(shape=(4,), name="layer1")
outputs = tf.keras.layers.Dense(64, activation='relu', kernel_initializer=initializer, name="layer2")(inputs)
outputs = tf.keras.layers.Dense(64,activation='relu', kernel_initializer=initializer, name="layer3")(outputs)
outputs = tf.keras.layers.Dense(64,activation='relu', kernel_initializer=initializer, name="layer4")(outputs)
outputs = tf.keras.layers.Dense(1, name="last")(inputs)
model = EE_PINN.log_PINN(inputs, outputs)

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    1e-1,
    decay_steps=1000.0,
    decay_rate=0.85,
    staircase=True)


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule))


faux_coords, y = generate_faux_coords(sample_cnt)

history = model.fit(faux_coords, y, batch_size=batch_size, epochs=500)

data_visualisation.save_losses_plot(EE_utils.timestamp_filename("losses_1e-1.jpg","/data/www.astro/2312403d/figs/"), history)
data_visualisation.save_fixed_point_plots(EE_utils.timestamp_filename("fixed_point_results.jpg","/data/www.astro/2312403d/figs/"), history)

EE_utils.test_metric_log(model, 10000)


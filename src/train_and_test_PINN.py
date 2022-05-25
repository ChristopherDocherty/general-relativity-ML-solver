import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np

from utils import testing, utils 
from visualisation import visualise
from models.einstein_tensor_PINN import EinsteinTensorPINN

from utils.callbacks import EaseInPDELoss

#Use this to choose which GPUs are used on a condor job
#If running locally then manually set the environment variables below
gpu_id = 2 

os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = 'true'
gpus = tf.config.list_physical_devices('GPU')
print(f"This is the number of GPU's being used: {len(gpus)}")


np.random.seed(1234)
tf.random.set_seed(1234)


sample_cnt = 10000
batch_size = 1000


initializer = tf.keras.initializers.GlorotUniform(
    seed = 1234
)


#Declartative declaration of network structure
inputs = tf.keras.Input(shape=(4,), name="layer1")
outputs = tf.keras.layers.Dense(32, activation='sigmoid', kernel_initializer=initializer, name="layer2")(inputs)
outputs = tf.keras.layers.Dense(32,activation='sigmoid', kernel_initializer=initializer, name="layer3")(outputs)
outputs = tf.keras.layers.Dense(32,activation='sigmoid', kernel_initializer=initializer, name="layer4")(outputs)
outputs = tf.keras.layers.Dense(1, kernel_initializer=initializer, name="last")(outputs)


model = EinsteinTensorPINN(inputs, outputs)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1.0e-2))


placeholder_coords, y = utils.generate_placeholder_coords(sample_cnt)
callback = EaseInPDELoss(model)

history = model.fit(placeholder_coords, y, batch_size=batch_size, epochs=10, callbacks=[callback])

visualise.save_losses_plot(utils.timestamp_filename("losses.jpg","/your_dir_here/"), history)

testing.test_models_metric_prediction(model, 5000)


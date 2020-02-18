import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from WeightedSum import WeightedSum

def create_base_generator():

    tensor_input = layers.Input(shape=(100,))
    
    first_layer = layers.Dense(128 * 4 * 4, kernel_initializer='he_normal')(tensor_input)
    first_layer_reshape = layers.Reshape((4, 4, 128))(first_layer)

    second_layer = layers.Conv2D(128, (4, 4), padding='same', kernel_initializer='he_normal')(first_layer_reshape)
    second_layer_batch_norm = layers.BatchNormalization()(second_layer)
    second_layer_leaky = layers.LeakyReLU(alpha=0.2)(second_layer_batch_norm)

    third_layer = layers.Conv2D(128, (3 , 3), padding='same', kernel_initializer='he_normal')(second_layer_leaky)
    third_layer_batch_norm = layers.BatchNormalization()(third_layer)
    third_layer_leaky = layers.LeakyReLU(alpha=0.2)(third_layer_batch_norm)

    out = layers.Conv2D(3, (1, 1), padding='same', kernel_initializer='he_normal')(third_layer_leaky)

    model = models.Model(tensor_input, out)

    return model

def progress_generator(model):

    block_end = model.layers[-2].output

    upsampling = layers.UpSampling2D()(block_end)

    first_layer = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(upsampling)
    first_layer_batch_norm = layers.BatchNormalization()(first_layer)
    first_layer_leaky = layers.LeakyReLU(alpha=0.2)(first_layer_batch_norm)

    second_layer = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(first_layer_leaky)
    second_layer_batch_norm = layers.BatchNormalization()(second_layer)
    second_layer_leaky = layers.LeakyReLU(alpha=0.2)(second_layer_batch_norm)
    
    out = layers.Conv2D(3, (1, 1), padding='same', kernel_initializer='he_normal')(second_layer_leaky)
    
    new_model = models.Model(model.input, out)

    out_old = model.layers[-1]

    out2 = out_old(upsampling)

    merged = WeightedSum()([out2, out])

    final_model = models.Model(model.input, merged)

    return [new_model, final_model]

def define_generator(n_blocks):
    model_list = []
    model = create_base_generator()
    model_list.append([model, model])
    
    for i in range(1, n_blocks):
        old_model = model_list[i - 1][0]
        models = progress_generator(old_model)
        model_list.append(models)
    
    return model_list

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from WeightedSum import WeightedSum


def create_base_discriminator():
    tensor_input = layers.Input(shape=(4, 4, 3))
    
    # fromRGB - spaja
    first_layer = layers.Conv2D(64, (1, 1), padding='same', kernel_initializer='he_normal')(tensor_input)
    first_layer_leaky = layers.LeakyReLU(alpha=0.2)(first_layer)

    second_layer = layers.Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(first_layer_leaky)
    second_layer_batch_norm = layers.BatchNormalization()(second_layer)
    second_layer_leaky = layers.LeakyReLU(alpha=0.2)(second_layer_batch_norm)

    third_layer = layers.Conv2D(128, (4, 4), padding='same', kernel_initializer='he_normal')(second_layer_leaky)
    third_layer_batch_norm = layers.BatchNormalization()(third_layer)
    third_layer_leaky = layers.LeakyReLU(alpha=0.2)(third_layer_batch_norm)

    flatten = layers.Flatten()(third_layer_leaky)
    out = layers.Dense(1)(flatten)

    model = models.Model(tensor_input, out)

    model.compile(loss='mse', optimizer=optimizers.Adam(lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))

    return model

def progress_discriminator(model, n_input_layers=3):
    
    input_shape = model.input.shape

    new_input_shape = (input_shape[1].value*2, input_shape[2].value*2, input_shape[3].value)

    input_layer = layers.Input(shape=new_input_shape)

    first_layer = layers.Conv2D(64, (1, 1), padding='same', kernel_initializer='he_normal')(input_layer)
    first_layer_leaky = layers.LeakyReLU(alpha=0.2)(first_layer)

    second_layer = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(first_layer_leaky)
    second_layer_batch_norm = layers.BatchNormalization()(second_layer)
    second_layer_leaky = layers.LeakyReLU(alpha=0.2)(second_layer_batch_norm)
    
    third_layer = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(second_layer_leaky)
    third_layer_batch_norm = layers.BatchNormalization()(third_layer)
    third_layer_leaky = layers.LeakyReLU(alpha=0.2)(third_layer_batch_norm)

    last_layer = layers.AveragePooling2D()(third_layer_leaky)
    final_layer = last_layer
    
    for i in range(n_input_layers, len(model.layers)):
        last_layer = model.layers[i](last_layer)

    new_model = models.Model(input_layer, last_layer)

    new_model.compile(loss='mse', optimizer=optimizers.Adam(lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))

    downsample = layers.AveragePooling2D()(input_layer)
    
    block_old = model.layers[1](downsample)
    block_old = model.layers[2](block_old)

    d = WeightedSum()([block_old, final_layer])

    for i in range(n_input_layers, len(model.layers)):
        d = model.layers[i](d)
    
    final_model = models.Model(input_layer, d)
    final_model.compile(loss="mse", optimizer=optimizers.Adam(lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))

    return [new_model, final_model]

def define_discriminator(n_blocks):
    model_list = list()
    model = create_base_discriminator()

    model_list.append([model, model])

    for i in range(1, n_blocks):
        old_model = model_list[i - 1][0]
        models = progress_discriminator(old_model)
        model_list.append(models)

    return model_list

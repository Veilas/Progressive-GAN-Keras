#! /usr/bin/env python3

###################################################################
#
#   Debug levels:
#     0 = all messages are logged (default behavior)
#     1 = INFO messages are not printed
#     2 = INFO and WARNING messages are not printed
#     3 = INFO, WARNING, and ERROR messages are not printed
#
###################################################################

import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K
from WeightedSum import WeightedSum
from discriminator import define_discriminator
from generator import define_generator
from os import listdir
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm


def save_list_weigths(model_list, name):
    path = "../model_weights/" + name
    for i in range(len(model_list)):
        model_list[i][0].save_weights(path + "%dx%d.h5" %(i, 0))
        model_list[i][1].save_weights(path + "%dx%d.h5" %(i, 1))

def load_list_weigths(model_list, name):
    path = "../model_weights/" + name
    for i in range(len(model_list)):
        model_list[i][0].load_weights(path + "%dx%d.h5" %(i, 0))
        model_list[i][1].load_weights(path + "%dx%d.h5" %(i, 1))

def save_biggest_model(model, name, level):
    path = "../model_weights/m_" + name
    model.save_weights(path + "x%d.h5" % level)

def load_biggest_model(model, name, level):
    path = "../model_weights/m_" + name
    model.load_weights(path + "x%d.h5" % level)

def save_models_w(discriminator, generator, depth):
    save_biggest_model(discriminator, "discriminator", depth)
    save_biggest_model(generator, "generator", depth)

def load_models_w(discriminator, generator, depth):
    load_biggest_model(discriminator, "discriminator", depth)
    load_biggest_model(generator, "generator", depth)

def save_model(model, name, depth):
    path = "../models/" + name
    model.save(path + "x%d.h5" % depth)


def load_dataset(size):
    folder_path = "../datasets/data%dx%d" % (size, size)
    image_files = listdir(folder_path)
    images = [0] * len(image_files)
    print("Loading dataset %dx%d: " % (size, size))
    for i in tqdm(range(len(image_files))):
        images[i] = cv2.imread(folder_path + "/" + image_files[i])
    dataset = np.asarray(images)
    print("Finished loading dataset")
    return dataset

def normalize(images):
    return (images.astype(np.float32) - 127.5) / 127.5

def denormalize(array):
    return (np.clip((array * 127.5) + 127.5), 0, 255).astype(np.uint8)
    

def generate_real_samples(dataset, batch_size):
    start = np.random.randint(0, dataset.shape[0] - batch_size)
    end = start + batch_size
    return dataset[start:end, :, :, :]

def generate_fake_samples(g_model, batch_size):
    noise = np.random.uniform(size=(batch_size, 100))
    fake_samples = g_model.predict(noise)
    return fake_samples

def merge(discriminators, generators):
    model_list = list()
    for i in range(len(discriminators)):
        d_models, g_models = discriminators[i], generators[i]
        d_models[0].trainable = False
        first_model = models.Sequential()
        first_model.add(g_models[0])
        first_model.add(d_models[0])
        first_model.compile(loss='mse', optimizer=optimizers.Adam(lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))

        d_models[1].trainable = False
        second_model = models.Sequential()
        second_model.add(g_models[1])
        second_model.add(d_models[0])
        second_model.compile(loss='mse', optimizer=optimizers.Adam(lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))

        model_list.append([first_model, second_model])
    return model_list

def update_fadein(models, step, n_steps):
    alpha = step / float(n_steps - 1)
    for model in models:
        for layer in model.layers:
            if isinstance(layer, WeightedSum):
                K.set_value(layer.alpha, alpha)

def train_epochs(d_model, g_model, gan_model, dataset, n_epochs, n_batch, fadein=False):
    batches_count = int(dataset.shape[0] / n_batch)
    n_steps = batches_count * n_epochs
    half_batch = int(batches_count / 2)

    for i in range(n_steps):
        if fadein:
            update_fadein([g_model, d_model, gan_model], i, n_steps)

        x_real, y_real = generate_real_samples(dataset, half_batch), np.ones((half_batch, 1))
        x_fake, y_fake = generate_fake_samples(g_model, half_batch), np.zeros((half_batch, 1))
        d_loss_real = d_model.train_on_batch(x_real, y_real)
        d_loss_fake = d_model.train_on_batch(x_fake, y_fake)
        z_input, z_score = np.random.uniform(size=(n_batch, 100)), np.ones((n_batch, 1))
        g_loss = gan_model.train_on_batch(z_input, z_score)

        print('>%d, loss_true_image=%.3f, loss_fake_image=%.3f model_loss=%.3f' % (i+1, d_loss_real, d_loss_fake, g_loss))

def train(g_models, d_models, gan_models, e_norm, e_fadein, n_batch ):
    for i in range(3, len(g_models)):
        dataset = normalize(load_dataset(2**(i + 2)))
        #train fade-in 
        train_epochs(model_list_disc[i][1],
                    model_list_gen[i][1],
                    model_list_gan[i][1],
                    dataset,
                    e_fadein,
                    n_batch,
                    True)

        #train streight through
        train_epochs(model_list_disc[i][0],
                model_list_gen[i][0],
                model_list_gan[i][0],
                dataset,
                e_norm,
                n_batch)
        # save_models_w(model_list_disc[i][1], model_list_gen[i][1], i + 1)

        save_model(model_list_disc[i][0], "discriminator", i + 1)
        save_model(model_list_gen[i][0], "generator", i + 1)

        save_list_weigths(model_list_disc, "discriminator")
        save_list_weigths(model_list_gen, "generator")

        del dataset
        
   

depth = 7

model_list_disc = define_discriminator(depth)
model_list_gen = define_generator(depth)

model_list_gan = merge(model_list_disc, model_list_gen)






train(model_list_gen, model_list_disc, model_list_gan, 100, 100, 512)



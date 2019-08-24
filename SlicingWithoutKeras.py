"""
This code works! It takes a keras graph and gets the output of the final layer. 
This output is a vector of the size necessary to fill all weights values.
The output is sliced and reshhaped in the spn.py file and loaded into the weights in the layers.py file
Everything else about the DCSPN is the same with the exception of the feed_dict now including the one-hot repersentation and the train images
"""



from __future__ import print_function

from tensorflow.keras import layers #actually imports the keras
from dcspn.spn import SumProductNetwork
from dcspn.layers import SumLayer, ProductLayer, GaussianLeafLayer
from dcspn.utilities import Database, plot_cost_graph


import tensorflow as tf
import numpy as np
import argparse
import os
import gc
import datetime
import logging

import matplotlib.pyplot as plt



########################################TLNN#####################################################
batch_size = 128
epochs = 2




#weightSize1 = 31370
#Size of the weights for the keras output
weightSize1 = 34507
weightSize2 = 10





# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(train_imgs, train_labels), (test_imgs, test_labels) = tf.keras.datasets.mnist.load_data()


#Reshaping and normalize the imported images
train_imgs = train_imgs.reshape(train_imgs.shape[0], img_rows, img_cols, 1)
test_imgs = test_imgs.reshape(test_imgs.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

train_imgs= train_imgs.astype('float32')
test_imgs = test_imgs.astype('float32')
train_imgs /= 255
test_imgs /= 255
#print('x_train shape:', x_train.shape)
print(train_imgs.shape[0], 'train samples')
print(test_imgs.shape[0], 'test samples')

# for feeding
feed_labels = tf.keras.utils.to_categorical(train_labels, 10)


# convert class vectors to binary class matrices
train_labels = tf.keras.utils.to_categorical(train_labels, weightSize1)
test_labels = tf.keras.utils.to_categorical(test_labels, weightSize1)


#The keras model
model = tf.keras.Sequential()

model.add(layers.Flatten(input_shape=input_shape))

model.add(layers.Dense(weightSize2, activation='softmax'))

model.add(layers.Dense(weightSize1, activation='softmax'))




#Compile the keras model
model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adadelta(),
              metrics=['accuracy'])

#Print a model summary
model.summary()


"""
model.fit(train_imgs, train_labels,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(test_imgs, test_labels))

"""
#Evaluating the keras model
score = model.evaluate(test_imgs, test_labels, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])




#Establishing the session
sess = tf.compat.v1.Session()


"""
#Get the output of the final layer
inp = model.input                                           # input placeholder
outputs = [layer.output for layer in model.layers]          # all layer outputs
#functors = [tf.keras.backend.function([inp, tf.keras.backend.learning_phase()], [out]) for out in outputs]    # evaluation functions
functor = tf.keras.backend.function([inp, tf.keras.backend.learning_phase()], outputs )

# Testing
test = np.random.random(input_shape)[np.newaxis,...]
#layer_outs = [func([test, 1.]) for func in functors]
external_weights = functor([test, 1.])
#print (layer_outs)

"""




#gewt the output of the final layer
tryHoldOutput = model.layers[2].output


print("\n\n\n\nTHE OUTPUT: ",tryHoldOutput)
print(tryHoldOutput[0])

external_weights = tryHoldOutput[0]




"""
for i in range(3):
    print("\n\nLAYER ", i)

    testIfDrop = model.layers[i].name.find("dropout")
    #print(layer_outs[i].name)
    print(external_weights[i])
    print(model.layers[i].name)


    if testIfDrop != -1:
        print("qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq")
        external_weights[i].fill(0)
        print(external_weights[i])
        
"""





print("\n##########################################################################################################\n")






###############################################DCS#pn##########################################################





#Variables for the DCSPN
reuse = 0
saveName = "testSave"
counter = 0 #For keeping track of where in the vector we are


SEED = 1234


img_height = 28 #the height in pixels of the input image
img_width = 28 #the width in pixels of the input image
img_channel = 1 #the number of channels of the input image. 
#Note: for greyscale img_channel = 1, for RGB imag_channel=3

valid_amount = 50

#The leaf components is the number of channels you want your tensor to have
#each pixel will have leaf_components number of means and std calculated for it
leaf_components = 4



#PROBLEM SOMEWHERE HERE
params_shape = (train_imgs.shape[1], train_imgs.shape[2], leaf_components)
leaf_means = np.zeros(params_shape)
leaf_stds = np.zeros(params_shape)
#sorted_data = np.sort(train_imgs, axis=0)

sorted_data = train_imgs


quantile_size = train_imgs.shape[0] / leaf_components 



#Get the leaf means and stds
for k in range(leaf_components):
    lower_idx = int(k * quantile_size)
    upper_idx = int((k + 1) * quantile_size)


    slice_data = sorted_data[lower_idx:upper_idx, :, :, :]

    
    leaf_means[:, :, k] = np.reshape(
        np.mean(slice_data, axis=0),
        (params_shape[0], params_shape[1]))
    _std = np.std(slice_data, axis=0)
    _std[_std == 0] = 1
    leaf_stds[:, :, k] = np.reshape(
        _std, (params_shape[0], params_shape[1]))
    





print("\nmeans shape")
print(leaf_means.shape)
print(leaf_stds.shape)






print("LOAD")


print("\n\n\n\n\n\n\n\n")



initializer = "uniform" #for the layer parameters

#The share parameter is related to the weights. 
#If share_parameter is false than the weights tensor will be the same size as the sum layer tensor
#If share_parameter is true that the weights tensor is 1x1xc where c is then number of leaf components. 
#In this case the weights are the same for all input pixels
share_parameters = False

#Determines if the sum node will perform of max or sum operation
#if false it will perform a sum, if true if will take the maximum child
hard_inference = False

#saves the shape of the input
input_shape = [img_height, img_width, img_channel]


#create an spn, this will hold all of the layers and be used for training and inferences
spn = SumProductNetwork(input_shape=input_shape, reuseValue=reuse)

# Build leaf layer
leaf_layer = GaussianLeafLayer(num_leaf_components=leaf_components)#these leaves will have the Gaussian distribution
#Each leaf, there will be 4/pixel as there are 4 leaf_components, will be a Gaussian distribution using the means and stds previously calculated


spn.add_layer(leaf_layer)#adds the leaf layer to the graph
spn.set_leaf_layer(leaf_layer)#sets it as the leaf layer



#create a sum layer
#A sum layer, for each pixel, combines the channels of that pixel by either adding all the values togethher or choosing the maximum value
#the sum layer will shrink the number of channels but not change the width or height
#if oc is the number of out channels and c is the number of original channels then the sum layer changed from wxhxc to wxhxoc
#sum_layer = SumLayer(out_channels=10,  #out_channels chooses how many output channnels there will be

sum_layer = SumLayer(out_channels=11,  #out_channels chooses how many output channnels there will be
                    hard_inference=hard_inference,
                    share_parameters=share_parameters,
                    initializer=initializer)
#because this is a toy example there is only one out channel as we only want one sum layer. The hard_inference and share_parameters are described above
#One out channel means that the tensor will be reduced from wxhxc to wxhx1


#connect the sum layer to the tree, the sum layer is the parent of the leaf layer
spn.add_forward_layer_edge(leaf_layer, sum_layer)
#this builds a tree upwards, so leaf_layer is the child of the sum_layer, building up towards the root




#the pool window is used by the product layer
#the size of the pooling window setermines which pixels will be filtered together
#The size of the pooling window determines the width and height of the new tensor that is produced
#if pwxph is the size of the pooling window and the input tensor is wxhxc then the new tensor is (w/pw)x(h/ph)xh
pool_window = (img_width,img_height)
#Because this is a toy example and only one product layer is desired the size of the pooling window should be thhe same as the size of the input

sum_pooling = True #to create and average sum before pooling


#create a product layer
#A product layer is like a filter layer in a CNN. It reduced the height and width based off the size of the pooling window
#The number of channels will remain the same
product_layer = ProductLayer(pooling_size=pool_window,
                              sum_pooling=sum_pooling)
#add the product layer to the graph. It will be the parent of the sum layer
spn.add_forward_layer_edge(sum_layer, product_layer)





root_layer = SumLayer(out_channels=1,  #out_channels chooses how many output channnels there will be
                    hard_inference=hard_inference,
                    share_parameters=share_parameters,
                    initializer=initializer)

spn.add_forward_layer_edge(product_layer, root_layer)



#The product layer is now 1x1x1, meaning it can no longer be reduced
#This means that it is the root layer an needs to be assigned as such
#You can create a special layer just to be the root if you wish but it is not necessary
spn.set_root_layer(root_layer)
print("The model is now created. Its name is spn.")


#This model has 3 layers: the leaf layer, a sum layer  and a product layer
#The leaf layer contains the values from the input and is a tensor of size wxhxc where w and h are the dimensions of the input and c is the number of leaf_components
#The next layer is a sum layer which sums together the channels. Because this is a toy example the sum layer here only has one out channel in order to shrink the model
#In the sum layer the model goes from wxhxc to wxhx1 with only one channel as the output 
#The final layer is the product layer. Once again, because this is a toy example only one product layer is desired. 
#As such the pooling window is the same size as the input(wxh) so the tensor goes from size wxhx1 to 1x1x1
#Because the size is 1x1x1 this layer cannot be reduced therefore this layer is also the root
#It must be set as the root so that the spn is aware of it


print("Starting to fit the model")

backward_masks = None #a mask for inferences, chooses which child is active
#All children of a product node are active. The max child of a sum node is active

forward = None


#defining loss function and optimizer, calls build forward to build tensorflow graph
#Before this point the layers were established but have no values
forward = spn.compile(learning_rate=0.01,
                      optimizer="adam",
                      reuse=False,
                      external_weights=external_weights)




#For fitting and getting accuracy. Need the extra two inputs to feed into thhe keras graph
feed_means_stds = {
    spn.leaf_layer.means: leaf_means,
    spn.leaf_layer.stds: leaf_stds,
    model.layers[0].input: feed_labels,
    model.layers[0].input: train_imgs}





#the fit function is what does the actual learning. It will perform a forward pass then perform gradiaent decent
#This loop will repeat within the function for the specified number of epochs
#The costs returned is the error over time, in other words, how different the produced images are from the correct image
#The cost is Negative Log-Likelihood(NLL)
spn.fit(train_data=train_imgs, epochs=1, add_to_feed=feed_means_stds, minibatch_size=64)


#raise ValueError





#MPE is Most Probable Explanation
print("\nMPE inference")

#marginalize half of the images 
#variables for storing the maginalized images
eval_data_marg = None

#marginalize the left side all images, the algorithms job is to complete them

eval_data_marg = Database.marginalize(
    test_imgs, ver_idxs=[[0, img_height]],
    hor_idxs=[[0, int(img_width / 2)]])



#mpe inference
spn_input = spn.inputs_marg
#constrct a backwards mask, choosing which child is activated
if backward_masks is None:

    print("\n\nBMN!!!!")
    spn_input = spn.inputs
    backward_masks = spn.build_backward_masks(forward)
mpe_leaf = spn.build_mpe_leaf(backward_masks, replace_marg_vars=spn_input)


print(spn_input)

#print(spn.inputs.eval(session=sess))
#print(spn.inputs[forward_input]["output"])


#perform the forward inference, getting the value that ends up in the root
root_values = spn.forward_inference(
    forward, eval_data_marg, add_to_feed=feed_means_stds,
    alt_input=spn_input)

#get the NLL value for the validation data
root_value = -1.0 * np.mean(root_values)
print('{"metric": "Val NLL", "value": %f}' % (root_value))



#BELOW HERE JUST OUTPUTTING IMAGE


#get the MPE 
mpe_assignment = spn.mpe_inference(
    mpe_leaf, eval_data_marg, add_to_feed=feed_means_stds,
    alt_input=spn_input)


unorm_eval_data = test_imgs











# MSE is Mean Squared Error
print("Computing Mean Square Error")

#where the images will be saved too, must already have the folder toy inside the folder outputs
save_imgs_path = "outputs/toy"

#Get the mean squared error
#mse = Database.mean_square_error(
#    mpe_assignment, unorm_eval_data, save_imgs_path=save_imgs_path)

#print("MSE: {}".format(mse))








#mpe_assignment, unorm_eval_data, save_imgs_path=save_imgs_path
#def mean_square_error(data, data_target, save_imgs_path=None):


# MSE needs data as 2D images
if len(mpe_assignment.shape) == 4:
    data = np.squeeze(mpe_assignment)
    unorm_eval_data = np.squeeze(unorm_eval_data)

# Simple MSE function
def _mse(d1, d2):
    # Shape of d1 and d2 should be [H, W]
    mse = np.mean(
        ((255 * d2).astype(dtype="int") -
            (255 * d1).astype(dtype="int")) ** 2)
    return mse



mse_img = {
    _mse(data[img_idx, :, :],
            unorm_eval_data[img_idx, :, :]): img_idx
    for img_idx in range(data.shape[0])
}
log_mse = open("{}/mse_log.txt".format(save_imgs_path), "w")

for idx, img_mse in enumerate(reversed(sorted(mse_img.keys()))):
    Database.save_image(
        "{}/{}.png".format(save_imgs_path, idx),
        data[mse_img[img_mse], :, :])
    log_mse.write("{},{}\n".format(idx, img_mse))
log_mse.close()




#Print information about the DCSPN
print("\n\n\nNOW THE DIMENSIONS")
print("layer 0- leaf layer")
print(spn.layers[0])
#print(spn.layers[0].shape)
#print(spn.layers[0].weights)

print("\nlayer 1- sum layer")
print(spn.layers[1])
#print(spn.layers[1].compute_output_shape(spn.layers[1]))
print(spn.layers[1].weights)

print("\nlayer 2- product layer")
print(spn.layers[2])
#print(spn.layers[2].shape)
#print(spn.layers[2].weights)

print("\nlayer 3- root layer")
print(spn.layers[3])
#print(spn.layers[3].shape)
print(spn.layers[3].weights)





























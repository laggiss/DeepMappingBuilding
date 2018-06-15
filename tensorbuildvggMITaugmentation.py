import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Flatten, Dropout
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from keras.layers import Activation
from keras.wrappers.scikit_learn import KerasClassifier
from scipy import misc
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import Concatenate
from keras.optimizers import RMSprop,SGD
from keras import backend as K
from keras.constraints import maxnorm
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

import numpy as np
import csv
import glob
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
import cv2

import dataset
import tensorflow as tf
K.set_image_dim_ordering('tf')

# wts3 is the final w1w2

# wts4 is the w1w2urml
currentWeights="C:/GIST/tensorflow/modelsave/modelwtsall1246layersnotest.h5"
trainingDataPath="C:/GIST/tensorflow/buildings/training_data"
classes = ['URML','URMM']#['C1H','C1L','C1M','W1','W2','URML','URMM']'W1','W2',

saveWeightsPath="c:/gist/tensorflow/modelsave/modelwtsall1246layersnotest.h5"
saveModelPath="c:/gist/tensorflow/modelsave/model1246layers.json"

def compute_accuracy(predictions, labels):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return labels[predictions.ravel() < 0.5].mean()
#tf.device = '/cpu:0'
# Number of color channels for the images: 1 channel for gray-scale.
num_channels = 3

# image dimensions (only squares for now)
img_size = 124

# Size of image when flattened to a single dimension
img_size_flat = img_size * img_size * num_channels

# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size)

# class info

num_classes = len(classes)

# Data step
train_path=trainingDataPath
#test_path='c:/gist/tensorflow/testing_data'


data = dataset.read_train_sets(train_path, img_size, classes, validation_size=0,bcropprop=.1)
#test_images, test_ids = dataset.read_test_set(test_path, img_size,classes)
#testdata = dataset.read_train_sets(test_path, img_size,classes,validation_size=0)

X_train=data.train.images
y_train=data.train.labels
# X_test=data.valid.images
# y_test = data.valid.labels

#input_shape=data.train.images.shape


print("Size of:")
print("- Training-set:\t\t{}".format(len(data.train.labels)))
#print("- Test-set:\t\t{}".format(len(test_images)))
print("- Validation-set:\t{}".format(len(data.valid.labels)))

#with tf.device('/gpu:0'):
dropout=0.50
model=Sequential()

model.add(Conv2D(64, (3, 3), input_shape=(img_size,img_size,3),padding="same",name='block_1_conv_1'))
model.add(Activation('relu',name='block_1_actv_1'))
model.add(ZeroPadding2D(name='block_1_pad_1'))
model.add(Conv2D(64, (3, 3),name='block_1_conv_2'))
model.add(Activation('relu',name='block_1_actv_2'))
model.add(MaxPooling2D(pool_size=(2,2),data_format="channels_last",name='block_1_pool_1'))
model.add(ZeroPadding2D(name='block_1_pad_2'))
model.add(Dropout(dropout,name='block_1_dropout_1'))


model.add(Conv2D(128, (3, 3),name='block_2_conv_1'))
model.add(Activation('relu',name='block_2_actv_1'))
model.add(ZeroPadding2D(name='block_2_pad_1'))
model.add(Conv2D(128, (3, 3),name='block_2_conv_2'))
model.add(Activation('relu',name='block_2_actv_2'))
model.add(MaxPooling2D(pool_size=(2,2),data_format="channels_last",name='block_2_pool_1'))
model.add(ZeroPadding2D(name='block_2_pad_2'))
model.add(Dropout(dropout,name='block_2_dropout_1'))


model.add(Conv2D(256, (3, 3),name='block_3_conv_1'))
model.add(Activation('relu',name='block_3_actv_1'))
model.add(ZeroPadding2D(name='block_3_pad_1'))
model.add(Conv2D(256, (3, 3),name='block_3_conv_2'))
model.add(Activation('relu',name='block_3_actv_2'))
model.add(ZeroPadding2D(name='block_3_pad_2'))
model.add(Conv2D(256, (3, 3),name='block_3_conv_3'))
model.add(Activation('relu',name='block_3_actv_3'))
model.add(ZeroPadding2D(name='block_3_pad_3'))
model.add(Conv2D(256, (3, 3),name='block_3_conv_4'))
model.add(Activation('relu',name='block_3_actv_4'))
model.add(MaxPooling2D(pool_size=(2,2),data_format="channels_last",name='block_3_pool_1'))
model.add(ZeroPadding2D(name='block_3_pad_4'))
model.add(Dropout(dropout,name='block_3_dropout_1'))

model.add(Conv2D(512, (3, 3),name='block_4_conv_1'))
model.add(Activation('relu',name='block_4_actv_1'))
model.add(ZeroPadding2D(name='block_4_pad_1'))
model.add(Conv2D(512, (3, 3),name='block_4_conv_2'))
model.add(Activation('relu',name='block_4_actv_2'))
model.add(ZeroPadding2D(name='block_4_pad_2'))
model.add(Conv2D(512, (3, 3),name='block_4_conv_3'))
model.add(Activation('relu',name='block_4_actv_3'))
model.add(ZeroPadding2D(name='block_4_pad_3'))
model.add(Conv2D(512, (3, 3), name='block_4_conv_4'))
model.add(Activation('relu',name='block_4_actv_4'))
model.add(MaxPooling2D(pool_size=(2,2),data_format="channels_last",name='block_4_pool_1'))
model.add(ZeroPadding2D(name='block_4_pad_4'))
model.add(Dropout(dropout,name='block_4_dropout_1'))


model.add(Conv2D(512, (3, 3),name='block_5_conv_1'))
model.add(Activation('relu',name='block_5_actv_1'))
model.add(ZeroPadding2D(name='block_5_pad_1'))
model.add(Conv2D(512, (3, 3),name='block_5_conv_2'))
model.add(Activation('relu',name='block_5_actv_2'))
model.add(ZeroPadding2D(name='block_5_pad_2'))
model.add(Conv2D(512, (3, 3),name='block_5_conv_3'))
model.add(Activation('relu',name='block_5_actv_3'))
model.add(ZeroPadding2D(name='block_5_pad_3'))
model.add(Conv2D(512, (3, 3),name='block_5_conv_4'))
model.add(Activation('relu',name='block_5_actv_4'))
model.add(MaxPooling2D(pool_size=(2,2),data_format="channels_last",name='block_5_pool_1'))

model.add(ZeroPadding2D(name='block_6_pad_4'))
model.add(Dropout(dropout,name='block_6_dropout_1'))

model.add(Conv2D(512, (3, 3),name='block_6_conv_1'))
model.add(Activation('relu',name='block_6_actv_1'))
model.add(ZeroPadding2D(name='block_6_pad_1'))
model.add(Conv2D(512, (3, 3),name='block_6_conv_2'))
model.add(Activation('relu',name='block_6_actv_2'))
model.add(ZeroPadding2D(name='block_6_pad_2'))
model.add(Conv2D(512, (3, 3),name='block_6_conv_3'))
model.add(Activation('relu',name='block_6_actv_3'))
model.add(ZeroPadding2D(name='block_6_pad_3'))
model.add(Conv2D(512, (3, 3),name='block_6_conv_4'))
model.add(Activation('relu',name='block_6_actv_4'))
model.add(MaxPooling2D(pool_size=(2,2),data_format="channels_last",name='block_6_pool_1'))


model.add(Flatten(name='flatten_1'))
model.add(Dense(4096,name='fc_1'))#, kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01)
model.add(Activation('relu',name='fc_actv_1'))
model.add(Dropout(dropout,name='fc_dropout_1'))
model.add(Dense(4096,name='fc_2'))
model.add(Activation('relu',name='fc_actv_2'))
model.add(Dropout(dropout,name='fc_dropout_2'))
model.add(Dense(1000,name='fc_6'))
model.add(Activation('relu',name='fc_actv_6'))
model.add(Dropout(dropout,name='fc_dropout_6'))
model.add(Dense(num_classes,name='fc_7'))
model.add(Activation('softmax',name='fc_actv_7'))

# for layer in model.layers[:16]:
#     layer.trainable = False
#
# #Load previous model weights
#
model.load_weights(currentWeights,by_name=True)


sgd = SGD(lr=0.001, decay=1e-6, momentum=0.5, nesterov=True)


model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])


    # batch size
batch_size = 16

#####
## No augmentation
#####

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.5,
        horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        r'C:\GIST\tensorflow\buildings\augmentation\training_data',  # this is the target directory
        target_size=(124, 124),  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='categorical')#,save_to_dir=r'C:\GIST\tensorflow\buildings\augmentation\preview')  # since we use binary_crossentropy loss, we need binary labels

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        r'C:\GIST\tensorflow\buildings\augmentation\validation_data',
        target_size=(124, 124),
        batch_size=batch_size,
        class_mode='categorical')


    # number of epochs
n_epoch = 100

mout=model.fit_generator(
        train_generator,
        steps_per_epoch=5000 // batch_size,
        epochs=n_epoch,
        validation_data=validation_generator,
        validation_steps=800 // batch_size,verbose=2)




#mout=model.fit(X_train,y_train,batch_size=batch_size,epochs=n_epoch,verbose=2,validation_split=validation_size)#,callbacks=[reduce_lr,SGDLearningRateTracker()])#validation_data=(X_test,y_test))

import matplotlib.pyplot as plt
plt.plot(mout.history['acc'])
plt.plot(mout.history['val_acc'])
max(mout.history['val_acc'])

## model testing
##
test_path='c:/gist/tensorflow/buildings/testing_data'
testdata = dataset.read_train_sets(test_path, img_size,classes,validation_size=0)
X_test=testdata.train.images
Y_test = testdata.train.labels
pred = model.predict(X_test,batch_size=batch_size)

# For 0 and 1 prediction run predv=[round(x[0]) for x in pred] and [x[0] for x in testdata.train.labels]
with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())

    predv = tf.argmax(pred,1).eval()
    truev = tf.argmax(Y_test,1).eval()
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
cmat=confusion_matrix(predv,truev)
print(cmat)
kappa=cohen_kappa_score(predv,truev)
print("Kappa: %".format(kappa))


mev=model.evaluate(X_test,Y_test,batch_size=batch_size)
print(mev)

# # Visualize model
# import graphviz
# import pydot
# from keras.utils import plot_model
# plot_model(model, to_file='c:/temp/model2.png',show_shapes=True)
#
## Saving model and weights

# serialize model to JSON
model_json = model.to_json()
with open(saveModelPath, "w") as json_file:
    json_file.write(model_json)
#
# Save model weights
import h5py
model.save_weights(saveWeightsPath)
print("Saved model to disk")



# pred = model.predict(testdata.train.images,batch_size=batch_size)
#
# # For 0 and 1 prediction run predv=[round(x[0]) for x in pred] and [x[0] for x in testdata.train.labels]
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#
#     predv = tf.argmax(pred,1).eval()
#     truev = tf.argmax(testdata.train.labels,1).eval()
#
# from sklearn.metrics import cohen_kappa_score
# from sklearn.metrics import confusion_matrix
# cmat=confusion_matrix(predv,truev)
# print(cmat)
# kappa=cohen_kappa_score(predv,truev)
# print("Kappa: %".format(kappa))
# mev=model.evaluate(X_train,y_train,batch_size=batch_size)
# print(mev)
#pev = model.evaluate(X_test,y_test,batch_size=batch_size)



import numpy as np
from matplotlib import pyplot as plt

from vis.utils import utils
from vis.utils.vggnet import VGG16
#from vis.visualization import visualize_class_activation, get_num_filters
from vis.visualization import visualize_class_activation,get_num_filters
# Build the VGG16 network with ImageNet weights
#model = mout#VGG16(weights='imagenet', include_top=True)
print('Model loaded.')

# The name of the layer we want to visualize
# (see model definition in vggnet.py)
layer_name = 'vizblock'
layer_idx = 4#[idx for idx, layer in enumerate(model.layers) if layer.name == layer_name][0]

# Visualize all filters in this layer.
filters = np.arange(get_num_filters(model.layers[layer_idx]))

# Generate input image for each filter. Here `text` field is used to overlay `filter_value` on top of the image.
vis_images = []
for idx in range(5):
    img = visualize_class_activation(model, layer_idx, filter_indices=idx)
    #img = utils.draw_text(img, str(idx))
    vis_images.append(img)

# Generate stitched image palette with 8 cols.
stitched = utils.stitch_images(vis_images, cols=8)
plt.axis('off')
plt.imshow(stitched)
plt.title(layer_name)
plt.show()


from matplotlib import pyplot as plt

from vis.utils import utils
from vis.visualization import visualize_class_activation


# Build the VGG16 network with ImageNet weights
#model = VGG16(weights='imagenet', include_top=True)
print('Model loaded.')

# The name of the layer we want to visualize
# (see model definition in vggnet.py)
layer_name = 'predictions'
layer_idx = 5#[idx for idx, layer in enumerate(model.layers) if layer.name == layer_name][0]

# Generate three different images of the same output index.
vis_images = []
for idx in [0, 0, 0]:
    img = visualize_class_activation(model, layer_idx, filter_indices=idx, max_iter=500)
    #img = utils.draw_text(img, str(idx))
    vis_images.append(img)

stitched = utils.stitch_images(vis_images)
plt.axis('off')
plt.imshow(stitched)
plt.title(layer_name)
plt.show()



import numpy as np
from matplotlib import pyplot as plt

from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import preprocess_input

from vis.utils import utils
#from vis.utils.vggnet import VGG16
import vis.visualization
from vis.visualization import visualize_saliency,overlay
from vis.visualization import visualize_cam

# Build the VGG16 network with ImageNet weights

print('Model loaded.')

# The name of the layer we want to visualize
# (see model definition in vggnet.py)
layer_name = 'fc_2'
#layer_idx = 80#idx for idx, layer in enumerate(model.layers) if layer.name == layer_name][0]
for l in enumerate(model.layers):
    if l[1].name == layer_name:
        layer_idx=l[0]

# Images corresponding to tiger, penguin, dumbbell, speedboat, spider
image_paths = [
#     "C:/GIST/tensorflow/buildings/training_data/C1H/1746.jpg",
# "C:/GIST/tensorflow/buildings/training_data/C1L/1762.jpg",
# "C:/GIST/tensorflow/buildings/training_data/C1M/763.jpg",
      "C:/GIST/tensorflow/buildings/training_data/W2/410.jpg",
    "C:/GIST/tensorflow/buildings/training_data/W2/835.jpg",
    "C:/GIST/tensorflow/buildings/training_data/W2/704.jpg",
#      "C:/GIST/tensorflow/buildings/training_data/URML/2119.jpg",
#      "C:/GIST/tensorflow/buildings/training_data/URMM/314.jpg",
"C:/GIST/tensorflow/buildings/training_data/W1/174.jpg",
"C:/GIST/tensorflow/buildings/training_data/W1/483.jpg",
     "C:/GIST/tensorflow/buildings/training_data/W1/596.jpg"
# #"C:/GIST/tensorflow/buildings/training_data/W2/365.jpg"
]

heatmaps = []
for path in image_paths:
    seed_img = utils.load_img(path, target_size=(124, 124))
    x = np.expand_dims(img_to_array(seed_img), axis=0)
    x = preprocess_input(x)
    pred_class = np.argmax(model.predict(x))

    # Here we are asking it to show attention such that prob of `pred_class` is maximized.
    heatmap = visualize_saliency(model, layer_idx, [pred_class], seed_img)

    # Overlay heatmap onto the image with alpha blend.
    heatmaps.append(overlay(seed_img, heatmap))

plt.axis('off')
plt.imshow(utils.stitch_images(heatmaps))
plt.title('Saliency map')
plt.show()



# VIZ CAM

layer_name = 'predictions'
layer_idx = 24#idx for idx, layer in enumerate(model.layers) if layer.name == layer_name][0]

# Images corresponding to tiger, penguin, dumbbell, speedboat, spider
image_paths = [
    "C:/GIST/tensorflow/buildings/training_data/W2/360.jpg",
   # "C:/GIST/tensorflow/buildings/training_data/URML/2236.jpg",
   # "C:/GIST/tensorflow/buildings/training_data/URMM/314.jpg",
    "C:/GIST/tensorflow/buildings/training_data/W1/170.jpg"
#"C:/GIST/tensorflow/buildings/training_data/W2/365.jpg"
]

heatmaps = []
for path in image_paths:
    seed_img = utils.load_img(path, target_size=(124, 124))
    x = np.expand_dims(img_to_array(seed_img), axis=0)
    x = preprocess_input(x)
    pred_class = np.argmax(model.predict(x))
    layer_name = 'dense_1'
    layer_idx = [idx for idx, layer in enumerate(model.layers)
                 if layer.name == layer_name][0]

    # Here we are asking it to show attention such that prob of `pred_class` is maximized.
    heatmap = visualize_cam(model, layer_idx, [pred_class], seed_img)

    # Overlay heatmap onto the image with alpha blend.
    heatmaps.append(overlay(seed_img, heatmap))

plt.axis('off')
plt.imshow(utils.stitch_images(heatmaps))
plt.title('Saliency map')
plt.show()




from keras.applications import VGG16
from vis.utils import utils
from keras import activations

# Utility to search for layer index by name.
# Alternatively we can specify this as -1 since it corresponds to the last layer.
layer_idx = 67#utils.find_layer_idx(model, 'fc_actv_8')

# Swap softmax with linear
model.layers[layer_idx].activation = activations.linear
model = utils.apply_modifications(model)

from vis.visualization import visualize_activation

from matplotlib import pyplot as plt

plt.rcParams['figure.figsize'] = (18, 6)

# 20 is the imagenet category for 'ouzel'
img = visualize_activation(model, layer_idx, filter_indices=3)
plt.imshow(img)


idx=59
# Generate input image for each filter.
new_vis_images = []
for i, idx in enumerate(filters):
    # We will seed with optimized image this time.
    img = visualize_activation(model, layer_idx, filter_indices=idx,
                               seed_input=vis_images[i],
                               input_modifiers=[Jitter(0.05)])

    # Utility to overlay text on image.
    img = utils.draw_text(img, 'Filter {}'.format(idx))
    new_vis_images.append(img)

# Generate stitched image palette with 5 cols so we get 2 rows.
stitched = utils.stitch_images(new_vis_images, cols=5)
plt.figure()
plt.axis('off')
plt.imshow(stitched)
plt.show()



from vis.visualization import get_num_filters
selected_indices = []
for layer_name in ['block_6_conv_3']:
    layer_idx = utils.find_layer_idx(model, layer_name)

    # Visualize all filters in this layer.
    filters = np.random.permutation(get_num_filters(model.layers[layer_idx]))[:10]
    selected_indices.append(filters)

    # Generate input image for each filter.
    vis_images = []
    for idx in filters:
        img = visualize_activation(model, layer_idx, filter_indices=idx)

        # Utility to overlay text on image.
        img = utils.draw_text(img, 'Filter {}'.format(idx))
        vis_images.append(img)

    # Generate stitched image palette with 5 cols so we get 2 rows.
    stitched = utils.stitch_images(vis_images, cols=5)
    plt.figure()
    plt.axis('off')
    plt.imshow(stitched)
    plt.show()


from keras.applications import VGG16
from vis.utils import utils
from keras import activations
from vis.input_modifiers import Jitter
# Utility to search for layer index by name.
# Alternatively we can specify this as -1 since it corresponds to the last layer.
layer_idx = utils.find_layer_idx(model, 'block_6_actv_4')

# Swap softmax with linear
model.layers[layer_idx].activation = activations.linear
model = utils.apply_modifications(model)

from vis.visualization import visualize_activation

from matplotlib import pyplot as plt

#plt.rcParams['figure.figsize'] = (18, 6)

# 20 is the imagenet category for 'ouzel'
img = visualize_activation(model, layer_idx, filter_indices=3,max_iter=500, input_modifiers=[Jitter(16)], verbose=True)
plt.imshow(img)






import numpy as np

categories = np.random.permutation(num_classes)[:num_classes]
layer_idx = utils.find_layer_idx(model, 'block_6_actv_4')
vis_images = []
image_modifiers = [Jitter(16)]
for idx in categories:
    img = visualize_activation(model, layer_idx, filter_indices=idx, max_iter=500, input_modifiers=image_modifiers)

    # Reverse lookup index to imagenet label and overlay it on the image.
    img = utils.draw_text(img, classes[idx])
    vis_images.append(img)

# Generate stitched images with 5 cols (so it will have 3 rows).
plt.rcParams['figure.figsize'] = (50, 50)
stitched = utils.stitch_images(vis_images, cols=5)
plt.axis('off')
plt.imshow(stitched)
plt.show()
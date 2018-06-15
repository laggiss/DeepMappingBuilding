from __future__ import absolute_import
from __future__ import print_function

import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.callbacks import ModelCheckpoint
from keras.engine.topology import get_source_inputs
from keras.layers import Activation, BatchNormalization
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten, Dropout
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.models import Model
from keras.models import Sequential
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import layer_utils
from keras.utils.data_utils import get_file

import dataset

WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = 'C:/GIST/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'


def plot_con_mat(pred, YY_test, norm=False):
    import itertools
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    # For 0 and 1 prediction run predv=[round(x[0]) for x in pred] and [x[0] for x in testdata.train.labels]
    with tf.Session() as sess:
        #     sess.run(tf.global_variables_initializer())
        predv = tf.argmax(pred, 1).eval()
        truev = tf.argmax(YY_test, 1).eval()
    cmat = confusion_matrix(predv, truev)

    def plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig('c:/temp/fig.png',dpi=150)
        plt.show()

    plt.figure(1)
    plot_confusion_matrix(cmat, classes, norm)

    return predv, truev


def VGG19_Wz(include_top=False, weights='imagenet',
             input_tensor=None, input_shape=None,
             pooling=None,
             classes=1000):
    """Instantiates the VGG19 architecture.

    Optionally loads weights pre-trained
    on ImageNet. Note that when using TensorFlow,
    for best performance you should set
    `image_data_format='channels_last'` in your Keras config
    at ~/.keras/keras.json.

    The model and the weights are compatible with both
    TensorFlow and Theano. The data format
    convention used by the model is the one
    specified in your Keras config file.

    # Arguments
        include_top: whether to include the 3 fully-connected
            layers at the top of the network.
        weights: one of `None` (random initialization)
            or 'imagenet' (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 48.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')
    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=48,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    if include_top:
        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='vgg19')

    # load weights
    if weights == 'imagenet':
        if include_top:
            weights_path = get_file('vgg19_weights_tf_dim_ordering_tf_kernels.h5',
                                    WEIGHTS_PATH,
                                    cache_subdir='models',
                                    file_hash='cbe5617147190e668d6c5d5026f83318')
        else:
            weights_path = WEIGHTS_PATH_NO_TOP
            # get_file('vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5',
            #                     WEIGHTS_PATH_NO_TOP,
            #                     cache_subdir='models',
            #                     file_hash='253f8cb515780f3b799900260a226db6')
        model.load_weights(weights_path)
        if K.backend() == 'theano':
            layer_utils.convert_all_kernels_in_model(model)

        if K.image_data_format() == 'channels_first':
            if include_top:
                maxpool = model.get_layer(name='block5_pool')
                shape = maxpool.output_shape[1:]
                dense = model.get_layer(name='fc1')
                layer_utils.convert_dense_weights_data_format(dense, shape, 'channels_first')

            if K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image data format convention '
                              '(`image_data_format="channels_first"`). '
                              'For best performance, set '
                              '`image_data_format="channels_last"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')
    return model


def plot_acc_loss():
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rc('xtick', labelsize=8)
    matplotlib.rc('ytick', labelsize=8)
    plt.figure(2)
    loss = mout.history['loss']
    val_loss = mout.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.subplot(211)
    plt.plot(epochs, loss, 'b', label='Training loss', color='r')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss',fontsize=10,y=.97,x=.5)
    #plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(212)
    acc = mout.history['acc']
    val_acc = mout.history['val_acc']
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy',fontsize=10,y=.97,x=.5)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("c:/temp/fig.png",dpi=150)
    plt.show()


K.set_image_dim_ordering('tf')

# wts3 is the final w1w2
# wts4 is the w1w2urml
currentWeights = "c:/gist/junweights.h5"  # ""C:/GIST/tensorflow/modelsave/model_augmentation_1.h5"
trainingDataPath = "C:/GIST/tensorflow/buildings/gentrain"
classes = ['C','S','W1', 'W2','URM']#['C1H', 'C1L', 'C1M', 'C2H', 'C2L', 'S1L', 'W1', 'W2', 'URML', 'URMM']#['C1H', 'C1L', 'C1M', 'C2H', 'C2L', 'S1L', 'W1', 'W2']  # 'URML', 'URMM']  # ,'URMM']#'W1','W2',

saveWeightsPath = "c:/gist/junweightsx.h5"  # currentWeights  # "c:/gist/best_build_weights2.h5"#"c:/gist/tensorflow/modelsave/model_augmentation_1.h5"
saveModelPath = "c:/gist/tensorflow/modelsave/model1246layersx.json"


def compute_accuracy(predictions, labels):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return labels[predictions.ravel() < 0.5].mean()


# tf.device = '/cpu:0'
# Number of color channels for the images: 1 channel for gray-scale.
num_channels = 3

# image dimensions (only squares for now)
img_size = 224

# Size of image when flattened to a single dimension
img_size_flat = img_size * img_size * num_channels

# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size)

# class info

num_classes = len(classes)

# Data step
train_path = trainingDataPath
# test_path='c:/gist/tensorflow/testing_data'


data = dataset.read_train_sets(train_path, img_size, classes, validation_size=.1, bcropprop=.1, randomstate=2)
# test_images, test_ids = dataset.read_test_set(test_path, img_size,classes)
# testdata = dataset.read_train_sets(test_path, img_size,classes,validation_size=0)

X_train = data.train.images
y_train = data.train.labels
X_test = data.valid.images
y_test = data.valid.labels

# from sklearn.utils import class_weight
# class_weight = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)


print("Size of:")
print("- Training-set:\t\t{}".format(len(data.train.labels)))
# print("- Test-set:\t\t{}".format(len(test_images)))
print("- Validation-set:\t{}".format(len(data.valid.labels)))

# sys.path.append(r"C:\Users\laggi\OneDrive\a11PyCharmCNN")
conv_base = VGG19_Wz(weights='imagenet', input_shape=(img_size, img_size, 3))
dropout = 0.33

# with tf.device('/gpu:0'):

model = Sequential()

model.add(conv_base)

model.add(Flatten(name='flatten_1'))
model.add(
    Dense(4096, name='fc_1'))  # , kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01)
model.add(Activation('relu', name='fc_actv_1'))
#model.add(Dropout(dropout,name='fc_dropout_1'))
model.add(Dense(4096, name='fc_2'))
model.add(Activation('relu', name='fc_actv_2'))
model.add(Dropout(dropout, name='fc_dropout_2'))
model.add(Dense(1000, name='fc_6'))
model.add(Activation('relu', name='fc_actv_6'))
model.add(BatchNormalization())
# model.add(Dropout(dropout,name='fc_dropout_6'))
model.add(Dense(num_classes, name='fc_7a'))
model.add(Activation('softmax', name='fc_actv_7a'))

# for layer in model.layers[:16]:
#     layer.trainable = False
#
# #Load previous model weights
#


model.load_weights(currentWeights, by_name=True)

# freeze all but top layers
# set_trainable = False
# for layer in conv_base.layers:
#     if layer.name == 'block5_conv1':
#         set_trainable = True
#     if set_trainable:
#         layer.trainable = True
#     else:
#         layer.trainable = False

conv_base.trainable = False

# for layer in model.layers:
#     layer.trainable=True


sgd = SGD(lr=0.001, decay=1e-6, momentum=0.4, nesterov=True)

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# batch size
batch_size = 16

# number of epochs
n_epoch = 30

datagen = ImageDataGenerator(
    rotation_range=2,
    width_shift_range=0.25,
    height_shift_range=.05,
    shear_range=.2,
    zoom_range=0.8,
    horizontal_flip=True,
    fill_mode='nearest'
    )

train_gen = datagen.flow(X_train, y_train, batch_size=batch_size)#
#xgen=datagen.flow_from_directory('c:/gist/tensorflow/buildings/gentrain/',shuffle=False,save_to_dir="c:/gist/del")

checkpointer = ModelCheckpoint(filepath=saveWeightsPath, verbose=1, save_best_only=True)
#classweights=[1,2,2,2,1]
#mout=model.fit(X_train,y_train,batch_size=batch_size,epochs=n_epoch,verbose=1,validation_split=.15)#,callbacks=[reduce_lr,SGDLearningRateTracker()])#validation_data=(X_test,y_test))
mout = model.fit_generator(generator=train_gen, steps_per_epoch=X_train.shape[0] // batch_size, epochs=n_epoch,
                           verbose=1, validation_data=(X_test, y_test),callbacks=[
        checkpointer])  # ,class_weight=classweights class_weight='auto', # ,callbacks=[reduce_lr,SGDLearningRateTracker()])#validation_data=(X_test,y_test))

# plot accuracy and loss
plot_acc_loss()

## model testing
##
test_path = 'c:/gist/tensorflow/buildings/gentest'
testdata = dataset.read_train_sets(test_path, img_size, classes, validation_size=0)
XX_test = testdata.train.images
YY_test = testdata.train.labels
model.load_weights(saveWeightsPath)
pred = model.predict(XX_test, batch_size=batch_size)
predv, truev = plot_con_mat(pred, YY_test, False)

# evaluate classes by class name
pt = []
for lab in predv:
    pt.append(classes[lab])
ptt = []
for lab in truev:
    ptt.append(classes[lab])
import pandas

comp = pandas.DataFrame.from_dict({'true': ptt, 'pred': pt})

import pandas as pd
from matplotlib import pyplot as plt

plt.figure(3)
with pd.option_context('display.max_rows', None, 'display.max_columns', 3):
    srt = comp.sort_values('true')
    print(srt.loc[srt.true == 'C1L'])

from sklearn.metrics import confusion_matrix

cmat = confusion_matrix(pt, ptt, labels=classes)

mev = model.evaluate(XX_test, YY_test, batch_size=batch_size)
print(mev)
from sklearn.metrics import cohen_kappa_score

kappa = cohen_kappa_score(predv, truev)
print("Kappa: {}".format(kappa))

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
# pev = model.evaluate(X_test,y_test,batch_size=batch_size)


import numpy as np
from matplotlib import pyplot as plt

from vis.utils import utils
# from vis.visualization import visualize_class_activation, get_num_filters
from vis.visualization import visualize_class_activation, get_num_filters

# Build the VGG16 network with ImageNet weights
# model = mout#VGG16(weights='imagenet', include_top=True)
print('Model loaded.')

# The name of the layer we want to visualize
# (see model definition in vggnet.py)
layer_name = 'vizblock'
layer_idx = 4  # [idx for idx, layer in enumerate(model.layers) if layer.name == layer_name][0]

# Visualize all filters in this layer.
filters = np.arange(get_num_filters(model.layers[layer_idx]))

# Generate input image for each filter. Here `text` field is used to overlay `filter_value` on top of the image.
vis_images = []
for idx in range(5):
    img = visualize_class_activation(model, layer_idx, filter_indices=idx)
    # img = utils.draw_text(img, str(idx))
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
# model = VGG16(weights='imagenet', include_top=True)
print('Model loaded.')

# The name of the layer we want to visualize
# (see model definition in vggnet.py)
layer_name = 'predictions'
layer_idx = 5  # [idx for idx, layer in enumerate(model.layers) if layer.name == layer_name][0]

# Generate three different images of the same output index.
vis_images = []
for idx in [0, 0, 0]:
    img = visualize_class_activation(model, layer_idx, filter_indices=idx, max_iter=500)
    # img = utils.draw_text(img, str(idx))
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
# from vis.utils.vggnet import VGG16
from vis.visualization import visualize_saliency, overlay
from vis.visualization import visualize_cam

# Build the VGG16 network with ImageNet weights

print('Model loaded.')

# The name of the layer we want to visualize
# (see model definition in vggnet.py)
layer_name = 'fc_actv_6'
# layer_idx = 80#idx for idx, layer in enumerate(model.layers) if layer.name == layer_name][0]
for l in enumerate(model.layers):
    if l[1].name == layer_name:
        layer_idx = l[0]

# Images corresponding to tiger, penguin, dumbbell, speedboat, spider
image_paths = [
    #     "C:/GIST/tensorflow/buildings/training_data/C1H/1746.jpg",
    # "C:/GIST/tensorflow/buildings/training_data/C1L/1762.jpg",
    # "C:/GIST/tensorflow/buildings/training_data/C1M/763.jpg",
    "C:/GIST/tensorflow/buildings/training_data/W2/410.jpg",
    "C:/GIST/tensorflow/buildings/training_data/W2/835.jpg",
    "C:/GIST/tensorflow/buildings/training_data/W2/704.jpg"
    #      "C:/GIST/tensorflow/buildings/training_data/URML/2119.jpg",
    #      "C:/GIST/tensorflow/buildings/training_data/URMM/314.jpg",
    # "C:/GIST/tensorflow/buildings/training_data/W1/174.jpg",
    # "C:/GIST/tensorflow/buildings/training_data/W1/483.jpg",
    # "C:/GIST/tensorflow/buildings/training_data/W1/596.jpg"
    # #"C:/GIST/tensorflow/buildings/training_data/W2/365.jpg"
    ]

heatmaps = []
for path in image_paths:
    seed_img = utils.load_img(path, target_size=(224, 224))
    x = np.expand_dims(img_to_array(seed_img), axis=0)
    x = preprocess_input(x)
    pred_class = np.argmax(model.predict(x))

    # Here we are asking it to show attention such that prob of `pred_class` is maximized.
    heatmap = visualize_saliency(model, layer_idx, [pred_class], seed_img)

    # Overlay heatmap onto the image with alpha blend.
    heatmaps.append(overlay(seed_img, heatmap))

plt.axis('off')
plt.imshow(utils.stitch_images(heatmaps))
plt.savefig("c:/temp/sal.png",dpi=200)
plt.title('Saliency map')
plt.show()

# VIZ CAM

layer_name = 'predictions'
layer_idx = 0  # idx for idx, layer in enumerate(model.layers) if layer.name == layer_name][0]

# Images corresponding to tiger, penguin, dumbbell, speedboat, spider
image_paths = [
    "C:/GIST/tensorflow/buildings/training_data/W2/835.jpg",
    # "C:/GIST/tensorflow/buildings/training_data/URML/2236.jpg",
    # "C:/GIST/tensorflow/buildings/training_data/URMM/314.jpg",
    "C:/GIST/tensorflow/buildings/training_data/W1/170.jpg"
    # "C:/GIST/tensorflow/buildings/training_data/W2/365.jpg"
    ]

heatmaps = []
for path in image_paths:
    seed_img = utils.load_img(path, target_size=(224, 224))
    x = np.expand_dims(img_to_array(seed_img), axis=0)
    x = preprocess_input(x)
    pred_class = np.argmax(model.predict(x))
    layer_name = 'block5_conv1'
    layer_idx = [idx for idx, layer in enumerate(conv_base.layers)
                 if layer.name == layer_name][0]

    # Here we are asking it to show attention such that prob of `pred_class` is maximized.
    heatmap = visualize_cam(model, layer_idx, [pred_class], seed_img,penultimate_layer_idx=17)

    # Overlay heatmap onto the image with alpha blend.
    heatmaps.append(overlay(seed_img, heatmap))

plt.axis('off')
plt.imshow(utils.stitch_images(heatmaps))
plt.title('Saliency map')
plt.show()

from vis.utils import utils
from keras import activations

# Utility to search for layer index by name.
# Alternatively we can specify this as -1 since it corresponds to the last layer.
layer_idx = 5  # utils.find_layer_idx(model, 'fc_actv_8')

# Swap softmax with linear
model.layers[layer_idx].activation = activations.linear
model = utils.apply_modifications(model)

from vis.visualization import visualize_activation

from matplotlib import pyplot as plt

plt.rcParams['figure.figsize'] = (18, 6)

# 20 is the imagenet category for 'ouzel'
img = visualize_activation(model, layer_idx, filter_indices=3)
plt.imshow(img)

idx = 59
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
from vis.visualization import visualize_activation
selected_indices = []
for layer_name in ['vgg19']:
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

# plt.rcParams['figure.figsize'] = (18, 6)

# 20 is the imagenet category for 'ouzel'
img = visualize_activation(model, layer_idx, filter_indices=3, max_iter=500, input_modifiers=[Jitter(16)], verbose=True)
plt.imshow(img)

import numpy as np

categories = np.random.permutation(num_classes)[:num_classes]
layer_idx = utils.find_layer_idx(model, 'vgg19')
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

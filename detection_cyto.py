import os
import tensorflow as tf
from tensorflow import keras
import shutil
from sklearn.model_selection import KFold, train_test_split
from keras.layers import Conv2D,BatchNormalization,Activation,MaxPooling2D,GlobalAveragePooling2D, Dropout
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.allow_growth=True
set_session(tf.Session(config=config))
from keras import callbacks


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

base_dir = '\\Users\\s-janewanthana\\Desktop\\model_edit\\tubulin'
NC_dir = "\\cp_A549_NC_tubulin_png"
WT_dir = "\\cp_A549_WT_tubulin_png"
os.listdir(base_dir+NC_dir)


file_NC = os.listdir(base_dir+NC_dir) 
file_WT = os.listdir(base_dir+WT_dir)


def writeFile(train_index, val_index, test_index, file_NC, file_WT, cross_index):
    for i in train_index:
        source_path = base_dir+ NC_dir+"\\" +file_NC[i]
        destination_path = base_dir+"\\Cross"+cross_index + "\\train\\1\\"
        if not os.path.exists(destination_path):
            os.makedirs(destination_path)
        destination_path = base_dir+"\\Cross"+cross_index + "\\train\\1\\"+ file_NC[i]
        shutil.copyfile(source_path, destination_path)
        
        source_path = base_dir+WT_dir+"\\"+file_WT[i]
        destination_path = base_dir+"\\Cross"+cross_index + "\\train\\0\\"
        if not os.path.exists(destination_path):
            os.makedirs(destination_path)
        destination_path = base_dir+"\\Cross"+cross_index + "\\train\\0\\"+ file_WT[i]
        shutil.copyfile(source_path, destination_path)
    
    for i in val_index:
        source_path = base_dir+ NC_dir+"\\" +file_NC[i]
        destination_path = base_dir+"\\Cross"+cross_index + "\\val\\1\\"
        if not os.path.exists(destination_path):
            os.makedirs(destination_path)
        destination_path = base_dir+"\\Cross"+cross_index + "\\val\\1\\"+ file_NC[i]
        shutil.copyfile(source_path, destination_path)
        
        source_path = base_dir+WT_dir+"\\"+file_WT[i]
        destination_path = base_dir+"\\Cross"+cross_index + "\\val\\0\\"
        if not os.path.exists(destination_path):
            os.makedirs(destination_path)
        destination_path = base_dir+"\\Cross"+cross_index + "\\val\\0\\"+ file_WT[i]
        shutil.copyfile(source_path, destination_path)
    
    for i in test_index:
        source_path = base_dir+NC_dir+"\\"+file_NC[i]
        destination_path = base_dir+"\\Cross"+cross_index + "\\test\\1\\"
        if not os.path.exists(destination_path):
            os.makedirs(destination_path)
        destination_path = base_dir+"\\Cross"+cross_index + "\\test\\1\\"+ file_NC[i]
        shutil.copyfile(source_path, destination_path)
        
        source_path = base_dir+WT_dir+"\\"+file_WT[i]
        destination_path =  base_dir+"\\Cross"+cross_index + "\\test\\0\\"
        if not os.path.exists(destination_path):
            os.makedirs(destination_path)
        destination_path =  base_dir+"\\Cross"+cross_index + "\\test\\0\\"+file_WT[i]
        shutil.copyfile(source_path, destination_path)

kf = KFold(n_splits=10)
k_count = 1
for train_index, test_index in kf.split(file_NC):
    train_index, val_index = train_test_split(train_index, test_size=0.2, random_state=42)
    writeFile(train_index, val_index, test_index, file_NC, file_WT, str(k_count))
    k_count+=1

from keras.preprocessing.image import ImageDataGenerator
train_dir = os.path.join(base_dir, 'Cross1\\train')
val_dir = os.path.join(base_dir, 'Cross1\\val')
# train_dir = os.path.join(base_dir, "png_A549_nocodazole_tubulin")


# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

Add our data-augmentation parameters to ImageDataGenerator
train_datagen = ImageDataGenerator(
    rotation_range=30,
    horizontal_flip=True,
    vertical_flip= True,
    fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_dir,
        # All images will be resized to 150x150
        target_size=(150, 150),
        color_mode="grayscale",
        batch_size=4,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        val_dir,
        target_size=(150, 150),
        color_mode="grayscale",
        batch_size=4,
        class_mode='binary')



from keras import layers
from keras import models

from keras import models

def vgg16p():
    
    model = models.Sequential()
    #block1
    model.add(layers.Conv2D(64, (3, 3),padding='same',activation='relu',
                        input_shape=(150, 150, 1), name='block1_conv1'))
    model.add(layers.Conv2D(64, (3, 3), padding='same',activation='relu', name='block1_conv2'))
    model.add(BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2), name='block1_pool'))
    #block2
    model.add(layers.Conv2D(128, (3, 3),padding='same', activation='relu', name='block2_conv1'))
    model.add(layers.Conv2D(128, (3, 3), padding='same',activation='relu', name='block2_conv2'))
    model.add(BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2), name='block2_pool'))
    #block3
    model.add(layers.Conv2D(256, (3, 3), padding='same',activation='relu', name='block3_conv1'))
    model.add(layers.Conv2D(256, (3, 3),padding='same', activation='relu', name='block3_conv2'))
    model.add(layers.Conv2D(256, (3, 3),padding='same', activation='relu', name='block3_conv3'))
    model.add(BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2), name='block3_pool'))
    #block4
    model.add(layers.Conv2D(512, (3, 3),padding='same', activation='relu', name='block4_conv1'))
    model.add(layers.Conv2D(512, (3, 3), padding='same',activation='relu', name='block4_conv2'))
    model.add(layers.Conv2D(512, (3, 3),padding='same', activation='relu', name='block4_conv3'))
    model.add(BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2), name='block4_pool'))
    #block5
    model.add(layers.Conv2D(512, (3, 3),padding='same', activation='relu', name='block5_conv1'))
    model.add(layers.Conv2D(512, (3, 3), padding='same',activation='relu', name='block5_conv2'))
    model.add(layers.Conv2D(512, (3, 3),padding='same', activation='relu', name='block5_conv3'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(layers.MaxPooling2D((2, 2), name='block5_pool'))
    #flat
    model.add(layers.Flatten())
    model.add(Dropout(0.5))
    model.add(layers.Dense(512, activation='relu', name='dense512'))
    model.add(layers.Dense(256, activation='relu', name='dense256'))
    model.add(layers.Dense(128, activation='relu', name='dense128'))
    model.add(layers.Dense(64, activation='relu', name='dense64'))
              
    model.add(layers.Dense(2, activation='softmax, name='dense2'))
    return model


model = vgg16p()
model.summary()
from keras import optimizers

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.Adam(lr=1e-6),
              metrics=['acc'])

sb = callbacks.ModelCheckpoint("C:\\Users\\s-janewanthana\\Desktop\\image_data_manuscript\\keras_aug\\tubulin\\TFC02_weight_tubulin_kcross1.h5", save_best_only=True, monitor="val_loss", mode="min", verbose=1)

history = model.fit_generator(
      train_generator,
      steps_per_epoch=50,
      epochs=300,
      validation_data=validation_generator,
      validation_steps=50,
      callbacks=[sb])
import pickle
pickle.dump(history.history, open('TFC02_pick_tubulin_kcross1', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)



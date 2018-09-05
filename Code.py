from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# BUILDING CNN
# Initializing CNN
classifier = Sequential()

# Step 1: Adding Convolution Layer
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

# Step 2: Pooling 
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# ADDING EXTRA LAYER
classifier.add(Convolution2D(64, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3: Flattening
classifier.add(Flatten())

# Step 4: Full Connection
classifier.add(Dense(activation = 'relu', output_dim = 128))
classifier.add(Dense(activation = 'sigmoid', output_dim = 1))

# Compiling The CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# FITTING THE CNN TO THE IMAGES
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale = 1./255,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

train_set = train_datagen.flow_from_directory('dataset/train_set',
                                              target_size = (64, 64),
                                              batch_size = 32,
                                              class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(train_set,
                         samples_per_epoch = 8000,
                         epochs = 25,
                         validation_data = test_set,
                         nb_val_samples = 2000)

# Make Single Prediction

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single/cat.4003.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
res = classifier.predict(test_image)
train_set.class_indices

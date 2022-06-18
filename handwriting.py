import tensorflow as tf
import matplotlib.pyplot as plot
import numpy as np
import cv2
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

IMAGE_SIZE=28

if not os.path.exists('./model/saved_model.pb'):
    print("Couldn't find trained Model.")
    print("Starting training process ...")

    handDigitalizationDataset = tf.keras.datasets.mnist
    (image_train, label_train), (image_test, label_test) = handDigitalizationDataset.load_data()
    image_train.shape

    plot.imshow(image_train[0])
    plot.show()
    plot.imshow(image_train[3], cmap= plot.cm.binary)

    image_train = tf.keras.utils.normalize(image_train, axis=1)
    image_test = tf.keras.utils.normalize(image_test, axis=1)

    image_trainer = np.array(image_train).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
    image_tester = np.array(image_test).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)

    # Creating Deep Learning Neural Network
    dnnModel = Sequential()

    # Adding the first Convolutional layer
    dnnModel.add(Conv2D(64, (3,3), input_shape = image_trainer.shape[1:]))
    dnnModel.add(Activation("relu"))
    dnnModel.add(MaxPooling2D(pool_size=(2,2)))

    # Adding the second Convolutional layer
    dnnModel.add(Conv2D(64, (3,3), input_shape = image_trainer.shape[1:]))
    dnnModel.add(Activation("relu"))
    dnnModel.add(MaxPooling2D(pool_size=(2,2)))

    # Adding the third Convolutional layer
    dnnModel.add(Conv2D(64, (3,3), input_shape = image_trainer.shape[1:]))
    dnnModel.add(Activation("relu"))
    dnnModel.add(MaxPooling2D(pool_size=(2,2)))

    # Adding First Fully Connected Layer
    dnnModel.add(Flatten())
    dnnModel.add(Dense(64))
    dnnModel.add(Activation("relu"))

    # Adding Second Fully Connected Layer
    dnnModel.add(Dense(32))
    dnnModel.add(Activation("relu"))

    # Adding Last Fully Connected Layer
    dnnModel.add(Dense(10))
    dnnModel.add(Activation("softmax"))

    # Compiling and Training the Model
    dnnModel.compile(loss= "sparse_categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
    dnnModel.fit(image_trainer, label_train, epochs=5, validation_split= 0.3)

    # Prediction
    prediction = dnnModel.predict([image_tester])
    print (np.argmax(prediction[22]))
    plot.imshow(image_test[22])

    # save model
    dnnModel.save('./model/')
else:
    # load existing model
    print("Existing trained model found.")
    print("Loading model...")
    trainedModel = tf.keras.models.load_model('./model/')

# Prototype

print ("Running Prototype")
yourImage = cv2.imread('./img/handwritten-number.jpg')
yourImage = cv2.cvtColor(yourImage, cv2.COLOR_BGR2GRAY)
yourImage = cv2.resize(yourImage, (IMAGE_SIZE,IMAGE_SIZE), interpolation=cv2.INTER_AREA)
yourImage = tf.keras.utils.normalize(yourImage, axis= 1)
yourImage = np.array(yourImage).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
prediction = dnnModel.predict(yourImage)
print(np.argmax(prediction))
import tensorflow as tf
import matplotlib.pyplot as plot
import numpy as np
import cv2
import os
import gtts
import time

from PIL import Image

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from playsound import playsound

# Constants and initial configurations
IMAGE_SIZE = 28
BOX_SIZE = 160
MODEL_NOT_FOUND = './voice/model-not-found.mp3'
START_TRAINING_PROCESS = './voice/start-training-process.mp3'
MODEL_FOUND = './voice/model-found.mp3'
LOADING_EXISTING_MODEL = './voice/loading-existing-model.mp3'
RUNNING_APPLICATION = './voice/running-prototype.mp3'
HANDWRITING_DETECTED = './voice/handwriting-detected.mp3'
DETECTED_DIGIT = './voice/detected-digit.mp3'
LANGUAGE = 'en'
VIEW_PORT_WIDTH = 640
VIEW_PORT_HEIGHT = 480


# Create Speech files
if not os.path.exists(MODEL_NOT_FOUND):
    tts = gtts.gTTS("Couldn't find trained Model.")
    tts.save(MODEL_NOT_FOUND)

if not os.path.exists(START_TRAINING_PROCESS):
    tts = gtts.gTTS("Starting training process!")
    tts.save(START_TRAINING_PROCESS)

if not os.path.exists(MODEL_FOUND):
    tts = gtts.gTTS("Existing trained model found!")
    tts.save(MODEL_FOUND)

if not os.path.exists(LOADING_EXISTING_MODEL):
    tts = gtts.gTTS("Loading model.")
    tts.save(LOADING_EXISTING_MODEL)

if not os.path.exists(RUNNING_APPLICATION):
    tts = gtts.gTTS("Running Prototype")
    tts.save(RUNNING_APPLICATION)

if not os.path.exists(HANDWRITING_DETECTED):
    tts = gtts.gTTS("I can see")
    tts.save(HANDWRITING_DETECTED)

MODEL_NOT_FOUND='./voice/model-not-found.mp3'

if not os.path.exists('./model/saved_model.pb'):
    print("Couldn't find trained Model.")
    playsound(MODEL_NOT_FOUND)

    print("Starting training process ...")
    playsound(START_TRAINING_PROCESS)

    handDigitalizationDataset = tf.keras.datasets.mnist
    (image_train, label_train), (image_test, label_test) = handDigitalizationDataset.load_data()
    image_train.shape

    plot.imshow(image_train[0])
    plot.imshow(image_train[0], cmap= plot.cm.binary)

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
    # playsound(MODEL_FOUND)

    print("Loading model...")
    # playsound(LOADING_EXISTING_MODEL)

    dnnModel = tf.keras.models.load_model('./model/')

# Prototype

print ("Running Prototype")
yourImage = cv2.imread('./img/test-input-number.jpg')
yourImage = cv2.cvtColor(yourImage, cv2.COLOR_BGR2GRAY)
yourImage = cv2.resize(yourImage, (IMAGE_SIZE,IMAGE_SIZE), interpolation=cv2.INTER_AREA)
yourImage = tf.keras.utils.normalize(yourImage, axis= 1)
yourImage = np.array(yourImage).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
prediction = dnnModel.predict(yourImage)
digit = np.argmax(prediction)
print(digit)

if os.path.exists(DETECTED_DIGIT):
    os.remove(DETECTED_DIGIT)

tts = gtts.gTTS(str(digit))
tts.save(DETECTED_DIGIT)
#playsound(HANDWRITING_DETECTED)
#playsound(DETECTED_DIGIT)

# threshold slider handler 
threshold = 150

# Read from camera
cap = cv2.VideoCapture(0)

# Set view port size e.g. 640 x 480
cap.set(3, VIEW_PORT_WIDTH)
cap.set(4, VIEW_PORT_HEIGHT)

# Set window name
originalWindow = cv2.namedWindow('Live Camera Feed')
processedWindow = cv2.namedWindow('Processed')
background = np.zeros((VIEW_PORT_HEIGHT, VIEW_PORT_WIDTH), np.uint8)
frameCount = 0

# Predict function
def predict(model, img):
    imgs = np.array([img])
    res = model.predict(imgs)
    index = np.argmax(res)
    #print(index)
    return str(index)

# Keep looping to get a live stream
while True:
    ret, frame = cap.read()
    # frame counter for showing text 
    frameCount += 1
    # black outer frame
    frame[0:VIEW_PORT_HEIGHT, 0:80] = 0
    frame[0:VIEW_PORT_HEIGHT, 560:VIEW_PORT_WIDTH] = 0
        
    
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    plot.imshow(grayFrame)
        
    # apply threshold
    _, thr = cv2.threshold(grayFrame, threshold, 255, cv2.THRESH_BINARY_INV)
    # get central image 
    resizedProcessedFrame = thr[(VIEW_PORT_HEIGHT//2)-75:(VIEW_PORT_HEIGHT//2)+75, (VIEW_PORT_WIDTH//2)-75:(VIEW_PORT_WIDTH//2)+75]
    background[(VIEW_PORT_HEIGHT//2)-75:(VIEW_PORT_HEIGHT//2)+75, (VIEW_PORT_WIDTH//2)-75:(VIEW_PORT_WIDTH//2)+75] = resizedProcessedFrame
    # resize for inference 
    iconImg = cv2.resize(resizedProcessedFrame, (28, 28))


    # Pass to model predictor 
    predictedNumber = predict(dnnModel,iconImg)
        
    # Clear background 
    if frameCount == 10:
        background[0:480, 0:80] = 0
        frameCount = 0
        cv2.putText(img=background, text=predictedNumber, org= (10, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(255, 235, 42), thickness=3)

    # Show text 
    cv2.putText(img=background, text="I can see", org= (10,30), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(255, 235, 42), thickness=1)
    cv2.putText(img=background, text="Press q to quit", org= (VIEW_PORT_HEIGHT, 455), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(255, 255, 255), thickness=1)
    cv2.rectangle(background, ((VIEW_PORT_WIDTH//2)-(BOX_SIZE//2), (VIEW_PORT_HEIGHT//2)-(BOX_SIZE//2)), ((VIEW_PORT_WIDTH//2)+(BOX_SIZE//2), (VIEW_PORT_HEIGHT//2)+(BOX_SIZE//2)), (255, 235, 42), thickness=2)
            
    # display frame 
    cv2.imshow(processedWindow, background)
    cv2.imshow('Live Camera Feed', frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
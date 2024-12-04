import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import os
import seaborn as sns
import sklearn.metrics as metrics
from tensorflow.keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input
from tensorflow.keras.optimizers import Adam as adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

NUM_CLASSES = 29
CHANNELS = 3
IMAGE_RESIZE = 224
RESNET50_POOLING_AVERAGE = 'avg'
DENSE_LAYER_ACTIVATION = 'softmax'
OBJECTIVE_FUNCTION = 'categorical_crossentropy'
LOSS_METRICS = ['accuracy']
NUM_EPOCHS = 1
EARLY_STOP_PATIENCE = 3
STEPS_PER_EPOCH_TRAINING = 10
STEPS_PER_EPOCH_VALIDATION = 10
BATCH_SIZE_TRAINING = 100
BATCH_SIZE_VALIDATION = 100
BATCH_SIZE_TESTING = 1

model = Sequential()
model.add(ResNet50(input_shape=(IMAGE_RESIZE, IMAGE_RESIZE, CHANNELS)))
model.add(Dense(NUM_CLASSES, activation = DENSE_LAYER_ACTIVATION))


model.summary()
opt = adam(learning_rate=0.001, decay=1e-6)
model.compile(optimizer = opt, loss = OBJECTIVE_FUNCTION, metrics = LOSS_METRICS)

image_size = IMAGE_RESIZE
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
    directory=r"./train/",
    target_size=(image_size, image_size),
    color_mode="rgb",
    
    class_mode="categorical",
    shuffle=True,
    
    batch_size=BATCH_SIZE_VALIDATION
)

valid_generator = train_datagen.flow_from_directory(
    directory=r"./valid/",
    target_size=(image_size, image_size),
    color_mode="rgb",
    
    class_mode="categorical",
    shuffle=True,
    
    batch_size=BATCH_SIZE_VALIDATION
)

(BATCH_SIZE_TRAINING, len(train_generator), BATCH_SIZE_VALIDATION, len(valid_generator))


cb_early_stopper = EarlyStopping(monitor = 'val_loss', patience = EARLY_STOP_PATIENCE)
cb_checkpointer = ModelCheckpoint(filepath = '../working/best.hdf5', monitor = 'val_loss', save_best_only = True, mode = 'auto')
fit_history = model.fit(
        train_generator,
        steps_per_epoch=STEPS_PER_EPOCH_TRAINING,
        epochs = NUM_EPOCHS,
        validation_data=valid_generator,
        validation_steps=STEPS_PER_EPOCH_VALIDATION)

model.save_weights("model.weights.h5")
print(fit_history.history.keys())

plt.figure(1, figsize = (15,8)) 
    
plt.subplot(221)  
plt.plot(fit_history.history['accuracy'])  
plt.plot(fit_history.history['val_accuracy'])  
plt.title('model accuracy')  
plt.ylabel('accuracy')  
plt.xlabel('epoch')  
plt.legend(['train', 'valid']) 
    
plt.subplot(222)  
plt.plot(fit_history.history['loss'])  
plt.plot(fit_history.history['val_loss'])  
plt.title('model loss')  
plt.ylabel('loss')  
plt.xlabel('epoch')  
plt.legend(['train', 'valid']) 

plt.show()


test_generator = train_datagen.flow_from_directory(
    directory=r"/test",
    target_size=(image_size, image_size),
    color_mode="rgb",
    batch_size=BATCH_SIZE_TESTING,
    class_mode=None,
    shuffle=False,
    seed=123
)
len(test_generator)
test_generator.reset()

pred = model.predict(test_generator, steps = len(test_generator), verbose = 1)

predicted_class_indices = np.argmax(pred, axis = 1)


true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

report = metrics.classification_report(true_classes, predicted_class_indices, target_names=class_labels)
print(report)

cm = metrics.confusion_matrix(true_classes, predicted_class_indices)

plt.figure(figsize = (20, 20))
sns.heatmap(cm, annot = True, cbar = False, fmt = 'd')
plt.show()

class_labels = (train_generator.class_indices)
class_labels = dict((v,k) for k,v in class_labels.items())

print(class_labels)



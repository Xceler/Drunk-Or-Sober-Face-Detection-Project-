import numpy as np
import cv2
import tensorflow as tf 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

data = 'Project'

datagen = ImageDataGenerator(
    rescale=1./255.0,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.5,
    shear_range=0.2,
    fill_mode='nearest',
    rotation_range=15,
    validation_split=0.2  # Splitting dataset into training and validation
)

train_gen = datagen.flow_from_directory(
    data,
    target_size=(224, 224), 
    batch_size=32,
    class_mode='binary',
    subset='training'  # Set as training data
)

val_gen = datagen.flow_from_directory(
    data,
    target_size=(224, 224), 
    batch_size=32,
    class_mode='binary',
    subset='validation'  # Set as validation data
)

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
prediction = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=prediction)

for layer in base_model.layers:
    layer.trainable = False 

model.compile(loss='binary_crossentropy',
              optimizer=Adam(learning_rate=0.0001),
              metrics=['accuracy'])

# Callbacks for early stopping and model checkpointing
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)


model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=20,  # Increase the number of epochs
    callbacks=[early_stop]
)

def preprocess_img(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image 

def predict(image_path):
    image = preprocess_img(image_path)
    prediction = model.predict(image)
    if prediction[0] <= 0.5:
        return 'Drunk'
    else:
        return 'Sober'
image_1 = 'Project/Sober/Soberface8A2_jpg.rf.df7b9de0093a3f95f2cadf59db945709.jpg'
result_1 = predict(image_1)
print('Sober Image:', result_1)


image_2 = 'Project/Sober/Soberface39A3_jpg.rf.bb2f99c0811e8a9d71f662c326f4ce62.jpg'
result_2 = predict(image_2)
print('Sober Image:', result_2)


image_3 = 'Project/Drunk/Drunkface8B3_jpg.rf.28452dccc58768a229ff48c5f33ab575.jpg'
result_3 = predict(image_3)
print('Drunk Image:', result_3)


image_4 = 'Project/Drunk/Drunkface75B2_jpg.rf.bb099c009ed461d701f218a612a8fc99.jpg'
result_4 = predict(image_4)
print('Drunk Image:', result_4)



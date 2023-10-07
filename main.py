from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout , MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from keras.callbacks import ReduceLROnPlateau
import os


def os_walk():
    for dirname, _, filenames in os.walk('./dataset/'):
        for filename in filenames:
            print(os.path.join(dirname, filename))



train_datagen = ImageDataGenerator(rescale=1./255)  # Add more augmentations as needed

train_generator = train_datagen.flow_from_directory(
    './dataset/asl_alphabet_train/asl_alphabet_train/',  # Replace with your folder path
    target_size=(224, 224),  # Resize images to a standard size
    batch_size=32,
    class_mode='categorical'
)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(29, activation='softmax')  # 29 classes based on your dataset
])


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=10  # Adjust as needed
)


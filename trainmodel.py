import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras.regularizers import l2
import tensorflow as tf

class AccuracyThresholdCallback(tf.keras.callbacks.Callback):
    def __init__(self, threshold=0.98):
        super(AccuracyThresholdCallback, self).__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        if logs.get('val_accuracy') >= self.threshold:
            print(f'\nReached {self.threshold*100}% accuracy - stopping training!')
            self.model.stop_training = True

def load_data(data_dir):
    X = []
    y = []
    classes = []
    for i, sign in enumerate(sorted(os.listdir(data_dir))):
        sign_dir = os.path.join(data_dir, sign)
        if os.path.isdir(sign_dir):
            classes.append(sign)
            for filename in os.listdir(sign_dir):
                if filename.endswith('.jpg'):
                    img_path = os.path.join(sign_dir, filename)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                    img = clahe.apply(img)
                    img = cv2.GaussianBlur(img, (3,3), 0)
                    img = cv2.resize(img, (96, 96))
                    X.append(img)
                    y.append(i)

    return np.array(X), np.array(y), classes

if __name__ == "__main__":
    data_dir = 'data'

    print("Loading data...")
    X, y, classes = load_data(data_dir)

    y_map = {old: new for new, old in enumerate(sorted(np.unique(y)))}
    y = np.array([y_map[val] for val in y])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    X_train = (X_train.astype('float32') - 127.5) / 127.5
    X_test = (X_test.astype('float32') - 127.5) / 127.5

    X_train = X_train.reshape(X_train.shape[0], 96, 96, 1)
    X_test = X_test.reshape(X_test.shape[0], 96, 96, 1)

    num_classes = len(classes)
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        shear_range=0.2,
        fill_mode='nearest',
        horizontal_flip=True,
        brightness_range=[0.8, 1.2]
    )

    model = Sequential([
        Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(96, 96, 1), kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),

        Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),

        Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),

        Conv2D(512, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Conv2D(512, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),

        Flatten(),
        Dense(512, activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    checkpoint = ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True, mode='max')
    accuracy_threshold = AccuracyThresholdCallback(threshold=0.98)

    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=32),
        epochs=150,
        validation_data=(X_test, y_test),
        callbacks=[reduce_lr, early_stopping, checkpoint, accuracy_threshold],
        steps_per_epoch=len(X_train) // 32
    )

    model.load_weights('best_model.keras')
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_accuracy:.4f}")

    model.save('asl_recognition_model3.keras')

    with open('class_names.txt', 'w') as f:
        for class_name in classes:
            f.write(f"{class_name}\n")

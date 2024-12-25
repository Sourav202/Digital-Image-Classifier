import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score

#Function to Read csv files and load data
def load_data(train_file, val_file):
    train_data = pd.read_csv(train_file)
    val_data = pd.read_csv(val_file)
    
    #Normalize pixel values to [0, 1]
    X_train = train_data.iloc[:, 1:].values / 255.0
    y_train = train_data.iloc[:, 0].values
    X_val = val_data.iloc[:, 1:].values / 255.0
    y_val = val_data.iloc[:, 0].values
    
    #Reshape matrices into the correct dimension
    X_train = X_train.reshape(-1, 84, 28, 1)
    X_val = X_val.reshape(-1, 84, 28, 1)
    return X_train, y_train, X_val, y_val

#Function to augment data so that the model can train on slightly different images
def augment_data(X):
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=3,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=[0.95, 1.05],
        shear_range=2,
    )
    datagen.fit(X)
    return datagen

#Learn the model
def learn(X, y):
    """
    Train a neural network using TensorFlow with data augmentation.
    """
    #Augment the data
    datagen = augment_data(X)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(84, 28, 1)),
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), padding="same", activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), padding="same", activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(128, kernel_size=(3, 3), padding="same", activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(256, kernel_size=(3, 3), padding="same", activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    #Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model with augmented data
    history = model.fit(datagen.flow(X, y, batch_size=32), epochs=10)
    
    return model, history

#Use the trained convolutional neural network to classify test data.
def classify(Xtest, model):
    predictions = model.predict(Xtest)
    yhat = np.argmax(predictions, axis=1)  # Convert probabilities to class labels
    return yhat

if __name__ == "__main__":
    train_file = 'A4data/A4train.csv'
    val_file = 'A4data/A4val.csv'
    
    X_train, y_train, X_val, y_val = load_data(train_file, val_file)
    model, history = learn(X_train, y_train)
    yhat_val = classify(X_val, model)
    accuracy = accuracy_score(y_val, yhat_val)
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")
    
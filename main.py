import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# load and split dataset into train and test splits
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0


# verify dataset
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()

# build model
conv_inputs = keras.Input(shape=(32, 32, 3))
conv_layer1 = layers.Conv2D(32, kernel_size=3, activation='relu')(conv_inputs)
max1 = layers.MaxPool2D(pool_size=(2, 2))(conv_layer1)
conv_layer2 = layers.Conv2D(64, kernel_size=3, activation='relu')(max1)
max2 = layers.MaxPool2D(pool_size=(2, 2))(conv_layer2)
conv_layer3 = layers.Conv2D(64, kernel_size=3, activation='relu')(max2)
flat = layers.Flatten()(conv_layer3)
den = layers.Dense(64, activation='relu')(flat)
conv_outputs = layers.Dense(10)(den)

conv_model = keras.Model(inputs=conv_inputs, outputs=conv_outputs)

# show model architecture
conv_model.summary()

# compile model using binary cross entropy loss function
conv_model.compile(optimizer='adam',
                   loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                   metrics=['mean_squared_error', 'accuracy'])

# train model
print('Fit model on training data')

history = conv_model.fit(train_images, train_labels, epochs=10,
                         validation_data=(test_images, test_labels))

# evaluate model
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

test_loss, test_acc = conv_model.evaluate(x=test_images,  y=test_labels)
print(test_acc)

# Working with larget example (multiclass classification)
## (Neural Network to classify images of different items of clothing)

When you have more than 2 classes, its known as **multiclass classification**.

And when you have only two classes, its knows as **binary classification**

To practice multiclass classification, we are going to build neural network to classify images of different items of clothing.
"""

import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist

# The data has already been sorted into training and test sets for us
(train_data, train_labels), (test_data, test_labels) = fashion_mnist.load_data() # load the data into train and test data

# Show the first training example
print(f"Training examples are:\n{train_data[0]}\n")
print(f"Training label are:\n{train_labels[0]}\n")

# Check the shape of the single example
train_data[0].shape, train_labels[0].shape

# Plot a single sample
import matplotlib.pyplot as plt
plt.imshow(train_data[7])

# Check the training sample
train_labels[7]

# Create a small list so we can index onto our training labels so they are in human-readable form
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
len(class_names)

# Plot an example image and its label
index_of_choice = 2000
plt.imshow(train_data[index_of_choice], cmap=plt.cm.binary)
plt.title(class_names[train_labels[index_of_choice]])

# Plot multiple random images of FASHION MNIST
import random
plt.figure(figsize=(7,7))

for i in range(4):
  ax = plt.subplot(2, 2, i+1)
  rand_index = random.choice(range(len(train_data)))
  plt.imshow(train_data[rand_index], cmap=plt.cm.binary) # here cmap is adding greyscale to image
  plt.title(class_names[train_labels[rand_index]])
  plt.axis(False)

"""# Building a multi-class classification model

For our multiclass classification model, we can use a similar architecture to our binary classifiers, however, we are going to have to tweak a few things:

* Input shape = 28 x 28 (the shape of one image)
* Output shape = 10 (one per class of clothing)
* Loss = tf.keras.losses.CategoricalCrossentropy()
  * If you labels are one hot encoded use CategoricalCrossentropy()
  * If you labels are in integer form use SparseCategoricalCrossentropy()
* Output Layer activation = softmax not sigmoid


"""

# Set random seed
tf.random.set_seed(42)

#Create the model
model_11 = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)), # Flatten your input shape data to avoid ValueError: Shapes (32,) and (32, 28, 10) are incompatible error
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(10, activation=tf.keras.activations.softmax)
])

# Also to run the data we need to convert our labels into one-hot encoded (in the form of 0 or 1) to use Categorical else use Sparse (but now use Categorical)
# Compile the model
model_11.compile(loss=tf.keras.losses.CategoricalCrossentropy(), # we can also fix this shape error by fixing Catogorical -> SparseCategorical
                 optimizer=tf.keras.optimizers.Adam(),
                 metrics=["accuracy"])

# Fit the model
non_norm_history = model_11.fit(train_data, tf.one_hot(train_labels, depth=10),
                                epochs=10,
                                validation_data=(test_data, tf.one_hot(test_labels, depth=10))) # validation data ensures the validity of the data with validated data, in this case we use test data

# To determine how muhc the accuracy hsould be for example if we have 10 labels then accuracy can be 100/10 = 10.0 so in our case above is 33 which is 3 times good
# But lets improve our model more
model_11.summary()

# Check the min and max value of training data
train_data.min(), train_data.max()

"""Neural networks prefer data to be scaled (or normalized), this means they like to have the numbers in the tensors they try to find patterns between 0 and 1"""

# We can get training and testing data between 0 & 1 by dividing by maximum
train_data_norm = train_data / 255.0
test_data_norm = test_data / 255.0

# Check the min and max values of the scaled training data
train_data_norm.min(), train_data_norm.max()

# Now our data is normalized, let build our model to find patterns in it

# Set seed
tf.random.set_seed(42)

# Create a model
model_12 = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

# Compile the model
model_12.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=["accuracy"]
)

# Fit the model
norm_history = model_12.fit(train_data_norm, train_labels,  epochs=10,
                            validation_data=(test_data_norm, test_labels))

"""**Note** Neural Networks tend to prefer data in numerical form as well as scaled/normalized (numbers  between 0 & 1)"""

# Let now compare the loss curves between normalized data and non-normalized dat model
import pandas as pd
# Plot the non normalized data loss curves
pd.DataFrame(non_norm_history.history).plot(title="Non-normalized data")
# Plot normalized loss data curve
pd.DataFrame(norm_history.history).plot(title="Normalized data")

"""**Note:** same model with even *slightly* different data can produce *dramatically* different results. So when you are comparing models its important tha tyou are comparing them on the same criteria.(e.g same architecture but different data or same data but different architecture)

# Find the Ideal learning rate
"""

# Set random seed
tf.random.set_seed(42)

# Create the model
model_13 = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

#compile the model
model_13.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                 optimizer=tf.keras.optimizers.Adam(),
                 metrics=["accuracy"])

# create the learning rate callback
lr_schedular = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-3 * 10**(epoch/20))

# Fit the model
find_lr_history = model_13.fit(train_data_norm,
                               train_labels,
                               epochs=40,
                               validation_data=(test_data_norm, test_labels),
                               callbacks=[lr_schedular])

# For finding ideal learning rate go to the lowest point and then go backwards slightly higher
# Plot the learning decay rate curve
import pandas as pd
import matplotlib.pyplot as plt


lrs = 1e-3 * (10**(tf.range(40)/20))
plt.semilogx(lrs, find_lr_history.history["loss"])
plt.xlabel("Learning rate")
plt.ylabel("Loss")
plt.title("Finding the ideal learning rate")

10**-3

# Lets refit he model with ideal learning rate
# Set seed

tf.random.set_seed(42)

# Create model
model_14 = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

# Compile the model
model_14.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                 optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                 metrics=["accuracy"])

# fit the model
history_14 = model_14.fit(train_data_norm, train_labels, epochs=20,
                          validation_data=(test_data_norm, test_labels))

"""## Evaluating our multiclass classification model

To evaluate out multi-class classification model we could:
* Evaluate its performance using other classification metrics (such as confusion matrix)
* Asses some of its predictions through visualization
* Improve its results (by training if for longer or changing the architecture)
* Save it and export it to use in our later applications

Lets go through the top two:
"""

# Create confusion matrix

# Note: the confusion matrix code we are about to write is a remix of scikit-learn's plot_confusion_matrix

import itertools
from sklearn.metrics import confusion_matrix



def make_confusion_matrix(y_true, y_pred, classes=None, figsize=(10,10), text_size=15):

  # Create the confusion matrix
  cm = confusion_matrix(y_true, y_pred)
  cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] # normalise our confusion matrix
  n_classes = cm.shape[0]

  # Let's prettify it
  fig, ax = plt.subplots(figsize=figsize)

  # Create a matrix plot
  cax = ax.matshow(cm, cmap=plt.cm.Blues)
  fig.colorbar(cax)

  #set labels to be classes

  if classes:
    labels = classes
  else:
    labels = np.arange(cm.shape[0])

  # Label the axes
  ax.set(title="Confusion Matrix",
        xlabel="Predicted Label",
        ylabel="True Label",
        xticks=np.arange(n_classes),
        yticks=np.arange(n_classes),
        xticklabels=labels,
        yticklabels=labels)

  # Set the x-axis label to the bottom
  ax.xaxis.set_label_position('bottom')
  ax.xaxis.tick_bottom()

  # Adjust the label size
  ax.yaxis.label.set_size(text_size)
  ax.xaxis.label.set_size(text_size)
  ax.title.set_size(text_size)

  # Set the threshold for different colors
  threshold = (cm.max() + cm.min()) / 2

  # Plot the text on each cell
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j] * 100:.1f}%)",
            horizontalalignment="center",
            color="white" if cm[i, j] > threshold else "black",
            size=text_size)

class_names

# make some predictions with our model

y_probs = model_14.predict(test_data_norm) #probs is short for prediction probablities

# View the first 5 predictions
y_probs[:5]

"""**Note** Remember to make predictions on the same kind of data your model was trained on. In otu case it was **test_data_norm**"""

y_probs[0], tf.argmax(y_probs[0]), class_names[tf.argmax(y_probs[0])]

# Convert all the prediction probablites into integers
y_preds = y_probs.argmax(axis=1)

# View the first 10 predictions labels
y_preds[:10]

test_labels

# Create confusion matrix completely
from sklearn.metrics import confusion_matrix
confusion_matrix(y_true=test_labels, y_pred=y_preds)

# Now lets make a prettier confusion matrix
make_confusion_matrix(y_true=test_labels, y_pred=y_preds,
                      classes=class_names,
                      figsize=(15, 15),
                      text_size=10
                      )

"""**Note** Often when working with images and other forms of visual data, its a good idea to visualize as much as possible to develop a furthur understanding of the data and the inputs and the outputs of the model.

How about we create a fun little function for:
* Plot a random image
* Make a prediction on said image
* We should label the plot with the truth label & the predicted label
"""

import random

def plot_random_image(model, images, true_labels, classes):
  """
  Picks a random image, plots it and labels it with a prediction and a truth label
  """
  # set up a random integer
  i = random.randint(0, len(images))

  # predictions and target
  target_image = images[i]
  pred_probs = model.predict(target_image.reshape(1, 28, 28))
  pred_label = classes[pred_probs.argmax()]
  true_label = classes[true_labels[i]]

  # Plot the image
  plt.imshow(target_image, cmap=plt.cm.binary)

  # Change the color of titles depending on if the prediction is right or wrong
  if pred_label == true_label:
    color = "green"
  else:
    color="red"

  # Add xlabel information (prediction/true label)
  plt.xlabel("Pred: {} {:2.0f}% (True: {})".format(pred_label, 100*tf.reduce_max(pred_probs), true_label),
             color=color) # set the color to green or red based on the prediction if its right or wrong

#Check out the random image as well as its prediction
plot_random_image(model=model_14,
                  images=test_data_norm, #always makes predictions on the same kind on which yor data was trained on
                  true_labels=test_labels,
                  classes=class_names)

"""# What patterns is our model learning?"""

# Find the layers of our most recent model
model_14.layers

# Extract a particular layer
model_14.layers[1]

# get the patterns of the layer in our network
weights, biases = model_14.layers[1].get_weights()

# Shapes
weights, weights.shape

"""# Now lets check out the bias vector

"""

biases, biases.shape

"""Every neuron has a bias vector. Each of these is paired with a weights matrix.

The bias vector also gets initialized as zeros, at least in the case of tensorflow dense layer

The bias vector dictates how much the patterns within the corresponding weights matrix should influence the next layer
"""

model_14.summary()

# Lets check out another way of viewing our deep learning model
from tensorflow.keras.utils import plot_model

# see the inputs and outputs
plot_model(model_14, show_shapes=True)

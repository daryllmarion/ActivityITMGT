import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import Model
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

mnist = keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, 3, activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.d1 = tf.keras.layers.Dense(128, activation='relu')
        self.d2 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)

model = MyModel()

loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='test_accuracy')


@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)


@tf.function
def test_step(images, labels):
    predictions = model(images)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)

# Fine-tuning the model with your wardrobe dataset
EPOCHS = 5

for epoch in range(EPOCHS):
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

    for images, labels in train_ds:
        train_step(images, labels)

    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch+1,
                          train_loss.result(),
                          train_accuracy.result()*100,
                          test_loss.result(),
                          test_accuracy.result()*100))

# Load the pre-trained VGG16 model
base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add data augmentation to the training dataset
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
    tf.keras.layers.experimental.preprocessing.RandomZoom(0.2),
])

# Modify the output layer of the pre-trained model to match the number of clothing categories
num_categories = 5  # Adjust this based on the number of clothing categories in your wardrobe dataset
x = base_model.output
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
predictions = tf.keras.layers.Dense(num_categories, activation='softmax')(x)

model = tf.keras.Model(inputs=base_model.input, outputs=predictions)


# Fine-tuning the model with your wardrobe dataset
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# Define a sample wardrobe dataset with image paths
wardrobe = {
    'item_1': {'image_path': '/Users/teabetita/Downloads/ITMGT-K Final Project Filez/wardrobe/images_finalproject1.png', 'color': 'pink', 'fabric': 'cotton', 'design': 'plain', 'category': 'shirt', 'length': 'long'},
    'item_2': {'image_path': '/Users/teabetita/Downloads/ITMGT-K Final Project Filez/wardrobe/images_finalproject2.png', 'color': 'pink', 'fabric': 'cotton', 'design': 'plain', 'category': 'shirt', 'length': 'long'},
    'item_3': {'image_path': '/Users/teabetita/Downloads/ITMGT-K Final Project Filez/wardrobe/images_finalproject3.png', 'color': 'blue', 'fabric': 'cotton', 'design': 'plain', 'category': 'shirt', 'length': 'long'},
    'item_4': {'image_path': '/Users/teabetita/Downloads/ITMGT-K Final Project Filez/wardrobe/images_finalproject4.png', 'color': 'green', 'fabric': 'cotton', 'design': 'plain', 'category': 'shirt', 'length': 'long'},
    'item_5': {'image_path': '/Users/teabetita/Downloads/ITMGT-K Final Project Filez/wardrobe/images_finalproject5.png', 'color': 'white', 'fabric': 'linen', 'design': 'plain', 'category': 'button up', 'length': 'long'},
    'item_6': {'image_path': '/Users/teabetita/Downloads/ITMGT-K Final Project Filez/wardrobe/images_finalproject6.png', 'color': 'beige', 'fabric': 'linen', 'design': 'plain', 'category': 'button up', 'length': 'long'},
    'item_7': {'image_path': '/Users/teabetita/Downloads/ITMGT-K Final Project Filez/wardrobe/images_finalproject7.png', 'color': 'pink', 'fabric': 'linen', 'design': 'plain', 'category': 'button up', 'length': 'long'},
    'item_8': {'image_path': '/Users/teabetita/Downloads/ITMGT-K Final Project Filez/wardrobe/images_finalproject8.png', 'color': 'white', 'fabric': 'cotton', 'design': 'plain', 'category': 'pants', 'length': 'full'},
    'item_9': {'image_path': '/Users/teabetita/Downloads/ITMGT-K Final Project Filez/wardrobe/images_finalproject9.png', 'color': 'black', 'fabric': 'cotton', 'design': 'plain', 'category': 'pants', 'length': 'full'},
    'item_10': {'image_path': '/Users/teabetita/Downloads/ITMGT-K Final Project Filez/wardrobe/images_finalproject10.png', 'color': 'white', 'fabric': 'leather', 'design': 'plain', 'category': 'shoes', 'length': 'short'},
    'item_11': {'image_path': '/Users/teabetita/Downloads/ITMGT-K Final Project Filez/wardrobe/images_finalproject11.png', 'color': 'pink', 'fabric': 'linen', 'design': 'pleated', 'category': 'top', 'length': 'short'},
    'item_12': {'image_path': '/Users/teabetita/Downloads/ITMGT-K Final Project Filez/wardrobe/images_finalproject12.png', 'color': 'black', 'fabric': 'cotton', 'design': 'plain', 'category': 'shirt', 'length': 'short'},
    'item_13': {'image_path': '/Users/teabetita/Downloads/ITMGT-K Final Project Filez/wardrobe/images_finalproject13.png', 'color': 'white', 'fabric': 'cotton', 'design': 'plain', 'category': 'top', 'length': 'long'},
    'item_14': {'image_path': '/Users/teabetita/Downloads/ITMGT-K Final Project Filez/wardrobe/images_finalproject14.png', 'color': 'green', 'fabric': 'cotton', 'design': 'plain', 'category': 'top', 'length': 'short'},
    'item_15': {'image_path': '/Users/teabetita/Downloads/ITMGT-K Final Project Filez/wardrobe/images_finalproject15.png', 'color': 'multi', 'fabric': 'rayon', 'design': 'patterned', 'category': 'top', 'length': 'long'},
    'item_16': {'image_path': '/Users/teabetita/Downloads/ITMGT-K Final Project Filez/wardrobe/images_finalproject16.png', 'color': 'multi', 'fabric': 'linen', 'design': 'patterned', 'category': 'top', 'length': 'short'},
    'item_17': {'image_path': '/Users/teabetita/Downloads/ITMGT-K Final Project Filez/wardrobe/images_finalproject17.png', 'color': 'orange', 'fabric': 'cotton', 'design': 'plain', 'category': 'top', 'length': 'short'},
    'item_18': {'image_path': '/Users/teabetita/Downloads/ITMGT-K Final Project Filez/wardrobe/images_finalproject18.png', 'color': 'beige', 'fabric': 'linen', 'design': 'pleated', 'category': 'top', 'length': 'short'},
    'item_19': {'image_path': '/Users/teabetita/Downloads/ITMGT-K Final Project Filez/wardrobe/images_finalproject19.png', 'color': 'green', 'fabric': 'satin', 'design': 'plain', 'category': 'dress', 'length': 'midi'},
    'item_20': {'image_path': '/Users/teabetita/Downloads/ITMGT-K Final Project Filez/wardrobe/images_finalproject20.png', 'color': 'magenta', 'fabric': 'linen', 'design': 'plain', 'category': 'jumpsuit', 'length': 'full'},
    'item_21': {'image_path': '/Users/teabetita/Downloads/ITMGT-K Final Project Filez/wardrobe/images_finalproject21.png', 'color': 'blue', 'fabric': 'silk', 'design': 'plain', 'category': 'dress', 'length': 'midi'},
    'item_22': {'image_path': '/Users/teabetita/Downloads/ITMGT-K Final Project Filez/wardrobe/images_finalproject22.png', 'color': 'black', 'fabric': 'cotton', 'design': 'plain', 'category': 'jumpsuit', 'length': 'full'},
    'item_23': {'image_path': '/Users/teabetita/Downloads/ITMGT-K Final Project Filez/wardrobe/images_finalproject23.png', 'color': 'red', 'fabric': 'cotton', 'design': 'plain', 'category': 'dress', 'length': 'long'},
    'item_24': {'image_path': '/Users/teabetita/Downloads/ITMGT-K Final Project Filez/wardrobe/images_finalproject24.png', 'color': 'black', 'fabric': 'rayon', 'design': 'pleated', 'category': 'dress', 'length': 'short'},
    'item_25': {'image_path': '/Users/teabetita/Downloads/ITMGT-K Final Project Filez/wardrobe/images_finalproject25.png', 'color': 'off white', 'fabric': 'knitted', 'design': 'cutout', 'category': 'dress', 'length': 'short'},
    'item_26': {'image_path': '/Users/teabetita/Downloads/ITMGT-K Final Project Filez/wardrobe/images_finalproject26.png', 'color': 'green', 'fabric': 'linen', 'design': 'patterned', 'category': 'pants', 'length': 'long'},
    'item_27': {'image_path': '/Users/teabetita/Downloads/ITMGT-K Final Project Filez/wardrobe/images_finalproject27.png', 'color': 'white', 'fabric': 'linen', 'design': 'plain', 'category': 'pants', 'length': 'long'},
    'item_28': {'image_path': '/Users/teabetita/Downloads/ITMGT-K Final Project Filez/wardrobe/images_finalproject28.png', 'color': 'beige', 'fabric': 'linen', 'design': 'plain', 'category': 'pants', 'length': 'full'},
    'item_29': {'image_path': '/Users/teabetita/Downloads/ITMGT-K Final Project Filez/wardrobe/images_finalproject29.png', 'color': 'green', 'fabric': 'linen', 'design': 'plain', 'category': 'pants', 'length': 'full'},
    'item_30': {'image_path': '/Users/teabetita/Downloads/ITMGT-K Final Project Filez/wardrobe/images_finalproject30.png', 'color': 'beige', 'fabric': 'linen', 'design': 'plain', 'category': 'pants', 'length': 'full'},
    'item_31': {'image_path': '/Users/teabetita/Downloads/ITMGT-K Final Project Filez/wardrobe/images_finalproject31.png', 'color': 'blue', 'fabric': 'denim', 'design': 'plain', 'category': 'jeans', 'length': 'long'},
    'item_32': {'image_path': '/Users/teabetita/Downloads/ITMGT-K Final Project Filez/wardrobe/images_finalproject32.png', 'color': 'dark blue', 'fabric': 'denim', 'design': 'plain', 'category': 'jeans', 'length': 'long'},
    'item_33': {'image_path': '/Users/teabetita/Downloads/ITMGT-K Final Project Filez/wardrobe/images_finalproject33.png', 'color': 'beige', 'fabric': 'linen', 'design': 'embroidered', 'category': 'shorts', 'length': 'short'},
    'item_34': {'image_path': '/Users/teabetita/Downloads/ITMGT-K Final Project Filez/wardrobe/images_finalproject34.png', 'color': 'white', 'fabric': 'linen', 'design': 'plain', 'category': 'shorts', 'length': 'short'},
    'item_35': {'image_path': '/Users/teabetita/Downloads/ITMGT-K Final Project Filez/wardrobe/images_finalproject35.png', 'color': 'black', 'fabric': 'denim', 'design': 'plain', 'category': 'shorts', 'length': 'short'},
    'item_36': {'image_path': '/Users/teabetita/Downloads/ITMGT-K Final Project Filez/wardrobe/images_finalproject36.png', 'color': 'blue', 'fabric': 'denim', 'design': 'plain', 'category': 'shorts', 'length': 'short'},
    'item_37': {'image_path': '/Users/teabetita/Downloads/ITMGT-K Final Project Filez/wardrobe/images_finalproject37.png', 'color': 'beige', 'fabric': 'linen', 'design': 'plain', 'category': 'shorts', 'length': 'short'},
}


# Prepare the image features for machine learning
image_paths = []
items = []

for item, attributes in wardrobe.items():
    image_paths.append(attributes['image_path'])
    items.append(item)

# Function to extract image features using the pre-trained model
def extract_image_features(image_path):
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize((224, 224))  # Resize the image to match the input size of ResNet50
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = tf.keras.applications.resnet50.preprocess_input(expanded_img_array)
    features = model.predict(preprocessed_img)
    flattened_features = tf.keras.layers.Flatten()(features)
    return flattened_features


# Extract image features for the wardrobe dataset
wardrobe_features = np.array([extract_image_features(img_path) for img_path in image_paths])

# Flatten the wardrobe_features array
wardrobe_features_flattened = wardrobe_features.reshape(len(wardrobe_features), -1)

# Create and fit the NearestNeighbors model
nn_model = NearestNeighbors(n_neighbors=5, metric='cosine')
nn_model.fit(wardrobe_features_flattened)

# Function to recommend outfit combinations based on user input image
def recommend_outfits(image_path, top_n=5):
    user_features = extract_image_features(image_path)
    user_features = tf.reshape(user_features, (1, -1))  # Reshape to 2-D

    # Find the indices of the nearest neighbors (most similar items) using kneighbors
    _, nearest_neighbor_indices = nn_model.kneighbors(user_features)

    recommended_outfits = [items[idx] for idx in nearest_neighbor_indices[0]]

    return recommended_outfits, [image_paths[idx] for idx in nearest_neighbor_indices[0]][:top_n]

# Example usage
user_image_path = '/Users/teabetita/Downloads/ITMGT-K Final Project Filez/pegs/images_finalprojectpeg7.png'

top_n_recommendations = 5
recommended_outfits, recommended_image_paths = recommend_outfits(user_image_path, top_n=top_n_recommendations)

print("Recommended outfit combinations:")
for outfit, image_path in zip(recommended_outfits, recommended_image_paths):
    print(outfit)
    img = Image.open(image_path)
    plt.imshow(img)
    plt.axis('off')
    plt.show()
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['figure.figsize'] = (8, 8)
matplotlib.rcParams['axes.grid'] = False

pretrained_model = tf.keras.applications.MobileNetV2(include_top=True, weights='imagenet')
pretrained_model.trainable = False
# pretrained label
decode_predictions = tf.keras.applications.mobilenet_v2.decode_predictions

# Helper function to preprocess images so that it can be inputted in MobileNetV2
def preprocess(image):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (224,224))
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    image = image[None, ...]
    return image

# Helper function to extract labels from probability vector
def get_image_label(probs):
    return decode_predictions(probs, top=1)[0][0]

# Original Image
# Using a sample of a Labrador Retriever from wikipedia
image_path = tf.keras.utils.get_file('YellowLabradorLooking_new.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg')
image_raw = tf.io.read_file(image_path)
image = tf.image.decode_image(image_raw)
image = preprocess(image)
image_probs = pretrained_model.predict(image)

# Look at the image
plt.figure()
plt.imshow(image[0] * 0.5 + 0.5) # To change [-1, 1] to [0, 1]
_, image_class, class_confidence = get_image_label(image_probs)
plt.title('{}: {:.2f}% Confidence'.format(image_class, class_confidence*100))
plt.show()

# Create adversarial image

# Implementing fast gradient sign method
loss_object = tf.keras.losses.CategoricalCrossentropy()
def create_adversariral_pattern(input_image, input_label):
    with tf.GradientTape() as tape:
        tape.watch(input_image)
        prediction = pretrained_model(input_image)
        loss = loss_object(input_label, prediction)
    # Get the gradient of the loss w,r,t to the Input image
    gradient = tape.gradient(loss, input_image)
    # Get the sign of the gradient to create perturbation
    signed_grad = tf.sign(gradient)
    return signed_grad

# Visualize perturbation images
labrador_retriever_index = 208
label = tf.one_hot(labrador_retriever_index, image_probs.shape[-1])
label = tf.reshape(label, (1, image_probs.shape[-1]))

perturbations = create_adversariral_pattern(image, label)
plt.imshow(perturbations[0] * 0.5 + 0.5) # Change [-1, 1] to [0, 1]
plt.show()

# Functions to display result
def display_images(image, description):
    _, label, confidence = get_image_label(pretrained_model.predict(image))
    plt.figure()
    plt.imshow(image[0]*0.5 + 0.5)
    plt.title('{} \n {} : {:.2f}% Confidence'.format(description, label, confidence*100))
    plt.show()

epsilons = [0, 0.01, 0.03, 0.06, 0.1, 0.3]
descriptions = [('Eplsilon = {:0.3f}'.format(eps) if eps else 'input') for eps in epsilons]

# Attack images with perturbations
for i, eps in enumerate(epsilons):
    adv_x = image + eps*perturbations
    adv_x = tf.clip_by_value(adv_x, -1, 1)
    display_images(adv_x, descriptions[i])
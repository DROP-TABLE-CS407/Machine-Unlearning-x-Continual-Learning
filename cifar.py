import pickle
import numpy as np
import matplotlib.pyplot as plt

# Path to the dataset
DATASET_PATH = 'cifar-10-batches-py'  # Replace with your actual path

# Load a single batch file
def load_batch(filename):
    with open(filename, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    data = batch[b'data']
    labels = batch[b'labels']
    data = data.reshape(-1, 3, 32, 32)  # CIFAR-10 images are 32x32x3
    return data, labels

# Load all training batches
def load_cifar10_data(path):
    train_data = []
    train_labels = []

    for i in range(1, 6):
        data, labels = load_batch(f"{path}/data_batch_{i}")
        train_data.append(data)
        train_labels += labels

    train_data = np.concatenate(train_data)
    test_data, test_labels = load_batch(f"{path}/test_batch")
    
    return train_data, train_labels, test_data, test_labels

# Load CIFAR-10 data
train_data, train_labels, test_data, test_labels = load_cifar10_data(DATASET_PATH)
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Helper function to display images
def show_image(image, label):
    img = np.transpose(image, (1, 2, 0))  # Rearrange dimensions for plotting (32x32x3)
    plt.imshow(img)
    plt.title(classes[label])
    plt.axis('off')
    plt.show()

# Show a random image from the training set
random_idx = np.random.randint(len(train_data))
show_image(train_data[random_idx], train_labels[random_idx])
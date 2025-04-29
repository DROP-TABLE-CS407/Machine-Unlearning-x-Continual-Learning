import pickle
import numpy as np
import matplotlib.pyplot as plt

# Path to the dataset
DATASET_PATH = 'cifar-10-python' 
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
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


def split_into_nclasses(train_data, train_labels, num_classes):
    # Get unique classes
    unique_classes = np.unique(train_labels)
    
    # Randomly select num_classes from unique_classes
    selected_classes = np.random.choice(unique_classes, num_classes, replace=False)
    
    subset_train = []
    subset_labels = []
    
    for data, label in zip(train_data, train_labels):
        if label in selected_classes:
            subset_train.append(data)
            subset_labels.append(label)
    
    return np.array(subset_train), np.array(subset_labels)


# Helper function to display images
def show_image(image, label):
    img = np.transpose(image, (1, 2, 0))  # Rearrange dimensions for plotting (32x32x3)
    plt.imshow(img)
    plt.title(CLASSES[label])
    plt.axis('off')
    plt.show()

# split the dataset into classes from an inputed list of label names
def split_into_classes(train_data, train_labels, selected_class_names, classes = CLASSES):
    
    subset_train = []
    subset_labels = []
    selected_class_labels = get_class_indexes(selected_class_names, classes)
    for data, label in zip(train_data, train_labels):
        if label in selected_class_labels:
            subset_train.append(data)
            subset_labels.append(label)
    
    return np.array(subset_train), np.array(subset_labels)

# Helper function to get indexes of selected classes mapping names to indexes/labels
def get_class_indexes(selected_classes, classes):
    indexes = []
    for cls in selected_classes:
        if cls in classes:
            indexes.append(classes.index(cls))
    return indexes

# Show a random image from the training set
random_idx = np.random.randint(len(train_data))
show_image(train_data[random_idx], train_labels[random_idx])



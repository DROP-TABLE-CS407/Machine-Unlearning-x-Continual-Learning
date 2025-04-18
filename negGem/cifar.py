import pickle
import numpy as np
import matplotlib.pyplot as plt

# Path to the dataset
DATASET_PATH = './cifar-10-python' 
DATASET_PATH_100 = './cifar-100-python'
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
CLASSES_100_UNORDERED = [
    "beaver", "dolphin", "otter", "seal", "whale",
    "aquarium fish", "flatfish", "ray", "shark", "trout",
    "orchids", "poppies", "roses", "sunflowers", "tulips",
    "bottles", "bowls", "cans", "cups", "plates",
    "apples", "mushrooms", "oranges", "pears", "sweet peppers",
    "clock", "computer keyboard", "lamp", "telephone", "television",
    "bed", "chair", "couch", "table", "wardrobe",
    "bee", "beetle", "butterfly", "caterpillar", "cockroach",
    "bear", "leopard", "lion", "tiger", "wolf",
    "bridge", "castle", "house", "road", "skyscraper",
    "cloud", "forest", "mountain", "plain", "sea",
    "camel", "cattle", "chimpanzee", "elephant", "kangaroo",
    "fox", "porcupine", "possum", "raccoon", "skunk",
    "crab", "lobster", "snail", "spider", "worm",
    "baby", "boy", "girl", "man", "woman",
    "crocodile", "dinosaur", "lizard", "snake", "turtle",
    "hamster", "mouse", "rabbit", "shrew", "squirrel",
    "maple", "oak", "palm", "pine", "willow",
    "bicycle", "bus", "motorcycle", "pickup truck", "train",
    "lawn-mower", "rocket", "streetcar", "tank", "tractor"
]
CLASSES_100 = sorted(CLASSES_100_UNORDERED)
# Load a single batch file
def load_batch(filename, dataset = "cifar-10"):
    if dataset == "cifar-10":
        with open(filename, 'rb') as f:
            batch = pickle.load(f, encoding='bytes')
        data = batch[b'data']
        labels = batch[b'labels']
        data = data.reshape(-1, 3, 32, 32)  # CIFAR-10 images are 32x32x3
        return data, labels
    elif dataset == "cifar-100":
        with open(filename, 'rb') as f:
            print(filename)
            batch = pickle.load(f, encoding='bytes')
        data = batch[b'data']
        labels = batch[b'fine_labels']
        data = data.reshape(-1, 3, 32, 32)  # CIFAR-100 images are 32x32x3
        return data, labels
    else:
        print("Invalid dataset")
        return None,None


def load_data(path, dataset = "cifar-10"):
    if dataset == "cifar-10":
        train_data = []
        train_labels = []

        for i in range(1, 6):
            data, labels = load_batch(f"{path}/data_batch_{i}")
            train_data.append(data)
            train_labels += labels

        train_data = np.concatenate(train_data)
        test_data, test_labels = load_batch(f"{path}/test_batch")
        
        return train_data, train_labels, test_data, test_labels
    elif dataset == "cifar-100":
        train_data = []
        train_labels = []
        train_data , train_labels = load_batch(f"{path}/train", dataset="cifar-100")
        test_data, test_labels = load_batch(f"{path}/test", dataset="cifar-100")

        # Map labels to the indexes of CLASSES_100_UNORDERED
        label_mapping = {i: CLASSES_100_UNORDERED.index(cls) for i, cls in enumerate(CLASSES_100)}
        train_labels = [label_mapping[label] for label in train_labels]
        test_labels = [label_mapping[label] for label in test_labels]


        return train_data, train_labels, test_data, test_labels
    else:
        print("Invalid dataset")
        return None,None,None,None
    


# Load all training batches
def load_cifar10_data(path):
    return load_data(path, dataset = "cifar-10")

# Load CIFAR-10 data
train_data, train_labels, test_data, test_labels = load_cifar10_data(DATASET_PATH)
train_data_100, train_labels_100, test_data_100, test_labels_100 = load_data(DATASET_PATH_100, dataset = "cifar-100")


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
def show_image(image, label, dataset = "cifar-10"):
    if dataset == "cifar-10":
        img = np.transpose(image, (1, 2, 0))  # Rearrange dimensions for plotting (32x32x3)
        plt.imshow(img)
        plt.title(CLASSES[label])
        plt.axis('off')
        plt.show()
    elif dataset == "cifar-100":
        img = np.transpose(image, (1, 2, 0))  # Rearrange dimensions for plotting (32x32x3)
        plt.imshow(img)
        plt.title(CLASSES_100_UNORDERED[label])
        plt.axis('off')
        plt.show()
    else:
        print("Invalid dataset")
        return None
    
    
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
#random_idx = np.random.randint(len(train_data))
#show_image(train_data[random_idx], train_labels[random_idx])
#random_idx = np.random.randint(len(train_data_100))
#show_image(train_data_100[random_idx], train_labels_100[random_idx], dataset = "cifar-100")




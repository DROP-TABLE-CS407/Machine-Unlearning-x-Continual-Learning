import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
# make cuda available if available

from cifar import load_cifar10_data, show_image

#start timer for training
import time
initialisation = time.time()

# make a resnet model

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
model.eval()

# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    print("|| GPU is available")
    model = model.cuda()

# Path to the dataset
DATASET_PATH = 'cifar-10-batches-py'  # Replace with your actual path

# Load CIFAR-10 data
train_data, train_labels, test_data, test_labels = load_cifar10_data(DATASET_PATH)
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# upscale all images to 224x224 from 32x32 from test_data and train_data
resized_train_data = np.zeros((len(train_data), 3, 224, 224), dtype=np.uint8)
resized_test_data = np.zeros((len(test_data), 3, 224, 224), dtype=np.uint8)

for i in range(len(train_data)):
    resized_train_data[i] = np.array(Image.fromarray(train_data[i].transpose(1, 2, 0)).resize((224, 224))).transpose(2, 0, 1)
for i in range(len(test_data)):
    resized_test_data[i] = np.array(Image.fromarray(test_data[i].transpose(1, 2, 0)).resize((224, 224))).transpose(2, 0, 1)

train_data = resized_train_data
test_data = resized_test_data

epoch = 2
learning_rate = 0.0005

# print  sizes of train data and test data
print("||==========================TRAINING HYPERPARAMETERS========================||")
print("|| Train data size = ", len(train_data))
print("|| Test data size = ", len(test_data))
print("|| Optimizer = Adam")
print("|| Loss function = CrossEntropyLoss")
print("|| Learning rate = ", learning_rate)
print("|| Epochs = ", epoch)

# define the loss function
criterion = torch.nn.CrossEntropyLoss()

# define the optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# turn the data into a tensor
train_labels_tensor = torch.tensor(train_labels)
resized_train_data_tensor = torch.tensor(resized_train_data).float()    

# move tensors to GPU if available
if torch.cuda.is_available():
    train_labels_tensor = train_labels_tensor.cuda()



# keep track of the loss
losses = []

# calculate the time taken to load the data
initialisation = time.time() - initialisation

# start training timer
training = time.time()

print("||=========================START TRAINING=======================||")

# train the model

for j in range(0, len(train_data), 1000):
    # get the input and output
    gpu_temp = resized_train_data_tensor[j:j+1000]
    label = train_labels_tensor[j:j+1000]
    if torch.cuda.is_available():
        gpu_temp = gpu_temp.cuda()
    for i in range(epoch):

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(gpu_temp)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()

        # print statistics
        losses.append(loss.item())
        

        print(f"|| Batch {j+1}, Epoch {i+1} Loss: {loss.item()}")
        
    # print the loss every epoch
    del gpu_temp


print("||==========================END TRAINING=======================||\n\n\n")

# calculate the time taken to train the model
training = time.time() - training

# keep track of accuracy
correct1 = 0
correct2 = 0
total1 = len(train_data)
total2 = len(test_data)

# start test/train accuracy timer
accuracy = time.time()

print("||===================START TEST/TRAIN ACCURACY=================||")

# turn the data into a tensor
test_data_tensor = torch.tensor(test_data).float()
test_labels_tensor = torch.tensor(test_labels)

# move tensors to GPU if available
if torch.cuda.is_available():
    test_data_tensor = test_data_tensor.cuda()
    test_labels_tensor = test_labels_tensor.cuda()
    
# test the model on the training data
with torch.no_grad():
    for i in range(0, len(train_data), 1000):
        # get the input and output
        img = resized_train_data_tensor[i:i+1000]
        label = train_labels_tensor[i:i+1000]
        
        if torch.cuda.is_available():
            img = img.cuda()

        # normalise the image
        #img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)

        # get the prediction
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)

        # get the number of correct predictions
        correct1 += (predicted == label).sum().item()
        
        del img
        
print(f"|| Train Set Accuracy: {correct1 / total1 * 100:.2f}%")
    
# test the model
with torch.no_grad():
    for i in range(0, len(test_data), 1000):
        # get the input and output
        img = test_data_tensor[i:i+1000]
        label = test_labels_tensor[i:i+1000]
        
        if torch.cuda.is_available():
            img = img.cuda()

        # normalise the image
        #img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)

        # get the prediction
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)

        # get the number of correct predictions
        correct2 += (predicted == label).sum().item()
        
        del img

# print the accuracy
print(f"|| Test Set Accuracy: {correct2 / total2 * 100:.2f}%")

print("||==============================END==========================||")

# calculate the time taken to get the accuracy
accuracy = time.time() - accuracy

# print timing information
print(f"|| Time to load data: {initialisation:.2f}s")
print(f"|| Time to train model: {training:.2f}s")
print(f"|| Time to get accuracy: {accuracy:.2f}s")
print(f"|| Total time: {initialisation + training + accuracy:.2f}s")

print("||==============================END==========================||")

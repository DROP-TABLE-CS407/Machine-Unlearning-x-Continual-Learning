import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import time
# make cuda available if available

from cifar import load_cifar10_data, show_image

# we want to create a resnet 18 model for 32x32 images
# avoid upscaling, the model will take 32x32 images on input as opposed to 224x224

initialisation = time.time()

class ResNet18CIFAR(torch.nn.Module):
    def __init__(self):
        super(ResNet18CIFAR, self).__init__()
        self.resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
        # change the first layer to accept 32x32 images with 3 channels rather than 224x224 images
        # check the size of the input layer
        print("|| conv1 weight size: ", self.resnet.conv1.weight.size())
        print("|| fc weight size: ", self.resnet.fc.weight.size())
        self.resnet.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet.fc = torch.nn.Linear(512, 10)
        self.resnet.maxpool = torch.nn.Identity()
        # change input layer to accept 32x32 images


        
        # List all layers in the resnet18 model
        for name, layer in self.resnet.named_children():
            print(f"Layer: {name} -> {layer}")
        
        #print("|| conv1 weight size: ", self.resnet.conv1.weight.size())
        #print("|| fc weight size: ", self.resnet.fc.weight.size())

    def forward(self, x):
        return self.resnet(x)
    
# make a resnet model
model = ResNet18CIFAR()
model.eval()



# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    print("||==========================GPU is available==========================||")
    model = model.cuda()


# Path to the dataset
DATASET_PATH = 'cifar-10-batches-py'  # Replace with your actual path

# Load CIFAR-10 data
train_data, train_labels, test_data, test_labels = load_cifar10_data(DATASET_PATH)
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

epoch = 400
learning_rate = 0.0005
batch_size = 32

# print  sizes of train data and test data
print("||==========================TRAINING HYPERPARAMETERS========================||")
print("|| Train data size = ", len(train_data))
print("|| Test data size = ", len(test_data))
print("|| Optimizer = Adam")
print("|| Loss function = CrossEntropyLoss")
print("|| Learning rate = ", learning_rate)
print("|| Epochs = ", epoch)
print("|| Batch size = ", batch_size)   

# define the loss function
criterion = torch.nn.CrossEntropyLoss()

# define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# turn the data into a tensor
train_labels_tensor = torch.tensor(train_labels)
train_data_tensor = torch.tensor(train_data).float()    

# move tensors to GPU if available
if torch.cuda.is_available():
    train_data_tensor = train_data_tensor.cuda()
    train_labels_tensor = train_labels_tensor.cuda()



# keep track of the loss
losses = []

# calculate the time taken to load the data
initialisation = time.time() - initialisation

# start training timer
training = time.time()

print("||=========================START TRAINING=======================||")

# train the model

#for j in range(0, len(train_data)):
for i in range(epoch):
    for j in range(0, len(train_data), batch_size):
        # get the input and output
        img = train_data_tensor[j:j+batch_size]
        label = train_labels_tensor[j:j+batch_size]
        
        # normalise the image
        #img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(img)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()

        # print statistics
        losses.append(loss.item())
    
    print(f"|| Epoch {i+1} Loss: {loss.item()}")    


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
        img = train_data_tensor[i:i+1000]
        label = train_labels_tensor[i:i+1000]
        

        # normalise the image
        #img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)

        # get the prediction
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)

        # get the number of correct predictions
        correct1 += (predicted == label).sum().item()
        
print(f"|| Train Set Accuracy: {correct1 / total1 * 100:.2f}%")
    
# test the model
with torch.no_grad():
    for i in range(0, len(test_data), 1000):
        # get the input and output
        img = test_data_tensor[i:i+1000]
        label = test_labels_tensor[i:i+1000]
        

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

torch.save(model, 'resnet18_cifar78ACC.pth')

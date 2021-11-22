import os.path

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np

import matplotlib.pyplot as plt


def weights_init(m):
	if type(m) == nn.Linear:
		m.weight.data.normal_(0.0, 1e-3)
		m.bias.data.fill_(0.)


def update_lr(optimizer, lr):
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr


# --------------------------------
# Device configuration
# --------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device: %s' % device)

# --------------------------------
# Hyper-parameters
# --------------------------------
input_size = 3
num_classes = 10
hidden_size = [128, 512, 512, 512, 512]
num_epochs = 50
batch_size = 200
learning_rate = 2e-3
learning_rate_decay = 0.95
reg = 0.001
num_training = 49000
num_validation = 1000
norm_layer = 'BN'  # norm_layer = 'BN'
print(hidden_size)

# -------------------------------------------------
# Load the CIFAR-10 dataset
# -------------------------------------------------
#################################################################################
# TODO: Q3.a Choose the right data augmentation transforms with the right       #
# hyper-parameters and put them in the data_aug_transforms variable             #
#################################################################################
data_aug_transforms = []
# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

data_aug_parameters = {
    "RC_size": 32, "RC_padding": 2,  # default = none
    "CJ_brightness": 0,  # default = 0
    "CJ_contrast": 0,  # default = 0
    "CJ_saturation": 0,  # default = 0
    "CJ_hue": 0,  # default = 0
    "P_padding": 3, "P_type": "constant",  # default = constant
    "HF_p": 0.5, "VF_p": 0.5, "RR_degrees": 10, "RG_p": 0.2}

data_aug_transforms += [transforms.RandomCrop(data_aug_parameters["RC_size"], padding=data_aug_parameters["RC_padding"]),
                        transforms.RandomHorizontalFlip(data_aug_parameters["HF_p"]),
                        # transforms.RandomVerticalFlip(data_aug_parameters["VF_p"]),
                        transforms.RandomRotation(degrees=data_aug_parameters["RR_degrees"]),
                        # transforms.ColorJitter(brightness=data_aug_parameters["CJ_brightness"], contrast=data_aug_parameters["CJ_contrast"], saturation=data_aug_parameters["CJ_saturation"], hue=data_aug_parameters["CJ_hue"]),
                        transforms.RandomGrayscale(data_aug_parameters["RG_p"]), ]


# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
norm_transform = transforms.Compose(data_aug_transforms + [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
cifar_dataset = torchvision.datasets.CIFAR10(root='datasets/', train=True, transform=norm_transform, download=True)

test_dataset = torchvision.datasets.CIFAR10(root='datasets/', train=False, transform=test_transform)

# -------------------------------------------------
# Prepare the training and validation splits
# -------------------------------------------------
mask = list(range(num_training))
train_dataset = torch.utils.data.Subset(cifar_dataset, mask)
mask = list(range(num_training, num_training + num_validation))
val_dataset = torch.utils.data.Subset(cifar_dataset, mask)

# -------------------------------------------------
# Data loader
# -------------------------------------------------
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


# -------------------------------------------------
# Convolutional neural network (Q1.a and Q2.a)
# Set norm_layer for different networks whether using batch normalization
# -------------------------------------------------
class ConvNet(nn.Module):
	def __init__(self, input_size, hidden_layers, num_classes, norm_layer=None):
		super(ConvNet, self).__init__()
		#################################################################################
		# TODO: Initialize the modules required to implement the convolutional layer    #
		# described in the exercise.                                                    #
		# For Q1.a make use of conv2d and relu layers from the torch.nn module.         #
		# For Q2.a make use of BatchNorm2d layer from the torch.nn module.              #
		# For Q3.b Use Dropout layer from the torch.nn module.                          #
		#################################################################################
		layers = []
		# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

		# Modified Hyperparameters
		dropout = 0  # default 0

		# Create input layer,  eventually also Normalization Layers
		layers += [nn.Conv2d(input_size, hidden_layers[0], 3, padding=1)]
		if norm_layer == 'BN': layers += [nn.BatchNorm2d(hidden_layers[0], device=device)]

		# For Each other layers
		for index, value in enumerate(hidden_layers[:-1]):
			layers += [nn.MaxPool2d((2, 2), 2), nn.ReLU()]
			layers += [nn.Dropout(dropout)]
			layers += [nn.Conv2d(hidden_layers[index], hidden_layers[index + 1], 3, padding=1)]
			if norm_layer == 'BN': layers += [nn.BatchNorm2d(hidden_layers[index + 1], device=device)]

		# Create output with flatten
		layers += [nn.MaxPool2d((2, 2), 2), nn.Flatten(), nn.ReLU()]
		layers += [nn.Linear(hidden_layers[-1], num_classes)]
		layers += [nn.Dropout(dropout)]

		self.layers = nn.Sequential(*layers)

	# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

	def forward(self, x):
		#################################################################################
		# TODO: Implement the forward pass computations                                 #
		#################################################################################
		# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

		out = self.layers(x)
		# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
		return out


# -------------------------------------------------
# Calculate the model size (Q1.b)
# if disp is true, print the model parameters, otherwise, only return the number of parameters.
# -------------------------------------------------
def PrintModelSize(model, disp=True):
	#################################################################################
	# TODO: Implement the function to count the number of trainable parameters in   #
	# the input model. This useful to track the capacity of the model you are       #
	# training                                                                      #
	#################################################################################
	# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
	model_sz = sum(p.numel() for p in model.parameters())
	print(model_sz)
	# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
	return model_sz


# -------------------------------------------------
# Calculate the model size (Q1.c)
# visualize the convolution filters of the first convolution layer of the input model
# -------------------------------------------------
def VisualizeFilter(model):
	#################################################################################
	# TODO: Implement the functiont to visualize the weights in the first conv layer#
	# in the model. Visualize them as a single image of stacked filters.            #
	# You can use matlplotlib.imshow to visualize an image in python                #
	#################################################################################
	# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
	weights = model.layers[0].weight.data

	# print(weights)
	max = torch.max(weights)
	min = torch.min(weights)

	# Figure out how 'wide' each range is
	leftSpan = max - min
	rightSpan = 255 - 0

	# Convert the left range into a 0-1 range (float)
	vals = (weights - min) / float(leftSpan)

	# Convert the 0-1 range into a value in the right range.
	weights = 0 + (vals * rightSpan)
	weights = weights.int()

	# weights = (weights-min)/max
	print(weights)
	fig = plt.figure(1)
	rows = 8
	columns = 16

	# print(weights.size(),weights)
	for idx, image in enumerate(weights):
		fig.add_subplot(rows, columns, idx + 1, frame_on=False, xticks=[], yticks=[], xticklabels=[], yticklabels=[])
		plt.imshow(image.cpu())
	plt.show()
	# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****


# ======================================================================================
# Q1.a: Implementing convolutional neural net in PyTorch
# ======================================================================================
# In this question we will implement a convolutional neural networks using the PyTorch
# library.  Please complete the code for the ConvNet class evaluating the model
# --------------------------------------------------------------------------------------
model = ConvNet(input_size, hidden_size, num_classes, norm_layer=norm_layer).to(device)
# Q2.a - Initialize the model with correct batch norm layer

model.apply(weights_init)
# Print the model
print(model)
# Print model size
# ======================================================================================
# Q1.b: Implementing the function to count the number of trainable parameters in the model
# ======================================================================================
PrintModelSize(model)
# ======================================================================================
# Q1.a: Implementing the function to visualize the filters in the first conv layers.
# Visualize the filters before training
# ======================================================================================
VisualizeFilter(model)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=reg)

# Train the model
lr = learning_rate
total_step = len(train_loader)
loss_train = []
loss_val = []
best_accuracy = None
accuracy_val = []
best_model = type(model)(input_size, hidden_size, num_classes, norm_layer=norm_layer)  # get a new instance
# best_model = ConvNet(input_size, hidden_size, num_classes, norm_layer=norm_layer)

for epoch in range(num_epochs):
	model.train()

	loss_iter = 0
	for i, (images, labels) in enumerate(train_loader):
		# Move tensors to the configured device
		images = images.to(device)
		labels = labels.to(device)

		# Forward pass
		outputs = model(images)
		loss = criterion(outputs, labels)

		# Backward and optimize
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		loss_iter += loss.item()

		if (i + 1) % 100 == 0:
			print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

	loss_train.append(loss_iter / (len(train_loader) * batch_size))

	# Code to update the lr
	lr *= learning_rate_decay
	update_lr(optimizer, lr)

	model.eval()
	with torch.no_grad():
		correct = 0
		total = 0
		loss_iter = 0
		for images, labels in val_loader:
			images = images.to(device)
			labels = labels.to(device)

			outputs = model(images)
			_, predicted = torch.max(outputs.data, 1)

			total += labels.size(0)
			correct += (predicted == labels).sum().item()

			loss = criterion(outputs, labels)
			loss_iter += loss.item()

		loss_val.append(loss_iter / (len(val_loader) * batch_size))

		accuracy = 100 * correct / total
		accuracy_val.append(accuracy)
		print('Validation accuracy is: {} %'.format(accuracy))
		#################################################################################
		# TODO: Q2.b Implement the early stopping mechanism to save the model which has #
		# the model with the best validation accuracy so-far (use best_model).          #
		#################################################################################

		# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
		early_stop = False
		patience = 3
		if epoch > patience - 1:
			for j in range(patience - 1):
				if max(accuracy_val) > list(reversed(accuracy_val))[j]:
					if "not_improving_epochs" in locals(): not_improving_epochs += 1
					else: not_improving_epochs = 1
					print('Not saving the model')
				else:
					not_improving_epochs = 0
					best_model = model
					print("Saving the model")
					break
				if not_improving_epochs >= patience:
					early_stop = True
					print('Early stopping')
					break
				break
		#if early_stop:
		#	print('Stop')
		#	break
torch.save(best_model.state_dict(), 'model.ckpt')
	# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
model.eval()

plt.figure(2)
plt.plot(loss_train, 'r', label='Train loss')
plt.plot(loss_val, 'g', label='Val loss')
plt.legend()
plt.show()

plt.figure(3)
plt.plot(accuracy_val, 'r', label='Val accuracy')
plt.legend()
plt.show()

#################################################################################
# TODO: Q2.b Implement the early stopping mechanism to load the weights from the#
# best model so far and perform testing with this model.                        #
#################################################################################
# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
model.load_state_dict(torch.load('model.ckpt'))
# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

# Compute accuracy on the test set
with torch.no_grad():
	correct = 0
	total = 0
	for images, labels in test_loader:
		images = images.to(device)
		labels = labels.to(device)
		outputs = model(images)
		_, predicted = torch.max(outputs.data, 1)
		total += labels.size(0)
		correct += (predicted == labels).sum().item()
		if total == 1000:
			break

	with open("C:\\Users\\orlan\\PycharmProjects\\AML_assignment2\\Other\\esnobn50\\out.txt", 'w') as f:
		f.write('Accuracy of the network on the {} test images: {} %'.format(total, 100 * correct / total))

	print('Accuracy of the network on the {} test images: {} %'.format(total, 100 * correct / total))

# Q1.c: Implementing the function to visualize the filters in the first conv layers.
# Visualize the filters before training
VisualizeFilter(model)

# Save the model checkpoint
# torch.save(model.state_dict(), 'model.ckpt')
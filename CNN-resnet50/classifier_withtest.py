import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import pandas as pd
import numpy as np
from torchvision import transforms, models, datasets
import time
import copy
import pickle
import csv
# from tiny_classifier import TinyCNN

optim_algorithm = 'adam'
weighted = False
runcount = 1


data_transforms = { 
		'train': transforms.Compose([
			#transforms.Resize((32,32)),
			transforms.ToTensor(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	]),
		'val': transforms.Compose([
			#transforms.Resize((32,32)),
			transforms.ToTensor(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	]),
		'test': transforms.Compose([
			#transforms.Resize((32,32)),
			transforms.ToTensor(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	])
	}

data_dir = 'images/'


image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
					for x in ['train', 'val', 'test']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=12, shuffle=True, num_workers=1)
					for x in ['train', 'val', 'test']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
class_names = image_datasets['train'].classes

print(class_names)
class_counts = [4993, 4982, 4988, 5000, 4951, 4980, 4999, 4813, 4471, 3940, 3505, 2021, 1487, 1358, 1317, 1217, 1140, 1111, 976, 889, 745, 681, 671, 582, 9896]
class_weights = np.array(class_counts)
class_weights = 1/class_weights
class_weights = list(class_weights)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

'''
fi_dataset = datasets.ImageFolder(root='', transform=data_transform)
dataset_loader = torch.utils.data.DataLoader(fi_dataset, batch_size=8, shuffle=True, num_workers=4)
'''

def train_model(model, criterion, optimizer, scheduler, num_epochs=50):
	since = time.time()

	print('copying the model')
	best_model_wts = copy.deepcopy(model.state_dict())
	print('done with deep copy')
	best_acc = 0.0

	print('entering training loop')
	for epoch in range(num_epochs):
		print('Epoch {}/{}'.format(epoch, num_epochs - 1))
		print('-' * 10)

		for phase in ['train', 'val']:
			if phase == 'train':
				scheduler.step()
				model.train()
			else:
				model.eval()

			running_loss = 0.0
			running_corrects = 0.0

			for inputs, labels in dataloaders[phase]:
				inputs = inputs.to(device)
				labels = labels.to(device)

				optimizer.zero_grad()

				# forward
				with torch.set_grad_enabled(phase == 'train'):
					outputs = model(inputs)
					#print(outputs[0])
					#print(labels)
					#print(len(inputs), len(outputs))
					logits, preds = torch.max(outputs, 1)
					#print(logits, labels)
					loss = criterion(outputs, labels)

					# backwards + optimizer only if in training phase
					if phase == 'train':
						loss.backward()
						optimizer.step()

				# statistics
				running_loss += loss.item() * inputs.size(0)
				running_corrects += torch.sum(preds == labels.data)

			epoch_loss = running_loss / dataset_sizes[phase]
			epoch_acc = running_corrects.double() / dataset_sizes[phase]

			print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

			# deep copy the model
			if phase == 'val' and epoch_acc > best_acc:
				best_acc = epoch_acc
				best_model_wts = copy.deepcopy(model.state_dict())

		latest_model = copy.deepcopy(model.state_dict())
		pickle.dump(latest_model, open("caddy_%s_run%d_latest.pickle" % (optim_algorithm, runcount), "wb"))
			

	time_elapsed = time.time() - since
	print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
	print('Best val Acc: {:4f}'.format(best_acc))

	'''# load best model weights
	model.load_state_dict(best_model_wts)
	return model'''

	# load best model weights
	latest_model = copy.deepcopy(model.state_dict())
	model.load_state_dict(best_model_wts)
	return model, latest_model


# Load a pretrained model and reset final fully connected layer
'''print('loading pretrained model')
model_ft = models.resnet50(pretrained=True)
num_ftrs = model_ft.fc.in_features
print('attaching fc layer')
model_ft.fc = nn.Linear(num_ftrs, 17)

# Load CNN
# model_ft = TinyCNN()

print('sending model to device')
model_ft = model_ft.to(device)

weights = torch.tensor(class_weights)
weights = weights.to(device)
if weighted:
	criterion = nn.CrossEntropyLoss(weight=weights)
else:
	criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
if optim_algorithm == 'sgd':
	optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.0005, momentum=0.9)
elif optim_algorithm == 'adam':
	optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.0005)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=100, gamma=0.01)

print('Training the classifier')
model_ft, latest_model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=50)

pickle.dump(model_ft, open("caddy_%s_run%d_best.pickle" % (optim_algorithm, runcount), "wb"))
pickle.dump(latest_model_ft, open("caddy_%s_run%d_latest.pickle" % (optim_algorithm, runcount), "wb"))'''

# loading the model
#model_ft = pickle.load(open("classifier.pickle", "rb"))

def test_model(model):
	nb_classes = 17
	confusion_matrix = torch.zeros(nb_classes, nb_classes)

	with torch.no_grad():
		for i, (inputs, classes) in enumerate(dataloaders['test']):
			inputs = inputs.to(device)
			classes = classes.to(device)
			outputs = model(inputs)
			_, preds = torch.max(outputs, 1)
			for t, p in zip(classes.view(-1), preds.view(-1)):
				confusion_matrix[t.long(), p.long()] += 1

	print(confusion_matrix)
	print('per class accuracy', confusion_matrix.diag()/confusion_matrix.sum(1))
	print('test classes', image_datasets['test'].classes)

	confusion_matrix.to(torch.int32)
	#confusion_matrix = confusion_matrix.numpy()

	#with open('resnet-50_test2.csv', 'w') as f:
	#	csvwriter = csv.writer(f)
	#	for l in confusion_matrix:
	#		l = list(l)
	#		csvwriter.writerows(l)

	np.savetxt('resnet-50_test2.csv', confusion_matrix)


model_ft = pickle.load(open("caddy_%s_run%d_best.pickle" % (optim_algorithm, runcount), "rb"))

test_model(model_ft)
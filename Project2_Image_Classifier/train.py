import argparse

from torch import nn, optim
from torchvision import datasets, transforms, models

from utils import load_data
from image_classifier import create_classifier, train_model, test_model, save_model

parser = argparse.ArgumentParser(description='Train a network on a data set with train.py.')
parser.add_argument('data_directory', action='store', help='Enter path to training data')
parser.add_argument('--arch', action='store', dest='pre_trained_model',
                    default='vgg16', help='Choose an architecture that has a layer named classifier, '
                                          'please note resnet18 is not satisfied')
parser.add_argument('--dropout', action='store', dest='dropout',
                    default=0.2, help='The probability of an element to be zeroed for dropout purpose', type=float)
parser.add_argument('--hidden_units', action='store', dest='hidden_units',
                    default=500, help='Set hidden units for model training', type=int)
parser.add_argument('--learning_rate', action='store', dest='learning_rate',
                    default=0.001, help='Set learning rate for model training', type=float)
parser.add_argument('--epochs', action='store', dest='epochs',
                    default=2, help='Set number of epochs for model training', type=int)
parser.add_argument('--print_every', action='store', dest='print_every',
                    default=5, help='Set the frequency of message print for model training', type=int)
parser.add_argument('--save_dir', action='store', dest='save_directory',
                    default='checkpoint.pth', help='Set directory to save checkpoints')
parser.add_argument('--gpu', action='store_true', default=False,
                    help='Use GPU for training, default is off', type=bool)
params = parser.parse_args()

data_dir = params.data_directory
pre_trained_model = params.pre_trained_model
dropout = params.dropout
hidden_units = params.hidden_units
lr = params.learning_rate
epochs = params.epochs
print_every = params.print_every
save_dir = params.save_directory
gpu_mode = params.gpu

# step 1: load the data
train_datasets, valid_datasets, test_datasets, train_loader, valid_loader, test_loader = load_data(data_dir=data_dir)

# step 2: Building and training the classifier

# step 2.1: load a pre-trained model
model = getattr(models, pre_trained_model)(pretrained=True)

# step 2.2: replace the pre-trained model with a new classifier
input_units = model.classifier[0].in_features
create_classifier(model=model, input_units=input_units, hidden_units=hidden_units, dropout=dropout)

# step 2.3: train the classifier layers using backpropagation using the pre-trained network to get the features
criterion = nn.NLLLoss()
# use adam optimizer to avoid local minimal
optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
model = train_model(model=model, epochs=epochs, train_loader=train_loader, valid_loader=valid_loader,
                    criterion=criterion, optimizer=optimizer, print_every=print_every, gpu_mode=gpu_mode)

# step 3: test model
test_model(model=model, test_loader=test_loader, criterion=criterion, gpu_mode=gpu_mode)

# step 4: save model
save_model(model=model, train_datasets=train_datasets, optimizer=optimizer, epochs=epochs, save_dir=save_dir)

import torch
from torch import nn
from collections import OrderedDict
import time
from PIL import Image
import numpy as np

def create_classifier(model, input_units, hidden_units, dropout):
    """
    create a new, untrained feed-forward network as a classifier based on the specified parameters
    :param model: the pre-trained model
    :param input_units: units of input layer
    :param hidden_units: units of hidden layer
    :param dropout: dropout rate
    :return:
    """
    # freeze parameter so that we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_units, hidden_units)),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(p=dropout)),
        ('fc2', nn.Linear(hidden_units, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    model.classifier = classifier


def train_model(model, epochs, train_loader, valid_loader, criterion, optimizer, print_every, gpu_mode):
    """
    train a model with specified parameters
    :param model: the model to be trained
    :param epochs: epochs to be used for training a model
    :param train_loader: training data loader
    :param valid_loader: validation data loader
    :param criterion: use negative likelihood loss function as criterion
    :param optimizer: adam optimizer
    :param print_every: frequency of training debug message print
    :param gpu_mode: whether to use gpu_mode
    :return: trained model
    """
    # move model to cuda device
    device = torch.device("cuda" if torch.cuda.is_available() and gpu_mode else "cpu")
    model.to(device)

    steps = 0
    running_loss = 0
    start = time.time()

    print("begin model training...")
    for epoch in range(epochs):
        epoch_start = time.time()
        for inputs, labels in train_loader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in valid_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        valid_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch + 1}/{epochs}.. "
                      f"Train loss: {running_loss / print_every:.3f}.. "
                      f"Valid loss: {valid_loss / len(valid_loader):.3f}.. "
                      f"Valid accuracy: {accuracy / len(valid_loader):.3f}")
                running_loss = 0
                model.train()
        epoch_end = time.time()
        print(f"Epoch time for training & validation: {(epoch_end - epoch_start):.3f} seconds")
    end = time.time()
    print(f"Total time for training: {(end - start):.3f} seconds")

    print("end model training...")
    return model


def test_model(model, test_loader, criterion, gpu_mode):
    """
    test the trained model using test dataset
    :param model: the trained model
    :param test_loader: test data loader
    :param criterion: the criterion
    :param gpu_mode: whether to use gpu_mode
    :return:
    """
    test_loss = 0
    accuracy = 0
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() and gpu_mode else "cpu")

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)
            test_loss += batch_loss.item()

            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print(f"Valid loss: {test_loss/len(test_loader):.3f}.. "
          f"Valid accuracy: {accuracy/len(test_loader):.3f}")


def save_model(model, train_datasets, optimizer, epochs, save_dir):
    """
    save the trained model as a pth file
    :param model: the trained model
    :param train_datasets: training dataset which contains the class to index
    :param optimizer: the optimizer
    :param epochs: the number of epochs during model training
    :param save_dir: the directory to save the pth file
    :return:
    """
    checkpoint = {
        # feature weights
        'state_dict': model.state_dict(),
        # mapping of classes to indices
        'class_to_idx': train_datasets.class_to_idx,
        # the new model classifier
        'classifier': model.classifier,
        # optimizer state
        'optimizer_state': optimizer.state_dict,
        # number of epochs
        'num_epochs': epochs,
    }

    torch.save(checkpoint, save_dir)


def load_checkpoint(model, filepath):
    """
    load the model from pth file
    :param model: the pre-trained model
    :param filepath: the pth file path
    :return: trained model with loaded weights and other parameters
    """
    checkpoint = torch.load(filepath)
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    # freeze parameter so that we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    return model


def process_image(image_path):
    """
    Scales, crops, and normalizes a PIL image for a PyTorch model, returns an Numpy array
    :param image_path: image path
    :return: numpy array of the image
    """
    # Process a PIL image for use in a PyTorch model
    # original pil image
    pil_image = Image.open(image_path)

    # resize the images where the shortest side is 256 pixels, keeping the aspect ratio
    ori_width, ori_height = pil_image.size
    (width, height) = (ori_width, ori_height)
    if ori_width > ori_height:
        height = 256
    else:
        width = 256
    #     print("w: {}, h: {}".format(width, height))
    pil_image.thumbnail((width, height))

    # crop out the center 224x224 portion of the image
    center = ori_width/4, ori_height/4
    half_crop_size = 224/2
    (left, upper, right, lower) = (center[0]-half_crop_size, center[1]-half_crop_size,
                                   center[0]+half_crop_size, center[1]+half_crop_size)
    pil_image = pil_image.crop((left, upper, right, lower))

    # Color channels of images are typically encoded as integers 0-255, but the model expected floats 0-1
    np_image = np.array(pil_image)/255

    # normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean)/std

    #     print(np_image)

    # The color channel needs to be first and retain the order of the other two dimensions.
    np_image = np_image.transpose(2, 0, 1)

    return np_image


def predict(image_path, model, topk=5):
    """
    Predict the class (or classes) of an image using a trained deep learning model.
    :param image_path:
    :param model:
    :param topk:
    :return:
    """

    # switch to cpu and eval mode
    model.eval()

    # process the image
    image = process_image(image_path)

    # convert to torch from numpy array
    ## fix errors:
    # Dimension error
    # Expected object of type torch.DoubleTensor but found type torch.FloatTensor'
    torch_image = torch.from_numpy(np.expand_dims(image, axis=0)).type(torch.FloatTensor)

    # model prediction to get log prob
    log_ps = model.forward(torch_image)

    # convert to predicted prob
    ps = torch.exp(log_ps)

    # use topk to get prob and index of class
    top_probs, top_indexes = ps.topk(topk)

    #     print(top_probs)
    #     print(top_classes)

    # convert to list
    top_probs_list = top_probs.detach().numpy()[0]
    top_indexes_list = top_indexes.detach().numpy()[0]

    # invert index-class
    class_to_index = model.class_to_idx
    index_to_class = {y:x for x, y in class_to_index.items()}
    #     print(index_to_class)

    # convert index list to class list
    top_classes_list = []
    for index in top_indexes_list:
        top_classes_list.append(index_to_class[index])

    return top_probs_list, top_classes_list
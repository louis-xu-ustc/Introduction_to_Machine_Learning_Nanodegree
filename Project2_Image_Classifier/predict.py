import argparse
import json

from torchvision import datasets, transforms, models

from image_classifier import load_checkpoint, process_image, predict

parser = argparse.ArgumentParser(description='Predict flower name from an image along with the prob of that name')
parser.add_argument('--image_path', action='store', default='flowers/test/1/image_06743.jpg')
parser.add_argument('--arch', action='store', dest='pre_trained_model',
                    default='vgg16', help='Choose architecture')
parser.add_argument('--save_dir', action='store', dest='save_directory',
                    default='checkpoint.pth', help='Set directory to save checkpoints')
parser.add_argument('--top_k', action='store', dest='topk', default=5,
                    help='The top K most likely class', type=int)
parser.add_argument('--category_names', action='store', dest='cat_to_name',
                    default='cat_to_name.json', help='Provide a mapping of categories to real names')
parser.add_argument('--gpu', action='store_true', default=False,
                    help='Use GPU for training, default is off')
params = parser.parse_args()

image_path = params.image_path
save_dir = params.save_directory
pre_trained_model = params.pre_trained_model
topk = params.topk
cat_to_name = params.cat_to_name
gpu_mode = params.gpu

print("predict.py params -----")
print("path: {}, save_dir: {}, pre_trained_model: {}, topk: {}, JSON file: {}, gpu_mode: {}".format(image_path, save_dir, pre_trained_model, topk, cat_to_name, gpu_mode))


with open(cat_to_name, 'r') as f:
    cat_to_name = json.load(f)
#     print(cat_to_name)

# step 1: download pre-trained model
model = getattr(models, pre_trained_model)(pretrained=True)
loaded_model = load_checkpoint(model, save_dir)

# step 2: process the test image
processed_image = process_image(image_path=image_path)

# step 3: prediction
probs, classes = predict(image_path, loaded_model, topk, gpu_mode)

print(probs)
print(classes)

# convert from class to names
names = []
for c in classes:
    names.append(cat_to_name[c])

print("the flower is most likely to be: {} with prob: {}%".format(names[0], round(probs[0]*100,4)))

import os
import numpy as np
from torchvision.models import resnet18, alexnet, googlenet, densenet161, resnet50
from torchvision.io.image import read_image
from torchvision.transforms.functional import normalize, resize
import matplotlib.pyplot as plt
import json

with open('imagenet_class_index.json') as labels_file:
    labels = json.load(labels_file)
    
models = [
    resnet18(pretrained=True).eval(),
    alexnet(pretrained=True).eval(),
    googlenet(pretrained=True).eval(),
    resnet50(pretrained=True).eval(),
    densenet161(pretrained=True).eval()
]

model_names = ['ResNet18', 'AlexNet', 'GoogleNet', 'ResNet50', 'DenseNet 161']

folder_path = 'monitor'

image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

correct_counts = [0] * len(models)
total_count = 0
imgnum = 0

# Loop over all the images in the folder
for image_file in image_files:
    # Load the image
    img = read_image(image_file)
    imgnum += 1
    print('Image',imgnum)
    # Display the image
    plt.imshow(img.permute(1,2,0))
    plt.show()

    # Normalize and resize the image
    input_tensor = normalize(resize(img, (224, 224)) / 255., [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    # Pass the image through each model to obtain a prediction
    for i, model in enumerate(models):
        out = model(input_tensor.unsqueeze(0))
        out = out.detach().cpu().numpy()

        # Get the predicted class index
        idx = np.argmax(out, axis=1)
        str_idx = str(idx[0])

        # Print the predicted class label and model name
        print('Model:', model_names[i])
        print('Image:', image_file)
        print('Class:', labels[str_idx])

        # Check if the predicted class label is correct
        # 664 is the index for 'monitor', '527' is the index for 'desktop computer' and '782' is the index for 'screen' in the JSON file
        if str_idx in ['664', '527', '782']:
            # Update the correct classification count for this model
            correct_counts[i] += 1

        # Print a dotted line after classifying the image with each model
        print('..........')
    print('---------------------------------------------------------')
    # Update the total image count
    total_count += 1

for i, model in enumerate(models):
    accuracy = (correct_counts[i] / total_count) * 100
    print(model_names[i], 'accuracy:', accuracy, '%')
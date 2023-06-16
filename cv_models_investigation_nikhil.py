#!/usr/bin/env python
# coding: utf-8

# # Loading Required Libraries and Modules

# I will firstly load required modules and libraries for processing images and applying pre-trained models for image classification.

# In[9]:


import os
import numpy as np
from torchvision.models import resnet18, alexnet, googlenet, densenet161, resnet50
from torchvision.io.image import read_image
from torchvision.transforms.functional import normalize, resize
import matplotlib.pyplot as plt
import json


# 'os' module is a built-in Python module that provides a way of interacting with the file system. In this code, it is used to set a path to a folder containing images.
# 
# 'numpy' is a popular numerical computing library for Python that provides support for large, multi-dimensional arrays and matrices. It is used in this code to manipulate the output of pre-trained models.
# 
# 'torchvision' is a library that provides access to popular datasets, model architectures, and image processing tools for PyTorch. In this code, it is used to load pre-trained models such as ResNet18, AlexNet, GoogLeNet, DenseNet161, and ResNet50.
# 
# 'read_image' is a function provided by torchvision.io.image that reads an image from a file and returns it as a PyTorch tensor.
# 
# 'normalize' and 'resize' are functions provided by 'torchvision.transforms.functional' that normalize and resize an image, respectively. In this code, they are used to preprocess the input image before passing it through a pre-trained model.
# 
# 'matplotlib.pyplot' is a plotting library for Python that is used in this code to display the input image.
# 
# 'json' is a built-in Python module that provides methods for working with JSON data. In this code, it is used to load a JSON file containing class labels for ImageNet.

# # Loading JSON file

# Now I will load the JSON file that contains the class labels for the ImageNet dataset

# In[10]:


with open('imagenet_class_index.json') as labels_file:
    labels = json.load(labels_file)


# # Pre-Trained CV Models

# A list of pre-trained models is created from the torchvision.models module and assigns them to a list called models. It also creates a list of model names to match the models in the models list. 
# 
# The 'models list' contains five pre-trained models, each with a different architecture.
# 
# The 'model_names list' contains the names of the models in the same order as the 'models' list. These names will be used to identify each model's prediction when classifying the input images.

# In[11]:


models = [
    resnet18(pretrained=True).eval(),
    alexnet(pretrained=True).eval(),
    googlenet(pretrained=True).eval(),
    resnet50(pretrained=True).eval(),
    densenet161(pretrained=True).eval()
]

model_names = ['ResNet18', 'AlexNet', 'GoogleNet', 'ResNet50', 'DenseNet 161']


# 'resnet18', 'alexnet', 'googlenet', 'resnet50', and 'densenet161' are all models provided by the torchvision.models module that have been pre-trained on the ImageNet dataset.
# 
# '.eval()' is a method used to set the model in evaluation mode. In evaluation mode, the model will not update its parameters and will only forward propagate the input to generate outputs. This is useful when using pre-trained models for inference.
# 
# 'pretrained=True' is an argument used to specify that the model should be loaded with pre-trained weights from the ImageNet dataset.

# # Path to Folder Containing the Images

# In[12]:


folder_path = 'monitor'

image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]


# I will be using the monitor folder which contains 43 images with different frames/angles of monitors.
# 
# 'os.listdir(folder_path)' method returns a list of all the files and directories present in the folder_path directory. Using a list comprehension, 'os.path.join(folder_path, f)' is used to create a list of file paths by joining the 'folder_path' and the filename 'f' for all the files in the directory.
# 
# 'os.path.isfile(os.path.join(folder_path, f))' checks if the item in the directory is a file, not a directory. This ensures that only files in the directory are included in the 'image_files' list.

# # Variable Initialization

# In[13]:


correct_counts = [0] * len(models)
total_count = 0
imgnum = 0


# 'correct_counts' is a list that stores the count of correct classifications for each model. It is initialized with zeros for each model by multiplying a list of zeros with the length of the models list.
# 
# 'total_count' is an integer variable that stores the total number of images processed. It is initialized to zero.
# 
# 'imgnum' is an integer variable that stores the current image number being processed. It is initialized to zero.
# 
# These variables are used to keep track of the number of images processed and the count of correct classifications for each model.

# # Image processing and Classification

# A loop is created to loop over all the images in a specified folder, load and display each image, normalize and resize it, and pass it through a list of pre-trained deep learning models. For each model, a prediction is obtained, the predicted class label and model name is printed, and if the predicted class label is correct is checked. Then counters are updated for correct classifications and total images processed.

# In[14]:


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
        # 664 is the index for 'monitor' in the JSON file
        if str_idx in ['664']:
            # Update the correct classification count for this model
            correct_counts[i] += 1

        # Print a dotted line after classifying the image with each model
        print('..........')
    print('---------------------------------------------------------')
    # Update the total image count
    total_count += 1


# As shown from the results above, most of the images were classified as either Monitor or Screen. Some images were also classified as Desktop computer. All models correctly classified image 29, 30, 31, 37 as monitor. Image 8, 9, 10, 11, 13, 15, 16, 18, 20, 21, 32, 34, 36 were classified by some models as monitor, some as screen and desktop_computer.
# 
# However there were totally incorrect classifications such as loudspeaker, joystick, dumbebell, projectile, iron, plane, backpack, chime, vacuum, acoustic guitar, table lamp, polaroid camera, iPod, modem, projector, printer, notebook, warplane, quill, espresso maker. None of the models gave a proper classification on image 17, 25, 26, 27, 33, 35, 39, 40, 41, 42. There are three possible reasons to why there could be incorrect predications:
# 
# Ambiguity in the image:  Sometimes, images may contain multiple objects or visual elements that could confuse the model, leading to incorrect classification. For example, image 41 has a backpack on picture which leads one of the model to classify the image as backpack instead.
# 
# Variations in image quality: Different factors such as lighting, angle, and background can affect the quality of the image and, as a result, influence the model's classification. For example, an image taken in low light conditions may be incorrectly classified due to the reduced quality of the image.
# 
# Model architecture and parameters: Different models use different architectures and parameters to classify images, which can impact their accuracy. Some models may be more suitable for certain types of images or classes, while others may perform better for different types

# # Accuracy of Each Model

# I will calculate the accuracy of each pre-trained model in classifying the images in the given folder. A loop is created over the list of models which uses the corresponding count of correct classifications obtained in the previous step to calculate the accuracy of each model. The accuracy is calculated as the ratio of correct classifications to the total number of images processed, expressed as a percentage. Finally, the accuracy of each model is printed along with its name.

# In[15]:


for i, model in enumerate(models):
    accuracy = (correct_counts[i] / total_count) * 100
    print(model_names[i], 'accuracy:', accuracy, '%')


# AlexNet and GoogleNet achieved the highest accuracy of 46.51%, while ResNet18 achieved the lowest accuracy of 34.88%. ResNet50 and DenseNet 161 both had accuracy in between, with ResNet50 achieving 44.19% and DenseNet 161 achieving 39.53%. Therefore, based on these results, AlexNet and GoogleNet seem to perform better on this specific set of images than ResNet18, ResNet50, and DenseNet 161. However, it is important to note that these accuracies may not be representative of the models' general performance and are specific to the set of images used in this evaluation.

# # Conclusion

# The code loads pre-trained image classification models (ResNet18, AlexNet, GoogleNet, ResNet50, and DenseNet 161) from the torchvision library and applies them to a set of images in a specified folder. The goal is to test the accuracy of each model in classifying images in the folder. The code uses the ImageNet dataset and class labels to provide meaningful output.
# 
# For each image in the folder, the code loads the image and resizes it to a shape compatible with the pre-trained models. It then passes the image through each model to obtain a prediction. The predicted class label and corresponding model name are printed for each image.
# 
# Finally, the code tallies the number of correct classifications for each model and reports the accuracy percentage for each model based on the total number of images processed.
# 
# The results show that the model with the highest accuracy is AlexNet with 46.5%, while the model with the lowest accuracy is ResNet18 with 34.8%. Overall, the accuracies of all models are relatively low.
# 
# The images themselves is the main factor for the low accuracy. The images are of poor quality and low resolution. The monitor images were taken at different angles, and mostly when the monitor's image was the side or rear view, the models were not able to classify properly. It may be beneficial for better monitor recognition to fine tune models using large dataset of monitor images, which contains side and rear views. Also another improvement is to rather use images with a plain background to reduce any ambiguity which could confuse the model.
# 
# In addition, it may also be helpful to use an ensemble of models, where multiple models are combined to make predictions. This can help to improve the accuracy and reliability of the predictions, especially if the models have different strengths and weaknesses. So comparing the results with more pre-trained models could show more various accuracies to analyze.
# 
# Overall, while the current results are not very accurate, they provide a starting point for further investigation and improvement. With additional optimization, it may be possible to achieve higher accuracy in the future.

# # Further Analysis

# From the image classification some of the images were also classified as screen and desktop computer which technically may also be counted as a suitable class for these images. Therefore in a separate file I ran the code but this time adding '527' which is the index for Desktop Computer and '782' which is the index for Screen. I want to see how the accuracy changes.

# ![image.png](attachment:image.png)

# The results show that DenseNet 161 has the highest accuracy of 72.09%, followed by ResNet50 and GoogleNet with 67.44% each. AlexNet has the lowest accuracy of 60.84%.
# ResNet18 has an accuracy of 48.84%, which is lower than all the other models except AlexNet. This may be because AlexNet is an older architecture and was outperformed by later models.
# Comparing the results, we can see that the deeper models (ResNet50, DenseNet 161) generally perform better than the shallower models (ResNet18, AlexNet). GoogleNet, which is a relatively complex architecture with multiple parallel branches, also performed well.

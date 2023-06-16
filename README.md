# pretrained_cv_models
Apply Pretrained Computer Vision Models and Evaluate Performance


# Aim

Learn how to apply pretrained computer vision models and investigate their  performance.


# Workflow:

1) Use 43 different frames of monitor
2) Use imagenet_class_index.json file.
3) According to the cv_pretrained_model_test.py example, write a python script for recognizing the images mentionned in point 1).
4) Execute the script. When running for the first time, there will be a pause at the beginning while the script downloads from the cloud storage to your computer neural network weights file.
5) Analyze the results by presenting the percentage of images that were correctly identified. It should be noted that among the 1000s of ImageNet classes, a few maybe suitable for the given images.
6) Repeat steps 3) – 5) with 2 or more pretrained computer vision models.

# Cat-Dog Classifier
This is an introduction to PyTorch using the standard computer vision Dogs-vs-Cats dataset.
The dataset is available with the famous Kaggle competition found at link: https://www.kaggle.com/c/dogs-vs-cats

## About the dataset
The dataset consists of the train and test sets with 25,000 labelled images in the training archive.  
The labelled images are saved with names in the format '(cat/dog)-(image number).jpg'. Hence the training data-labels can easily be inferred from the dataset itself.
The test archive, on the other hand, includes images without the labels as '(image number).jpg'.

## About the code
So far, the problem has been tackled using various techniques  
(summarized in history_of_models.xlsx, which also saves the best accuracy achieved on the validation sets)  
Mainly, 3 CNN architectures have been used - ResNet50, ResNet18 and AlexNet.   
There are separate files for each of the models, with ResNet18 (CatDogClassifier_v2.py) and AlexNet (CatDogClassifier_AlexNet.py) sharing a common training function found in trainer.py.
There is also a separate dataHandling.py to handle the dataset preparation and batch-formations, common to all the models. 

## Observed results
ResNet18 trained for 2 epochs using the SGD optimizer has thus-far achieved the best validation-set accuracy.  
ResNet50, though being a significantly larger model, offers a comparable level of accuracy to ResNet18.

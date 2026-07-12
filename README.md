# Human Activity Recognition (HAR) Project

Human Activity Recognition (HAR) is an important application of machine learning and pattern recognition that focuses on identifying human activities using sensor data. Modern smartphones and wearable devices contain sensors such as accelerometers and gyroscopes that continuously capture information about human body movement.

This project implements machine learning models (K-Nearest Neighbors and Support Vector Machine) to recognize human activities from smartphone sensor data (accelerometer and gyroscope). 
The activities classified are: Walking, Walking Upstairs, Walking Downstairs, Sitting, Standing, and Laying.

## Requirements
To install the required dependencies, run:
```bash
pip install -r requirements.txt
```

## Running the Code
Simply run `main.py` to automatically download the UCI HAR Dataset, process it, train both models, and evaluate them:
```bash
python main.py
```

The script will output the Accuracy, Precision, Recall, and F1-Score for both KNN and SVM models, and save their confusion matrices as PNG images in this directory.

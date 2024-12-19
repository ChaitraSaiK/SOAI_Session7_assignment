# MNIST Classification with Efficient CNN

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-99.42%25-success.svg)
![Parameters](https://img.shields.io/badge/Parameters-7.3K-informational)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Objective

To classify the MNIST dataset and achieve 99.4% test accuracy consistently using a model with:
- Fewer than 8,000 parameters
- Training within 15 epochs

## Approach and Results

1. Baseline Model
Goal: Design a simple CNN model with fewer than 8,000 parameters to establish a baseline.
Results:
Parameters: 7,348
Best Train Accuracy: 80.05%
Best Test Accuracy: 79.29%
Analysis:
The model is overfitting with low accuracy.
Highlights the need for additional techniques to improve performance.

2. Adding Batch Normalization
Goal: Incorporate Batch Normalization to enhance accuracy and training efficiency.
Results:
Parameters: 7,348
Best Train Accuracy: 99.66%
Best Test Accuracy: 99.32%
Analysis:
Batch Normalization significantly improves the model's efficiency.
However, the model continues to exhibit overfitting.

3. Introducing Dropout for Regularization
Goal: Reduce overfitting by applying Dropout as a regularization technique.
Results:
Parameters: 7,348
Best Train Accuracy: 99.07%
Best Test Accuracy: 99.40%
Analysis:
Dropout reduces the gap between train and test accuracy, mitigating overfitting.
Regularization ensures that some neuron outputs are nullified, making the model more robust.

4. Applying Data Augmentation
Goal: Use data augmentation to create a more complex training dataset and further improve test accuracy.
Results:
Parameters: 7,348
Best Train Accuracy: 97.85%
Best Test Accuracy: 99.42%
Analysis:
Augmentation adds variability to the training data, leading to better generalization.
Test accuracy consistently reaches 99.4% within two epochs.


## Key Insights

Batch Normalization and Dropout significantly improved the model’s performance by enhancing efficiency and reducing overfitting.
Data Augmentation played a crucial role in achieving the target accuracy by introducing complexity to the training data.
This structured approach highlights the impact of each method in improving model accuracy while adhering to parameter and epoch constraints.


## Final Model

Parameters: 7,348
Best Test Accuracy: 99.42%
Consistency: Achieved 99.4% test accuracy within 15 epochs in consecutive runs.
This structured and iterative methodology ensures the efficient use of parameters and epochs to meet the target accuracy.
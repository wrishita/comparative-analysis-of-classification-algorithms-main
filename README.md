# 🍷 Red Wine Quality Classification

A comprehensive machine learning project that predicts wine quality based on physicochemical properties using various classification algorithms.

![Wine Classification](https://img.shields.io/badge/Wine-Classification-darkred) 
![Python 3.6+](https://img.shields.io/badge/Python-3.6+-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Latest-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset Information](#dataset-information)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Algorithms Implemented](#algorithms-implemented)
6. [Performance Evaluation](#performance-evaluation)
7. [Results](#results)
8. [Conclusion](#conclusion)
9. [License](#license)  

## 📊 Project Overview

This project investigates how different physicochemical properties of red wine affect its quality rating. By analyzing attributes like acidity, alcohol content, and sulfate levels, the project builds classification models to predict whether a wine's quality is "good" or "bad" based on expert ratings. 

The analysis includes comprehensive data visualization, preprocessing, and the implementation of seven different classification algorithms with detailed performance comparisons.

## 🍇 Dataset Information

The dataset used is the [Red Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/Wine+Quality) from the UCI Machine Learning Repository.

### Features

The dataset contains the following physicochemical properties:

1. **fixed acidity** (g/dm³) – Primarily tartaric acid
2. **volatile acidity** (g/dm³) – Amount of acetic acid
3. **citric acid** (g/dm³) – Adds freshness and flavor
4. **residual sugar** (g/dm³) – Sugar remaining after fermentation
5. **chlorides** (g/dm³) – Amount of salt
6. **free sulfur dioxide** (mg/dm³) – Free SO₂ levels
7. **total sulfur dioxide** (mg/dm³) – Total amount of SO₂
8. **density** (g/cm³) – Wine density
9. **pH** – Acidity or basicity level (0-14)
10. **sulphates** (g/dm³) – Wine additive (preservative)
11. **alcohol** (%) – Percent alcohol content by volume

### Target Variable

- **quality**: Originally scored between 0-10, binarized into "good" (score > 6.5) and "bad" (score <= 6.5) for classification

## 🔧 Installation

To set up this project locally, follow these steps:

```bash
# Clone this repository
git clone https://github.com/yourusername/red-wine-quality-classification.git
cd red-wine-quality-classification

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### Dependencies

The project requires the following Python libraries:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- math
- collections

## 🚀 Usage

You can run the project by executing the Jupyter notebook:

```bash
jupyter notebook Red_Wine_Quality_Classification.ipynb
```


## 🧮 Algorithms Implemented

This project implements and compares seven classification algorithms:

1. **Logistic Regression**
   - A statistical method that estimates the probability of a binary outcome
   - Uses a logistic function to model the probability of wine being "good"

2. **K-Nearest Neighbors (K-NN)**
   - Classifies based on the majority class of K nearest training examples
   - Implementation uses distance weighting with 32 neighbors

3. **Support Vector Machine (SVM - Linear)**
   - Creates a linear decision boundary to separate wine classes
   - Maximizes the margin between classes

4. **Support Vector Machine (SVM - RBF Kernel)**
   - Applies the "kernel trick" to handle non-linear classification
   - Uses radial basis function to transform features into higher-dimensional space

5. **Naive Bayes**
   - Probabilistic classifier based on Bayes' theorem with feature independence assumption
   - Fast training and prediction times

6. **Decision Tree Classification**
   - Creates a tree-like model of decisions based on feature values
   - Highly interpretable model that splits data based on information gain

7. **Random Forest Classification**
   - Ensemble method creating multiple decision trees
   - Final prediction determined by majority voting
   - Uses 800 estimators with entropy criterion

## 📏 Performance Evaluation

The models are evaluated using multiple metrics to ensure comprehensive performance assessment:

- **Confusion Matrix** - Shows true positives, false positives, true negatives, and false negatives

![Confusion Matrix](https://i.ibb.co/mFPTT4Vw/image.png)

- **Accuracy (Training & Test)** - Proportion of correct predictions

![Train and Test Accuracy](https://i.ibb.co/h1DSSSnv/image.png)

- **Cross-Validation Score** - Average accuracy across multiple validation folds

![Cross-Validation Score](https://i.ibb.co/FkD4WJ21/image.png)
 
- **Precision** - Proportion of true positives among positive predictions

![Precision Score](https://i.ibb.co/bjFjHjcy/image.png)
 
- **Recall (Sensitivity)** - Proportion of true positives correctly identified

![Recall](https://i.ibb.co/KHw7ttW/image.png)
 
- **F1 Score** - Harmonic mean of precision and recall

![F1 Score](https://i.ibb.co/Nd14wykT/image.png)
 
- **Specificity** - Proportion of true negatives correctly identified

![Specificity](https://i.ibb.co/zW0GYqTr/image.png)
 

## 📈 Results

**The Random Forest classifier** achieved the best overall performance:

- **Test Accuracy**: 91.25%
- **F1 Score**: 0.95
- **Precision**: 0.978
- **Specificity**: 0.8065
- **Cross-Validation Score**: 0.914

![Comparision](https://i.ibb.co/TMg6Jf1x/image.png)



*Other notable performers:*
- **K-Nearest Neighbors** and **SVM (Kernel)** also performed well, with high recall and F1 scores
- **Naive Bayes** achieved the highest recall (0.959) but with lower specificity (0.4868)
- **SVM (Linear)** showed perfect precision (1.0) but the lowest specificity (0.0)

## 🏆 Conclusion

The analysis revealed that **Random Forest Classification** is the optimal choice for this red wine quality prediction task, demonstrating the most balanced performance across all metrics. The model effectively leverages the relationships between physicochemical properties and wine quality ratings to provide reliable predictions.

Key findings:
- Random Forest outperformed other algorithms in overall accuracy and F1 score
- The model showed strong generalization capability through cross-validation
- Features like alcohol content, volatile acidity, and sulphates were particularly important in predicting wine quality

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

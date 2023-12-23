# Topics

# Task | Performance | Experience

### MNIST:
**Task:** Image classification of handwritten digits (0-9) using the MNIST dataset.

**Performance:** Evaluating accuracy, precision, recall, and F1 score, with top models achieving over 99% accuracy.

**Experience:** Fundamental for learning image classification, providing hands-on experience in model building and evaluation.

### Auto Drive Bot:
**Task:** Navigate, Autonomous driving system, requiring perception, decision-making, and vehicle control.

**Performance:** Metrics include collision avoidance, rule adherence, and response to dynamic obstacles for safe and smooth navigation.

**Experience:** Complex task involving computer vision, machine learning, and robotics, offering expertise development in real-world autonomous systems.

# Note:
A computer program is said to learn from experience E with respect to some task T and some performance measure P. If its performance on T, as measured by P, improves with experience E.

# Machine Learning Approaches:
- **Supervised Learning:**
  - Training the model on a labeled dataset.
  - Input-output pairs are provided for the algorithm to learn the mapping function.

- **Unsupervised Learning:**
  - Learning patterns and relationships in unlabeled data.
  - Clustering and dimensionality reduction are common tasks.

- **Semi-Supervised Learning:**
  - Utilizing both labeled and unlabeled data for training.
  - Beneficial when labeled data is scarce.
  
- **Reinforcement Learning:**
  - Training a model to make sequences of decisions by interacting with an environment.
  - Rewards and penalties guide the learning process.

- **Active Learning:**
  - Involves selecting and labeling the most informative data points for model training.
  - The model queries the user or other information sources to improve its performance.


# Classifier Evaluation Metrics:

1. **Accuracy:**
   - **Definition:** Measures the overall correctness of the model by calculating the ratio of correctly predicted instances to the total instances.
   - **Formula:**
     $\[
     \text{Accuracy} = \frac{\text{True Positives + True Negatives}}{\text{Total Number of Predictions}}
     \]$

2. **Error Rate:**
   - **Definition:** The complement of accuracy, representing the proportion of incorrectly predicted instances.
   - **Formula:**
     $\[
     \text{Error Rate} = \frac{\text{False Positives + False Negatives}}{\text{Total Number of Predictions}}
     \]$

3. **Sensitivity (True Positive Rate or Recall):**
   - **Definition:** Measures the ability of the model to correctly identify positive instances out of the total actual positive instances.
   - **Formula:**
     $\[
     \text{Sensitivity} = \frac{\text{True Positives}}{\text{True Positives + False Negatives}}
     \]$

4. **Specificity (True Negative Rate):**
   - **Definition:** Measures the ability of the model to correctly identify negative instances out of the total actual negative instances.
   - **Formula:**
     $\[
     \text{Specificity} = \frac{\text{True Negatives}}{\text{True Negatives + False Positives}}
     \]$

5. **Precision (Positive Predictive Value):**
   - **Definition:** Measures the accuracy of positive predictions. It is the ratio of true positives to the total predicted positives.
   - **Formula:**
     $\[
     \text{Precision} = \frac{\text{True Positives}}{\text{True Positives + False Positives}}
     \]$

6. **Recall (Sensitivity or True Positive Rate):**
   - **Definition:** Measures the ability of the model to correctly identify all relevant instances. It is the ratio of true positives to the total actual positives.
   - **Formula:**
     $\[
     \text{Recall} = \frac{\text{True Positives}}{\text{True Positives + False Negatives}}
     \]$

7. **F-measure (F1 Score):**
   - **Definition:** The harmonic mean of precision and recall, providing a balanced metric.
   - **Formula:**
     $\[
     \text{F-measure} = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision + Recall}}
     \]$

### Validation Methods:

1. **Holdout Method:**
   - **Definition:** The dataset is split into two subsets: a training set and a testing set. The model is trained on the training set and evaluated on the testing set.
   - **Formulae:** Not applicable (this is a method, not a metric).

2. **Cross-Validation:**
   - **Definition:** The dataset is divided into k subsets (folds), and the model is trained and evaluated k times. Each time, a different fold is used as the test set, and the remaining folds are used for training.
   - **Formulae:** Not applicable (this is a method, not a metric).

These metrics and methods provide insights into different aspects of classifier performance and are crucial for assessing the quality of a machine learning model. The choice of metrics and validation methods depends on the characteristics of the dataset and the goals of the modeling task.

# K-fold Cross-Validation:
K-fold cross-validation is a method used to assess a machine learning model's performance. The dataset is divided into k subsets, and the model is trained and evaluated k times. In each iteration, one subset serves as the test set, while the remaining subsets are used for training. This process provides a more robust performance estimate, helping to account for dataset variability and reduce the risk of overfitting or underfitting. Common choices for k are 5 or 10.

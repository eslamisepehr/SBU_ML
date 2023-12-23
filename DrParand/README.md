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


### Classifier Evaluation Metrics:

1. **Accuracy:**
   - **Definition:** Accuracy measures the overall correctness of the model by calculating the ratio of correctly predicted instances to the total instances.
   - **Formula:**
     $$\[
     \text{Accuracy} = \frac{\text{Number of Correct Predictions}}{\text{Total Number of Predictions}}
     \]$$

2. **Error Rate:**
   - **Definition:** The Error Rate is the complement of accuracy and represents the proportion of incorrectly predicted instances.
   - **Formula:**
     $$\[
     \text{Error Rate} = \frac{\text{Number of Incorrect Predictions}}{\text{Total Number of Predictions}}
     \]$$

3. **Sensitivity (True Positive Rate or Recall):**
   - **Definition:** Sensitivity measures the ability of the model to correctly identify positive instances (true positives) out of the total actual positive instances.
   - **Formula:**
     $$\[
     \text{Sensitivity} = \frac{\text{True Positives}}{\text{True Positives + False Negatives}}
     \]$$

4. **Specificity (True Negative Rate):**
   - **Definition:** Specificity measures the ability of the model to correctly identify negative instances (true negatives) out of the total actual negative instances.
   - **Formula:**
     $$\[
     \text{Specificity} = \frac{\text{True Negatives}}{\text{True Negatives + False Positives}}
     \]$$

**Note:** Terms in the formulas:
- **True Positives (TP):** The number of instances correctly predicted as positive.
- **True Negatives (TN):** The number of instances correctly predicted as negative.
- **False Positives (FP):** The number of instances incorrectly predicted as positive.
- **False Negatives (FN):** The number of instances incorrectly predicted as negative.

These metrics provide a comprehensive view of a classifier's performance, addressing aspects of correctness, error, and the ability to correctly identify positive and negative instances. It's important to consider the specific characteristics of the problem and the desired outcomes when choosing which metrics to prioritize.

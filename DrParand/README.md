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

# Support Vector Machine

# Decision Tree:

### Definition:
A **Decision Tree** is a supervised machine learning algorithm used for classification and regression. It recursively partitions data into subsets based on attributes, forming a tree-like structure.

### Method:
- **Divide and Conquer:**
  - Divides the dataset into subsets based on attribute values.
  - Recursively applies the process, creating a tree structure.
  - Decision-making involves traversing from root to leaf.

### Material of the Tree:
- **Root:**
  - Represents the initial dataset and the first splitting attribute.
- **Branch:**
  - Represents a decision or a test on an attribute.
- **Intermediate Node:**
  - Represents a data subset and a decision point for further splits.
- **Leaf:**
  - Represents the final outcome or decision.

### Application in Industry:
- **Customer Relationship Management:**
  - Predicting customer churn.
- **Credit Scoring:**
  - Evaluating credit risk.
- **Medical Diagnosis:**
  - Assisting in medical decision-making.
- **Fraud Detection:**
  - Identifying patterns in financial transactions.

### ID3 (Iterative Dichotomiser 3):

#### Definition:
- **ID3** is a top-down, recursive, and greedy decision tree algorithm designed for classification tasks.

#### Method:
- **Top Down:**
  - Begins with the entire dataset, recursively selects the best attribute.
- **Greedy:**
  - Chooses the attribute maximizing information gain.
- **Recursive:**
  - Repeats for each subset until a stopping criterion.

### C4.5:

#### Definition:
- **C4.5** is a top-down, recursive, and greedy decision tree algorithm handling both categorical and numerical data.

#### Method:
- **Top Down and Recursive:**
  - Employs a top-down and recursive approach for tree construction.
- **Greedy:**
  - Uses a greedy strategy based on information gain or gain ratio.

### CART (Classification and Regression Trees):

#### Definition:
- **CART** is a decision tree algorithm supporting classification and regression, using binary splits and Gini impurity for classification.

#### Method:
- **Top Down and Recursive:**
  - Similar to ID3 and C4.5, using a top-down and recursive approach.
- **Greedy:**
  - Employs a greedy strategy to find the best binary split based on Gini impurity.

These decision tree algorithms are fundamental in machine learning, providing transparent and interpretable models for various applications in different industries.


# Entropy in Decision Trees:

**Entropy** is a measure of uncertainty or disorder in a dataset. In the context of binary classification, the entropy of a set $\( S \)$ is calculated using the formula:

$$\[ \text{Entropy}(S) = - p \cdot \log_2(p) - (1 - p) \cdot \log_2(1 - p) \]$$

where:
- $\( p \)$ is the proportion of positive instances in the set \( S \).
- $\( (1 - p) \)$ is the proportion of negative instances in the set \( S \).
- $\( \log_2 \)$ is the logarithm base 2.

### Real Example:

Let's consider a dataset with 100 emails, where 40 are spam (positive) and 60 are not spam (negative).

1. **Calculate the Proportions:**
   - $\( p \)$ (proportion of spam) = 40/100 = 0.4
   - $\( 1 - p \)$ (proportion of not spam) = 1 - 0.4 = 0.6

2. **Calculate Entropy:**
   $$\[ \text{Entropy}(S) = - (0.4 \cdot \log_2(0.4)) - (0.6 \cdot \log_2(0.6)) \]$$

3. **Interpretation:**
   - If the resulting entropy is close to 0, the dataset is pure, indicating low uncertainty.
   - If the entropy is close to 1, the dataset is evenly split, indicating high uncertainty.

This entropy calculation guides decision tree algorithms in choosing the attribute that minimizes entropy, contributing to the effectiveness of the tree in making informed splits.


# Deep Learning:
Deep Learning is a subset of machine learning that involves the use of artificial neural networks to model and solve complex problems. It is characterized by the depth of these neural networks, comprising multiple layers (deep architectures). Deep learning algorithms learn hierarchical representations of data, automatically extracting features at different levels, enabling them to excel in tasks such as image and speech recognition, natural language processing, and more. Deep learning has gained prominence due to its ability to automatically learn and adapt to intricate patterns and representations in large datasets, making it particularly powerful for tasks with high-dimensional and unstructured data.

# High-Level View of Neural Networks (NN):

Neural networks are a class of machine learning models inspired by the structure and functioning of the human brain. They consist of interconnected nodes, called neurons, organized into layers. Neural networks can be used for a variety of tasks, including classification, regression, pattern recognition, and more.

**Components of Neural Networks:**

1. **Input Layer:**
   - Neurons in this layer receive the input data.

2. **Hidden Layers:**
   - Layers between the input and output layers.
   - Neurons in these layers process and transform the input data.

3. **Output Layer:**
   - Produces the final output or prediction.

4. **Weights and Biases:**
   - Parameters that the network learns during training to make accurate predictions.
   - Weights determine the strength of connections between neurons.
   - Biases allow neurons to activate even when input is 0.

5. **Activation Function:**
   - Introduces non-linearity to the model.
   - Common activation functions include ReLU (Rectified Linear Unit) and Sigmoid.

6. **Loss Function:**
   - Measures the difference between predicted and actual outputs.
   - The goal during training is to minimize this loss.

7. **Optimization Algorithm:**
   - Adjusts weights and biases to minimize the loss function.
   - Examples include Stochastic Gradient Descent (SGD) and Adam.

### Feedforward:

**Definition:**
- **Feedforward** refers to the flow of data through a neural network from the input layer to the output layer without any feedback connections or loops.
  
### Activation Function:

**Definition:**
- **Activation Function** introduces non-linearity to the output of a neuron in a neural network. It allows the network to learn complex patterns and relationships.
  
**Example Activation Functions:**
- **ReLU (Rectified Linear Unit):** $\(f(x) = \max(0, x)\)$
- **Sigmoid:** $\(f(x) = \frac{1}{1 + e^{-x}}\)$
- **TanH (Hyperbolic Tangent):** $\(f(x) = \frac{e^{2x} - 1}{e^{2x} + 1}\)$

### Loss Function:

**Definition:**
- **Loss Function** measures the difference between the predicted output and the true target values. It quantifies how well the model is performing.
  
**Examples of Loss Functions:**
- **Mean Squared Error (MSE):** $$\(L(y, \hat{y}) = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2\)$$
- **Cross-Entropy Loss (Log Loss):** $$\(L(y, \hat{y}) = -\sum_{i=1}^{n} y_i \log(\hat{y}_i)\)$$ (for classification tasks)

### Back Propagation:

**Definition:**
- **Back Propagation** is an optimization algorithm used to train neural networks. It involves computing gradients of the loss function with respect to the model's parameters and updating the parameters in the opposite direction of the gradient to minimize the loss.

### Feed:

**Definition:**
- In the context of neural networks, **Feed** is a part of the training process where input data is fed into the network for forward propagation, resulting in predictions or output.

### Stochastic Gradient Descent (SGD):

**Definition:**
- **Stochastic Gradient Descent (SGD)** is an optimization algorithm used to minimize the loss function during training. It updates the model's parameters by taking small steps in the direction of the negative gradient, making it suitable for large datasets.

**Steps in SGD:**
1. **Randomly Sample Batch:** Select a small random batch of data.
2. **Forward Propagation:** Compute predictions using the current model.
3. **Compute Loss:** Calculate the loss between predictions and true values.
4. **Backward Propagation:** Compute gradients of the loss with respect to parameters.
5. **Update Parameters:** Adjust model parameters in the opposite direction of the gradient.

In summary, these terms are fundamental concepts in the training and operation of neural networks, contributing to their ability to learn and make predictions on various tasks.


## Perceptron in Deep Learning

The perceptron is a fundamental building block of neural networks and serves as a simple model for a biological neuron. In deep learning, perceptrons are typically used as the basic unit in the design of artificial neural networks.

### Mathematical Representation:

The output (y) of a perceptron is calculated as follows:

$$\[ y = \text{activation}\left(\sum_{i=1}^{n} (x_i \cdot w_i) + b\right) \]$$

Where:
- $\(x_i\)$ is the $\(i\)$-th input,
- $\(w_i\)$ is the weight associated with the \(i\)-th input,
- $\(b\)$ is the bias term,
- $\(\sum_{i=1}^{n} (x_i \cdot w_i) + b\)$ is the weighted sum of inputs plus bias,
- $\(\text{activation}(\cdot)\)$ is an activation function (commonly a step function, sigmoid, or hyperbolic tangent).

The bias term \(b\) allows the perceptron to produce an output even when all inputs are zero. It essentially acts as a threshold for activation.

### Training the Perceptron:

The weights $(\(w_1, w_2, \ldots, w_n\))$ and the bias $(\(b\))$ are adjusted during the training process to learn from the input data. The perceptron is trained using a learning algorithm, such as the perceptron learning algorithm or stochastic gradient descent, to minimize the error in its predictions.

### Limitations of a Single Perceptron:

A single perceptron has limitations and can only learn linearly separable functions. However, by combining multiple perceptrons in layers and using non-linear activation functions, more complex functions can be learned, forming the basis of neural networks.

### Example of a Step Activation Function:

A common choice for the activation function is a step function, which outputs 1 if the weighted sum plus bias is greater than or equal to zero, and 0 otherwise:

$$\[ \text{activation}(z) = \begin{cases} 1, & \text{if } z \geq 0 \\ 0, & \text{if } z < 0 \end{cases} \]$$

In practice, other activation functions like sigmoid, hyperbolic tangent, or rectified linear unit (ReLU) are often used.

The perceptron model is foundational to the development of artificial neural networks and deep learning architectures.


## Hyperplane in a Perceptron

In the context of a perceptron and binary classification, a hyperplane is a key concept that defines the decision boundary used to separate instances of different classes. A hyperplane is a flat affine subspace of one dimension less than the ambient space it divides. For a perceptron, which is a linear classifier, the hyperplane represents the boundary between instances of one class and instances of the other class.

### Hyperplane in a Perceptron:

In a perceptron, the hyperplane is defined by the weights $(\(w_1, w_2, \ldots, w_n\))$ and the bias $(\(b\))$. The decision rule is based on whether a data point lies above or below this hyperplane.

For a perceptron with two features (\(x_1, x_2\)), the hyperplane equation is given by:

$$\[ w_1 \cdot x_1 + w_2 \cdot x_2 + b = 0 \]$$

This equation represents a line in a two-dimensional space, and it is a hyperplane because it separates the space into two regions. If a data point $\((x_1, x_2)\)$ lies on one side of the hyperplane, it is classified as belonging to one class; if it lies on the other side, it is classified as belonging to the other class.

### Decision Rule:

The decision rule for a perceptron with a hyperplane is determined by the sign of the expression $\(w_1 \cdot x_1 + w_2 \cdot x_2 + b\)$. If the expression is positive, the point is on one side of the hyperplane and classified as one class; if negative, it is on the other side and classified as the other class.

### Learning the Hyperplane:

During the training phase, the perceptron adjusts its weights and bias to learn the optimal hyperplane that minimizes classification errors. This is typically achieved through an iterative process using a learning algorithm such as the perceptron learning algorithm or stochastic gradient descent.

### Extension to Higher Dimensions:

In higher dimensions with more features, the hyperplane is defined by the general equation:

$$\[ w_1 \cdot x_1 + w_2 \cdot x_2 + \ldots + w_n \cdot x_n + b = 0 \]$$

The decision boundary is still a hyperplane, but it's now a subspace in an $\(n\)$-dimensional space that separates instances of different classes.

Understanding the hyperplane in a perceptron is crucial for grasping how linear classifiers make decisions based on the relationships between features. It's important to note that while a single perceptron is limited to learning linearly separable functions, more complex functions can be learned by stacking multiple perceptrons or using more sophisticated architectures like multi-layer perceptrons (neural networks).

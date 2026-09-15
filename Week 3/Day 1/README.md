# Week 3 - Day 1: Deep Learning Fundamentals

## 1. What is Deep Learning?
Deep Learning is a subfield of Machine Learning based on Artificial Neural Networks (ANNs) with multiple layers (hence "deep"). Unlike traditional machine learning, which often requires manual feature engineering, deep learning algorithms automatically learn hierarchical representations of data.

---

## 2. Neural Networks & Structure

### Artificial Neurons (Perceptrons)
An artificial neuron is the basic building block of a neural network. It:
1. Takes multiple inputs ($x_1, x_2, \dots, x_n$).
2. Multiplies each input by a corresponding weight ($w_1, w_2, \dots, w_n$).
3. Adds a bias ($b$).
4. Passes the weighted sum through an **activation function** ($f$) to produce the output ($y$).

$$y = f\left(\sum_{i=1}^{n} w_i x_i + b\right)$$

### Layers
*   **Input Layer:** Receives the raw input features. No computation happens here.
*   **Hidden Layers:** Intermediate layers between input and output. They extract increasingly abstract features.
*   **Output Layer:** Produces the final prediction (e.g., a single continuous value for regression, or class probabilities for classification).

### Weights and Bias
*   **Weights ($w$):** Control the strength of the connection between neurons. They determine how much influence an input has on the output.
*   **Bias ($b$):** An extra parameter added to the weighted sum, allowing the activation function to shift left or right (independent of the inputs).

---

## 3. Activation Functions
Activation functions introduce non-linearity into the network, allowing it to learn complex, non-linear patterns.

*   **Sigmoid:** Maps inputs to a range between $0$ and $1$. Often used for binary classification.
    $$\sigma(z) = \frac{1}{1 + e^{-z}}$$
*   **ReLU (Rectified Linear Unit):** Returns $0$ if the input is negative, and the input itself if positive. It is the most widely used activation function in hidden layers due to its computational efficiency and reduction of the vanishing gradient problem.
    $$f(z) = \max(0, z)$$
*   **Softmax:** Converts a vector of raw scores (logits) into a probability distribution that sums to $1$. Commonly used in the output layer for multi-class classification.
    $$\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j} e^{z_j}}$$

---

## 4. Training Concepts

### Forward Propagation
The process where input data passes forward through the network's layers to generate a prediction.

### Loss Functions
Quantify how far the network's prediction is from the actual target.
*   **Mean Squared Error (MSE):** Used for regression.
*   **Binary Cross-Entropy:** Used for binary classification.
*   **Categorical Cross-Entropy:** Used for multi-class classification.

### Backpropagation & Gradient Descent
*   **Backpropagation:** The algorithm used to calculate the gradient of the loss function with respect to each weight and bias by applying the chain rule of calculus from the output layer back to the input layer.
*   **Gradient Descent:** An optimization algorithm used to update the weights and biases in the direction that minimizes the loss function.
    $$W \leftarrow W - \eta \cdot \frac{\partial L}{\partial W}$$
    where $\eta$ is the **learning rate**.

### Key Hyperparameters
*   **Epoch:** One complete pass of the entire training dataset through the neural network.
*   **Batch Size:** The number of training examples processed in a single forward/backward pass.
*   **Learning Rate ($\eta$):** A scalar determining the step size taken toward the minimum during optimization.

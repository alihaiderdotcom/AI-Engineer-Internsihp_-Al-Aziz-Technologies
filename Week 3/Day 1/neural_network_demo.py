import numpy as np

# 1. Activation Functions
def sigmoid(z):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-z))

def relu(z):
    """Rectified Linear Unit (ReLU) activation function."""
    return np.maximum(0, z)

def softmax(z):
    """Softmax activation function for a 1D or 2D array (logits)."""
    shift_z = z - np.max(z, axis=-1, keepdims=True)  # Stability trick to prevent overflow
    exps = np.exp(shift_z)
    return exps / np.sum(exps, axis=-1, keepdims=True)


# 2. Artificial Neuron Implementation
class Neuron:
    def __init__(self, weights, bias, activation_func):
        self.weights = np.array(weights)
        self.bias = bias
        self.activation_func = activation_func

    def forward(self, inputs):
        # Calculate weighted sum: z = w1*x1 + w2*x2 + ... + b
        z = np.dot(inputs, self.weights) + self.bias
        # Apply activation function
        return self.activation_func(z)


# 3. Simple Neural Network (2 Inputs -> 1 Hidden Layer with 2 Neurons -> 1 Output Neuron)
class SimpleNeuralNetwork:
    def __init__(self):
        # Initialize weights and biases manually for demonstration
        # Hidden Layer (2 neurons)
        self.w11, self.w12 = np.array([0.5, -0.2]), np.array([0.1, 0.8])
        self.b1, self.b2 = 0.1, -0.5
        
        # Output Layer (1 neuron)
        self.w_out = np.array([0.7, -0.4])
        self.b_out = 0.2

    def forward(self, x):
        # Hidden layer computations
        h1_input = np.dot(x, self.w11) + self.b1
        h1_output = relu(h1_input)

        h2_input = np.dot(x, self.w12) + self.b2
        h2_output = relu(h2_input)

        # Output layer computations
        hidden_outputs = np.array([h1_output, h2_output])
        out_input = np.dot(hidden_outputs, self.w_out) + self.b_out
        prediction = sigmoid(out_input)

        return prediction


def main():
    print("--- Testing Activation Functions ---")
    test_val = np.array([-2.0, 0.0, 3.0])
    print(f"Input: {test_val}")
    print(f"Sigmoid: {sigmoid(test_val)}")
    print(f"ReLU: {relu(test_val)}")
    print(f"Softmax: {softmax(test_val)}\n")

    print("--- Testing Single Neuron ---")
    # Neuron with 3 inputs, weights [0.5, 1.0, -1.5], bias 0.5, using ReLU
    neuron = Neuron(weights=[0.5, 1.0, -1.5], bias=0.5, activation_func=relu)
    inputs = np.array([1.0, 2.0, 1.0])
    output = neuron.forward(inputs)
    print(f"Inputs: {inputs}")
    print(f"Neuron Output: {output}\n")

    print("--- Testing Simple Neural Network Forward Pass ---")
    nn = SimpleNeuralNetwork()
    sample_input = np.array([0.5, 1.5])
    prediction = nn.forward(sample_input)
    print(f"Network Input: {sample_input}")
    print(f"Network Prediction (Probability): {prediction:.4f}")


if __name__ == "__main__":
    main()

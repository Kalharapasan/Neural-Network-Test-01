import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras

# ============================================================================
# PART 1: NEURAL NETWORK FROM SCRATCH
# ============================================================================

class NeuralNetwork:
    def __init__(self, layers):
        """
        Initialize neural network with specified layer sizes
        layers: list of integers representing number of neurons in each layer
        """
        self.layers = layers
        self.num_layers = len(layers)
        
        # Initialize weights and biases randomly
        self.weights = []
        self.biases = []
        
        for i in range(1, self.num_layers):
            # Xavier initialization
            w = np.random.randn(layers[i], layers[i-1]) * np.sqrt(2.0 / layers[i-1])
            b = np.zeros((layers[i], 1))
            self.weights.append(w)
            self.biases.append(b)
    
    def sigmoid(self, z):
        """Sigmoid activation function"""
        # Clip z to prevent overflow
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))
    
    def sigmoid_derivative(self, z):
        """Derivative of sigmoid function"""
        s = self.sigmoid(z)
        return s * (1 - s)
    
    def relu(self, z):
        """ReLU activation function"""
        return np.maximum(0, z)
    
    def relu_derivative(self, z):
        """Derivative of ReLU function"""
        return (z > 0).astype(float)
    
    def forward_pass(self, x):
        """Forward propagation"""
        activation = x.reshape(-1, 1)
        activations = [activation]
        zs = []
        
        for i in range(len(self.weights)):
            z = np.dot(self.weights[i], activation) + self.biases[i]
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)
        
        return activations, zs
    
    def backward_pass(self, x, y, activations, zs):
        """Backward propagation"""
        m = x.shape[0] if len(x.shape) > 1 else 1
        
        # Initialize gradients
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        
        # Output layer error
        delta = (activations[-1] - y.reshape(-1, 1)) * self.sigmoid_derivative(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].T)
        
        # Backpropagate error
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self.sigmoid_derivative(z)
            delta = np.dot(self.weights[-l+1].T, delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].T)
        
        return nabla_w, nabla_b
    
    def train(self, X, y, epochs=1000, learning_rate=0.1, batch_size=32):
        """Train the neural network"""
        n_samples = X.shape[0]
        costs = []
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            total_cost = 0
            
            # Mini-batch gradient descent
            for i in range(0, n_samples, batch_size):
                batch_X = X_shuffled[i:i+batch_size]
                batch_y = y_shuffled[i:i+batch_size]
                
                batch_nabla_w = [np.zeros(w.shape) for w in self.weights]
                batch_nabla_b = [np.zeros(b.shape) for b in self.biases]
                
                batch_cost = 0
                
                for x_sample, y_sample in zip(batch_X, batch_y):
                    # Forward pass
                    activations, zs = self.forward_pass(x_sample)
                    
                    # Calculate cost
                    cost = 0.5 * np.sum((activations[-1] - y_sample.reshape(-1, 1))**2)
                    batch_cost += cost
                    
                    # Backward pass
                    nabla_w, nabla_b = self.backward_pass(x_sample, y_sample, activations, zs)
                    
                    # Accumulate gradients
                    for j in range(len(batch_nabla_w)):
                        batch_nabla_w[j] += nabla_w[j]
                        batch_nabla_b[j] += nabla_b[j]
                
                # Update weights and biases
                batch_len = len(batch_X)
                for j in range(len(self.weights)):
                    self.weights[j] -= learning_rate * batch_nabla_w[j] / batch_len
                    self.biases[j] -= learning_rate * batch_nabla_b[j] / batch_len
                
                total_cost += batch_cost
            
            # Store average cost for this epoch
            avg_cost = total_cost / n_samples
            costs.append(avg_cost)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Cost: {avg_cost:.6f}")
        
        return costs
    
    def predict(self, X):
        """Make predictions"""
        predictions = []
        for x in X:
            activations, _ = self.forward_pass(x)
            predictions.append(activations[-1].flatten())
        return np.array(predictions)

# ============================================================================
# PART 2: EXAMPLE USAGE AND COMPARISON
# ============================================================================

def create_sample_data():
    """Create sample dataset for testing"""
    # Create binary classification dataset
    X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, 
                             n_informative=2, random_state=42, n_clusters_per_class=1)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normalize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, scaler

def test_custom_neural_network():
    """Test our custom neural network implementation"""
    print("=" * 60)
    print("TESTING CUSTOM NEURAL NETWORK")
    print("=" * 60)
    
    # Create sample data
    X_train, X_test, y_train, y_test, scaler = create_sample_data()
    
    # Create and train neural network
    nn = NeuralNetwork([2, 4, 4, 1])  # 2 inputs, 2 hidden layers with 4 neurons each, 1 output
    
    print("Training custom neural network...")
    costs = nn.train(X_train, y_train, epochs=500, learning_rate=0.1, batch_size=32)
    
    # Make predictions
    train_predictions = nn.predict(X_train)
    test_predictions = nn.predict(X_test)
    
    # Calculate accuracy
    train_acc = np.mean((train_predictions > 0.5).astype(int).flatten() == y_train)
    test_acc = np.mean((test_predictions > 0.5).astype(int).flatten() == y_test)
    
    print(f"Custom NN - Training Accuracy: {train_acc:.4f}")
    print(f"Custom NN - Test Accuracy: {test_acc:.4f}")
    
    return costs

def test_tensorflow_network():
    """Test TensorFlow/Keras neural network"""
    print("\n" + "=" * 60)
    print("TESTING TENSORFLOW/KERAS NEURAL NETWORK")
    print("=" * 60)
    
    # Create sample data
    X_train, X_test, y_train, y_test, scaler = create_sample_data()
    
    # Create TensorFlow model
    model = keras.Sequential([
        keras.layers.Dense(4, activation='relu', input_shape=(2,)),
        keras.layers.Dense(4, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    
    # Compile model
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    print("Training TensorFlow neural network...")
    # Train model
    history = model.fit(X_train, y_train,
                       epochs=100,
                       batch_size=32,
                       validation_split=0.2,
                       verbose=0)
    
    # Evaluate model
    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"TensorFlow NN - Training Accuracy: {train_acc:.4f}")
    print(f"TensorFlow NN - Test Accuracy: {test_acc:.4f}")
    
    return history

def visualize_results(costs, history):
    """Visualize training results"""
    plt.figure(figsize=(12, 4))
    
    # Plot custom network cost
    plt.subplot(1, 2, 1)
    plt.plot(costs)
    plt.title('Custom Neural Network - Training Cost')
    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    plt.grid(True)
    
    # Plot TensorFlow network accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('TensorFlow Neural Network - Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def iris_classification_example():
    """Example with Iris dataset using TensorFlow"""
    print("\n" + "=" * 60)
    print("IRIS CLASSIFICATION EXAMPLE")
    print("=" * 60)
    
    # Load Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Split and normalize data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Create model for multi-class classification
    model = keras.Sequential([
        keras.layers.Dense(8, activation='relu', input_shape=(4,)),
        keras.layers.Dense(6, activation='relu'),
        keras.layers.Dense(3, activation='softmax')  # 3 classes
    ])
    
    # Compile model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Train model
    print("Training Iris classification model...")
    model.fit(X_train, y_train, epochs=100, batch_size=16, verbose=0)
    
    # Evaluate
    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"Iris Classification - Training Accuracy: {train_acc:.4f}")
    print(f"Iris Classification - Test Accuracy: {test_acc:.4f}")
    
    # Make predictions
    predictions = model.predict(X_test)
    predicted_classes = np.argmax(predictions, axis=1)
    
    print("\nSample predictions:")
    for i in range(5):
        print(f"True: {iris.target_names[y_test[i]]}, "
              f"Predicted: {iris.target_names[predicted_classes[i]]}, "
              f"Confidence: {np.max(predictions[i]):.3f}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("Neural Network Implementation in Python")
    print("=" * 60)
    
    try:
        # Test custom implementation
        costs = test_custom_neural_network()
        
        # Test TensorFlow implementation
        history = test_tensorflow_network()
        
        # Visualize results
        print("\nGenerating visualizations...")
        visualize_results(costs, history)
        
        # Iris classification example
        iris_classification_example()
        
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print("This program demonstrates:")
        print("1. Neural network implementation from scratch using NumPy")
        print("2. Comparison with TensorFlow/Keras implementation")
        print("3. Binary classification example")
        print("4. Multi-class classification with Iris dataset")
        print("5. Training visualization and performance metrics")
        
    except ImportError as e:
        print(f"Missing required library: {e}")
        print("Please install required packages:")
        print("pip install numpy matplotlib scikit-learn tensorflow")
    
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Make sure all required libraries are installed and try again.")

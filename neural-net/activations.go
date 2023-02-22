package neuralnet

import "math"

// Sigmoid: takes a real-valued input and squashes it to range between 0 and 1. σ(x) = 1 / (1 + exp(−x))

func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func sigmoidPrime(x float64) float64 {
	return sigmoid(x) * (1.0 - sigmoid(x))
}

// tanh: takes a real-valued input and squashes it to the range [-1, 1]. tanh(x) = 2σ(2x) − 1
// ReLU: ReLU stands for Rectified Linear Unit. It takes a real-valued input and thresholds it at zero (replaces negative values with zero). f(x) = max(0, x)

package neuralnet

import (
	"errors"
	"math"
	"math/rand"
	"time"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

type NeuralNet struct {
	config        NeuralNetConfig
	hiddenWeights *mat.Dense
	hiddenBias    *mat.Dense
	outputWeights *mat.Dense
	outputBias    *mat.Dense
}

type NeuralNetConfig struct {
	CountInputNeurons  int
	CountOutputNeurons int
	CountHiddenNeurons int // TODO: change this to layer sizes config object to support multiple layers
	CountEpochs        int
	LearningRate       float64
}

func New(config NeuralNetConfig) *NeuralNet {
	return &NeuralNet{config: config}
}

func (nn *NeuralNet) Train(inputVars, dependentVars *mat.Dense) error {
	nn.hiddenWeights = mat.NewDense(nn.config.CountInputNeurons, nn.config.CountHiddenNeurons, nil)
	nn.hiddenBias = mat.NewDense(1, nn.config.CountHiddenNeurons, nil)
	// any other weight-layer matrices need to have rows equal to cols of previous matrix and rows equal to cols of next matrix
	nn.outputWeights = mat.NewDense(nn.config.CountHiddenNeurons, nn.config.CountOutputNeurons, nil)
	nn.outputBias = mat.NewDense(1, nn.config.CountOutputNeurons, nil)

	randPopulateMatrices(nn.hiddenWeights, nn.hiddenBias, nn.outputWeights, nn.outputBias)

	output := new(mat.Dense)

	err := nn.backpropagate(inputVars, dependentVars, nn.hiddenWeights, nn.hiddenBias, nn.outputWeights, nn.outputBias, output)
	if err != nil {
		return err
	}
	return nil
}

func (nn *NeuralNet) Predict(x *mat.Dense) (*mat.Dense, error) {
	if nn.hiddenWeights == nil || nn.outputWeights == nil {
		return nil, errors.New("the supplied weights are empty")
	}
	if nn.hiddenBias == nil || nn.outputBias == nil {
		return nil, errors.New("the supplied biases are empty")
	}

	output := new(mat.Dense)
	feedForward(x, nn.hiddenWeights, nn.hiddenBias, nn.outputWeights, nn.outputBias, output)
	return output, nil
}

func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func sigmoidPrime(x float64) float64 {
	return sigmoid(x) * (1.0 - sigmoid(x))
}

func (nn *NeuralNet) backpropagate(inputVars, dependentVars, hiddenWeights, biasHidden, weightsOutput, biasOutput, output *mat.Dense) error {
	for i := 0; i < nn.config.CountEpochs; i++ {

		// feed forward
		hiddenLayerActivations := feedForward(inputVars, hiddenWeights, biasHidden, weightsOutput, biasOutput, output)

		// backpropogate
		networkError := new(mat.Dense)
		networkError.Sub(dependentVars, output)

		slopeOutputLayer := new(mat.Dense)
		applySigmoidPrime := func(_, _ int, v float64) float64 { return sigmoidPrime(v) }
		slopeOutputLayer.Apply(applySigmoidPrime, output)
		slopeHiddenLayer := new(mat.Dense)
		slopeHiddenLayer.Apply(applySigmoidPrime, hiddenLayerActivations)

		derivativeOutput := new(mat.Dense)
		derivativeOutput.MulElem(networkError, slopeOutputLayer)
		errorAtHiddenLayer := new(mat.Dense)
		errorAtHiddenLayer.Mul(derivativeOutput, weightsOutput.T())

		derivativeHiddenLayer := new(mat.Dense)
		derivativeHiddenLayer.MulElem(errorAtHiddenLayer, slopeHiddenLayer)

		// adjust params
		nn.adjustParams(weightsOutput, hiddenLayerActivations, derivativeOutput)
		err := nn.adjustBias(biasOutput, derivativeOutput)
		if err != nil {
			return err
		}
		nn.adjustParams(hiddenWeights, inputVars, derivativeHiddenLayer)
		hiddenErr := nn.adjustBias(biasHidden, derivativeHiddenLayer)
		if hiddenErr != nil {
			return hiddenErr
		}
	}
	return nil
}

func feedForward(inputVars, hiddenWeights, biasHidden, weightsOutput, biasOutput, output *mat.Dense) *mat.Dense {
	summingJunction := new(mat.Dense)
	summingJunction.Mul(inputVars, hiddenWeights)
	summingJunction.Apply(func(i, j int, v float64) float64 { return v + biasHidden.At(0, j) }, summingJunction)

	nonLinearActivations := new(mat.Dense)
	applySigmoid := func(_, _ int, v float64) float64 { return sigmoid(v) }
	nonLinearActivations.Apply(applySigmoid, summingJunction)

	outputLayerInput := new(mat.Dense)
	outputLayerInput.Mul(nonLinearActivations, weightsOutput)
	outputLayerInput.Apply(func(i, j int, v float64) float64 { return v + biasOutput.At(0, j) }, outputLayerInput)
	output.Apply(applySigmoid, outputLayerInput)
	return nonLinearActivations
}

func (nn *NeuralNet) adjustParams(original, activations, derivative *mat.Dense) {
	adjustedOutput := new(mat.Dense)
	adjustedOutput.Mul(activations.T(), derivative)
	adjustedOutput.Scale(nn.config.LearningRate, adjustedOutput)
	original.Add(original, adjustedOutput)
}

func (nn *NeuralNet) adjustBias(original, derivative *mat.Dense) error {
	adjustedbiasOutput, err := sumAlongAxis(0, derivative)
	if err != nil {
		return err
	}
	adjustedbiasOutput.Scale(nn.config.LearningRate, adjustedbiasOutput)
	original.Add(original, adjustedbiasOutput)
	return nil
}

func sumAlongAxis(axis int, m *mat.Dense) (*mat.Dense, error) {

	numRows, numCols := m.Dims()

	var output *mat.Dense

	switch axis {
	case 0:
		data := make([]float64, numCols)
		for i := 0; i < numCols; i++ {
			col := mat.Col(nil, i, m)
			data[i] = floats.Sum(col)
		}
		output = mat.NewDense(1, numCols, data)
	case 1:
		data := make([]float64, numRows)
		for i := 0; i < numRows; i++ {
			row := mat.Row(nil, i, m)
			data[i] = floats.Sum(row)
		}
		output = mat.NewDense(numRows, 1, data)
	default:
		return nil, errors.New("invalid axis, must be 0 or 1")
	}

	return output, nil
}

func randPopulateMatrices(matrices ...*mat.Dense) {
	randomSource := rand.NewSource(time.Now().UnixNano())
	randomGenerator := rand.New(randomSource)
	for _, matrix := range matrices {
		rawMatrix := matrix.RawMatrix().Data
		for i := range rawMatrix {
			rawMatrix[i] = randomGenerator.Float64()
		}
	}
}

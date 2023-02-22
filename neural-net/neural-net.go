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
	biasHidden    *mat.Dense
	weightsOutput *mat.Dense
	biasOutput    *mat.Dense
}

type NeuralNetConfig struct {
	CountInputNeurons  int
	CountOutputNeurons int
	CountHiddenNeurons int
	CountEpochs        int
	LearningRate       float64
}

func New(config NeuralNetConfig) *NeuralNet {
	return &NeuralNet{config: config}
}

func (nn *NeuralNet) Train(independentVars, dependentVars *mat.Dense) error {
	nn.hiddenWeights = mat.NewDense(nn.config.CountInputNeurons, nn.config.CountHiddenNeurons, nil)
	nn.biasHidden = mat.NewDense(1, nn.config.CountHiddenNeurons, nil)
	nn.weightsOutput = mat.NewDense(nn.config.CountHiddenNeurons, nn.config.CountOutputNeurons, nil)
	nn.biasOutput = mat.NewDense(1, nn.config.CountOutputNeurons, nil)

	randPopulateMatrices(nn.hiddenWeights, nn.biasHidden, nn.weightsOutput, nn.biasOutput)

	output := new(mat.Dense)

	err := nn.backpropagate(independentVars, dependentVars, nn.hiddenWeights, nn.biasHidden, nn.weightsOutput, nn.biasOutput, output)
	if err != nil {
		return err
	}
	return nil
}

func (nn *NeuralNet) Predict(x *mat.Dense) (*mat.Dense, error) {
	if nn.hiddenWeights == nil || nn.weightsOutput == nil {
		return nil, errors.New("the supplied weights are empty")
	}
	if nn.biasHidden == nil || nn.biasOutput == nil {
		return nil, errors.New("the supplied biases are empty")
	}

	output := new(mat.Dense)
	feedForward(x, nn.hiddenWeights, nn.biasHidden, nn.weightsOutput, nn.biasOutput, output)
	return output, nil
}

func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func sigmoidPrime(x float64) float64 {
	return sigmoid(x) * (1.0 - sigmoid(x))
}

func (nn *NeuralNet) backpropagate(independentVars, dependentVars, hiddenWeights, biasHidden, weightsOutput, biasOutput, output *mat.Dense) error {
	for i := 0; i < nn.config.CountEpochs; i++ {

		// feed forward
		hiddenLayerActivations := feedForward(independentVars, hiddenWeights, biasHidden, weightsOutput, biasOutput, output)

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
		nn.adjustParams(hiddenWeights, independentVars, derivativeHiddenLayer)
		hiddenErr := nn.adjustBias(biasHidden, derivativeHiddenLayer)
		if hiddenErr != nil {
			return hiddenErr
		}
	}
	return nil
}

func feedForward(independentVars, hiddenWeights, biasHidden, weightsOutput, biasOutputt, output *mat.Dense) *mat.Dense {
	hiddenLayerInput := new(mat.Dense)
	hiddenLayerInput.Mul(independentVars, hiddenWeights)
	hiddenLayerInput.Apply(func(i, j int, v float64) float64 { return v + biasHidden.At(0, j) }, hiddenLayerInput)

	hiddenLayerActivations := new(mat.Dense)
	applySigmoid := func(_, _ int, v float64) float64 { return sigmoid(v) }
	hiddenLayerActivations.Apply(applySigmoid, hiddenLayerInput)

	outputLayerInput := new(mat.Dense)
	outputLayerInput.Mul(hiddenLayerActivations, weightsOutput)
	outputLayerInput.Apply(func(i, j int, v float64) float64 { return v + biasOutputt.At(0, j) }, outputLayerInput)
	output.Apply(applySigmoid, outputLayerInput)
	return hiddenLayerActivations
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

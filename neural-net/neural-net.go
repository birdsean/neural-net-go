package neuralnet

import (
	"errors"

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

func (nn *NeuralNet) Train(inputVars, desiredOutputs *mat.Dense) error {
	nn.hiddenWeights = mat.NewDense(nn.config.CountInputNeurons, nn.config.CountHiddenNeurons, nil)
	nn.hiddenBias = mat.NewDense(1, nn.config.CountHiddenNeurons, nil)
	// any other weight-layer matrices need to have rows equal to cols of previous matrix and rows equal to cols of next matrix
	nn.outputWeights = mat.NewDense(nn.config.CountHiddenNeurons, nn.config.CountOutputNeurons, nil)
	nn.outputBias = mat.NewDense(1, nn.config.CountOutputNeurons, nil)

	randPopulateMatrices(nn.hiddenWeights, nn.hiddenBias, nn.outputWeights, nn.outputBias)

	output := new(mat.Dense)

	err := nn.backpropagate(inputVars, desiredOutputs, nn.hiddenWeights, nn.hiddenBias, nn.outputWeights, nn.outputBias, output)
	if err != nil {
		return err
	}
	return nil
}

func (nn *NeuralNet) Predict(inputVars *mat.Dense) (*mat.Dense, error) {
	if nn.hiddenWeights == nil || nn.outputWeights == nil {
		return nil, errors.New("the supplied weights are empty")
	}
	if nn.hiddenBias == nil || nn.outputBias == nil {
		return nil, errors.New("the supplied biases are empty")
	}

	output := new(mat.Dense)
	nn.feedForward(inputVars, output)
	return output, nil
}

func (nn *NeuralNet) backpropagate(inputVars, desiredOutputs, hiddenWeights, biasHidden, weightsOutput, biasOutput, output *mat.Dense) error {
	for i := 0; i < nn.config.CountEpochs; i++ {

		// feed forward
		hiddenLayerActivations := nn.feedForward(inputVars, output)

		// backpropogate
		networkError := new(mat.Dense)
		networkError.Sub(desiredOutputs, output)
		derivativeOutput := calcDerivatives(output, networkError)

		errorAtHiddenLayer := new(mat.Dense)
		errorAtHiddenLayer.Mul(derivativeOutput, weightsOutput.T())
		derivativeHiddenLayer := calcDerivatives(hiddenLayerActivations, errorAtHiddenLayer)

		// adjust weights
		nn.adjustWeights(weightsOutput, hiddenLayerActivations, derivativeOutput)
		err := nn.adjustBias(biasOutput, derivativeOutput)
		if err != nil {
			return err
		}

		nn.adjustWeights(hiddenWeights, inputVars, derivativeHiddenLayer)
		hiddenErr := nn.adjustBias(biasHidden, derivativeHiddenLayer)
		if hiddenErr != nil {
			return hiddenErr
		}
	}
	return nil
}

func (nn *NeuralNet) feedForward(inputVars, output *mat.Dense) *mat.Dense {
	summingJunction := new(mat.Dense)
	summingJunction.Mul(inputVars, nn.hiddenWeights)
	summingJunction.Apply(func(i, j int, v float64) float64 { return v + nn.hiddenBias.At(0, j) }, summingJunction)

	nonLinearActivations := new(mat.Dense)
	applySigmoid := func(_, _ int, v float64) float64 { return sigmoid(v) }
	nonLinearActivations.Apply(applySigmoid, summingJunction)

	outputLayerInput := new(mat.Dense)
	outputLayerInput.Mul(nonLinearActivations, nn.outputWeights)
	outputLayerInput.Apply(func(i, j int, v float64) float64 { return v + nn.outputBias.At(0, j) }, outputLayerInput)
	output.Apply(applySigmoid, outputLayerInput)
	return nonLinearActivations
}

func (nn *NeuralNet) adjustWeights(originalWeights, activations, derivative *mat.Dense) {
	adjustedOutput := new(mat.Dense)
	adjustedOutput.Mul(activations.T(), derivative)
	adjustedOutput.Scale(nn.config.LearningRate, adjustedOutput)
	originalWeights.Add(originalWeights, adjustedOutput)
}

func (nn *NeuralNet) adjustBias(original, derivative *mat.Dense) error {
	adjustedbiasOutput := sumAlongColumn(derivative)
	adjustedbiasOutput.Scale(nn.config.LearningRate, adjustedbiasOutput)
	original.Add(original, adjustedbiasOutput)
	return nil
}

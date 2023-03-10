package neuralnet

import (
	"errors"
	"fmt"

	"gonum.org/v1/gonum/mat"
)

type NeuralNet struct {
	config        NeuralNetConfig
	hiddenWeights []*mat.Dense
	hiddenBias    []*mat.Dense
	outputWeights *mat.Dense
	outputBias    *mat.Dense
}

type NeuralNetConfig struct {
	CountInputNeurons  int
	CountOutputNeurons int
	HiddenLayers       []int
	CountEpochs        int
	LearningRate       float64
}

func New(config NeuralNetConfig) *NeuralNet {
	countEdges := 0
	prevNodes := config.CountInputNeurons
	for _, nodeCount := range config.HiddenLayers {
		countEdges += prevNodes * nodeCount
		prevNodes = nodeCount
	}
	countEdges += config.CountOutputNeurons * prevNodes
	fmt.Printf("Initializing Neural Net with %d edges and %d epochs\n", countEdges, config.CountEpochs)
	return &NeuralNet{config: config}
}

func (nn *NeuralNet) Train(inputVars, desiredOutputs *mat.Dense) {
	prevColCount := nn.config.CountInputNeurons
	for i := 0; i < len(nn.config.HiddenLayers); i++ {
		edges := mat.NewDense(prevColCount, nn.config.HiddenLayers[i], nil)
		bias := mat.NewDense(1, nn.config.HiddenLayers[i], nil)
		randPopulateMatrices(edges, bias)
		nn.hiddenWeights = append(nn.hiddenWeights, edges)
		nn.hiddenBias = append(nn.hiddenBias, bias)
		prevColCount = nn.config.HiddenLayers[i]
	}
	nn.outputWeights = mat.NewDense(prevColCount, nn.config.CountOutputNeurons, nil)
	nn.outputBias = mat.NewDense(1, nn.config.CountOutputNeurons, nil)
	randPopulateMatrices(nn.outputWeights, nn.outputBias)

	output := new(mat.Dense)
	nn.backpropagate(inputVars, desiredOutputs, output)
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

func (nn *NeuralNet) backpropagate(inputVars, desiredOutputs, output *mat.Dense) {
	for i := 0; i < nn.config.CountEpochs; i++ {
		// feed forward
		hiddenLayerActivations := nn.feedForward(inputVars, output)

		// backpropogate
		networkError := new(mat.Dense)
		networkError.Sub(desiredOutputs, output)

		errorAtHiddenLayer := new(mat.Dense)
		previousWeights := nn.outputWeights
		previousBias := nn.outputBias
		previousDerivatives := calcDerivatives(output, networkError)
		for j := len(nn.hiddenWeights) - 1; j >= 0; j-- {
			errorAtHiddenLayer.Mul(previousDerivatives, previousWeights.T())
			nn.adjustWeights(previousWeights, hiddenLayerActivations[j], previousDerivatives)
			nn.adjustBias(previousBias, previousDerivatives)
			previousWeights = nn.hiddenWeights[j]
			previousBias = nn.hiddenBias[j]
			previousDerivatives = calcDerivatives(hiddenLayerActivations[j], errorAtHiddenLayer)
			errorAtHiddenLayer.Reset()
		}

		nn.adjustWeights(previousWeights, inputVars, previousDerivatives)
		nn.adjustBias(previousBias, previousDerivatives)
	}
}

func (nn *NeuralNet) feedForward(inputVars, output *mat.Dense) []*mat.Dense {
	applySigmoid := func(_, _ int, v float64) float64 { return sigmoid(v) }
	summingJunction := new(mat.Dense)
	previousActivations := inputVars
	activations := []*mat.Dense{}
	for idx := 0; idx < len(nn.hiddenWeights); idx++ {
		summingJunction.Mul(previousActivations, nn.hiddenWeights[idx])
		summingJunction.Apply(func(i, j int, v float64) float64 { return v + nn.hiddenBias[idx].At(0, j) }, summingJunction)

		nonLinearActivations := new(mat.Dense)
		nonLinearActivations.Apply(applySigmoid, summingJunction)
		activations = append(activations, nonLinearActivations)

		previousActivations = nonLinearActivations
		summingJunction.Reset()
	}

	outputLayerInput := new(mat.Dense)
	outputLayerInput.Mul(activations[len(activations)-1], nn.outputWeights)
	outputLayerInput.Apply(func(i, j int, v float64) float64 { return v + nn.outputBias.At(0, j) }, outputLayerInput)
	output.Apply(applySigmoid, outputLayerInput)
	return activations
}

func (nn *NeuralNet) adjustWeights(originalWeights, activations, derivative *mat.Dense) {
	adjustedOutput := new(mat.Dense)
	adjustedOutput.Mul(activations.T(), derivative)
	adjustedOutput.Scale(nn.config.LearningRate, adjustedOutput)
	originalWeights.Add(originalWeights, adjustedOutput)
}

func (nn *NeuralNet) adjustBias(original, derivative *mat.Dense) {
	adjustedbiasOutput := sumAlongColumn(derivative)
	adjustedbiasOutput.Scale(nn.config.LearningRate, adjustedbiasOutput)
	original.Add(original, adjustedbiasOutput)
}

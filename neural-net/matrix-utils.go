package neuralnet

import (
	"math/rand"
	"time"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

func sumAlongColumn(m *mat.Dense) *mat.Dense {
	_, numCols := m.Dims()
	var output *mat.Dense

	data := make([]float64, numCols)
	for i := 0; i < numCols; i++ {
		col := mat.Col(nil, i, m)
		data[i] = floats.Sum(col)
	}
	output = mat.NewDense(1, numCols, data)

	return output
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

func calcDerivatives(outputSignals, errorSignals *mat.Dense) *mat.Dense {
	outputSlopes := calcTangentSlopes(outputSignals)
	derivatives := new(mat.Dense)
	derivatives.MulElem(errorSignals, outputSlopes)
	return derivatives
}

func calcTangentSlopes(outputSignals *mat.Dense) *mat.Dense {
	slopes := new(mat.Dense)
	applySigmoidPrime := func(_, _ int, v float64) float64 { return sigmoidPrime(v) }
	slopes.Apply(applySigmoidPrime, outputSignals)
	return slopes
}

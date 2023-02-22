package neuralnet

import (
	"reflect"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func basicNN() NeuralNet {
	return NeuralNet{
		config: NeuralNetConfig{
			CountInputNeurons:  2,
			CountOutputNeurons: 2,
			CountHiddenNeurons: 2,
			CountEpochs:        1,
			LearningRate:       1,
		},
		hiddenWeights: mat.NewDense(2, 2, []float64{1, 1, 1, 1}),
		hiddenBias:    mat.NewDense(1, 2, []float64{1, 1}),
		outputWeights: mat.NewDense(2, 2, []float64{1, 1, 1, 1}),
		outputBias:    mat.NewDense(1, 2, []float64{1, 1}),
	}
}

func TestNeuralNet_adjustBias(t *testing.T) {
	type args struct {
		original   *mat.Dense
		derivative *mat.Dense
	}
	tests := []struct {
		name string
		args args
		want mat.Matrix
	}{
		{
			name: "adjusts bias",
			args: args{original: mat.NewDense(1, 2, []float64{1, 1}), derivative: mat.NewDense(2, 2, []float64{1, 1, 1, 1})},
			want: mat.NewDense(1, 2, []float64{3, 3}),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			nn := basicNN()
			nn.adjustBias(tt.args.original, tt.args.derivative)
			if !reflect.DeepEqual(tt.args.original, tt.want) {
				t.Errorf("adjustBias=%v, want %v", tt.args.original, tt.want)
			}
		})
	}
}

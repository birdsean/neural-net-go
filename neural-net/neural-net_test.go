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

func TestNeuralNet_adjustWeights(t *testing.T) {
	type args struct {
		originalWeights *mat.Dense
		activations     *mat.Dense
		derivative      *mat.Dense
	}
	tests := []struct {
		name string
		args args
		want *mat.Dense
	}{
		{
			name: "adjust weights",
			args: args{
				originalWeights: mat.NewDense(2, 3, []float64{1, 1, 1, 1, 1, 1}),
				activations:     mat.NewDense(5, 2, []float64{2, 2, 2, 2, 2, 2, 2, 2, 2, 2}),
				derivative:      mat.NewDense(5, 3, []float64{3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3}),
			},
			want: mat.NewDense(2, 3, []float64{31, 31, 31, 31, 31, 31}),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			nn := basicNN()

			nn.adjustWeights(tt.args.originalWeights, tt.args.activations, tt.args.derivative)
			if !reflect.DeepEqual(tt.args.originalWeights, tt.want) {
				t.Errorf("adjustWeights=%v, want %v", tt.args.originalWeights, tt.want)
			}
		})
	}
}

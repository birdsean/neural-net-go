package neuralnet

import (
	"reflect"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func populateMatrix(n float64, m *mat.Dense) *mat.Dense {
	rawMatrix := m.RawMatrix().Data
	for i := range rawMatrix {
		rawMatrix[i] = n
	}
	return m
}

func Test_sumAlongColumn(t *testing.T) {
	type args struct {
		m *mat.Dense
	}
	tests := []struct {
		name string
		args args
		want *mat.Dense
	}{
		{
			name: "sum along 2x3 matrix works",
			args: args{populateMatrix(2, mat.NewDense(2, 3, nil))},
			want: mat.NewDense(1, 3, []float64{4, 4, 4}),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := sumAlongColumn(tt.args.m); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("sumAlongColumn() = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_randPopulateMatrices(t *testing.T) {
	type args struct {
		matrices []*mat.Dense
	}
	m1 := mat.NewDense(1, 1, nil)
	m2 := mat.NewDense(2, 2, nil)
	tests := []struct {
		name string
		args args
	}{
		{
			name: "populates multiple matrices",
			args: args{[]*mat.Dense{m1, m2}},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if m1.Norm(1) != 0 || m2.Norm(1) != 0 {
				t.Errorf("randPopulatematrices test unexpectedly started with a filled matrix. m1 is empty: %v, m2 is empty: %v", m1.Norm(1) != 0, m2.Norm(1) != 0)
			}
			randPopulateMatrices(tt.args.matrices...)
			if m1.Norm(1) == 0 || m2.Norm(1) == 0 {
				t.Errorf("randPopulatematrices returned an unchanged matrix. m1 is empty: %v, m2 is empty: %v", m1.Norm(1) == 0, m2.Norm(1) == 0)
			}
		})
	}
}

func Test_calcDerivatives(t *testing.T) {
	type args struct {
		outputSignals *mat.Dense
		errorSignals  *mat.Dense
	}
	tests := []struct {
		name string
		args args
		want *mat.Dense
	}{
		{
			name: "calculates derivates between two matrices",
			args: args{outputSignals: mat.NewDense(2, 2, []float64{0, 0, 0, 0}), errorSignals: mat.NewDense(2, 2, []float64{0.5, 0.5, 1, 0.5})},
			want: mat.NewDense(2, 2, []float64{0.125, 0.125, 0.25, 0.125}),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := calcDerivatives(tt.args.outputSignals, tt.args.errorSignals); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("calcDerivatives() = %v, want %v", got, tt.want)
			}
		})
	}
}

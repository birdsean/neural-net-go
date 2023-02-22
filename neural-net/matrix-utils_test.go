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

package neuralnet

import "testing"

func Test_sigmoid(t *testing.T) {
	type args struct {
		x float64
	}
	tests := []struct {
		name string
		args args
		want float64
	}{
		{
			name: "calculates sigmoid(2) correctly",
			args: args{x: 2},
			want: 0.88079707797788,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := sigmoid(tt.args.x)
			if got < tt.want-0.00000000001 || got > tt.want+0.00000000001 {
				t.Errorf("sigmoid() = %v, want %v", got, tt.want)
			}
		})
	}
}

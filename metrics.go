package GDRegressor

import "github.com/sschrs/matrix"

func MSE(yReal, yPredicted [][]float64) float64 {
	real := matrix.AsMatrix(yReal)
	predicted := matrix.AsMatrix(yPredicted)

	return real.Subtract(predicted).Apply(func(x matrix.Col) matrix.Col {
		return x * x
	}).Mean()
}

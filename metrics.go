package GDRegressor

import (
	"github.com/sschrs/matrix"
	"math"
)

func MSE(yReal, yPredicted [][]float64) float64 {
	real := matrix.AsMatrix(yReal)
	predicted := matrix.AsMatrix(yPredicted)

	return real.Subtract(predicted).Apply(func(x matrix.Col) matrix.Col {
		return x * x
	}).Mean()
}

func RMSE(yReal, yPredicted [][]float64) float64 {
	return math.Sqrt(MSE(yReal, yPredicted))
}

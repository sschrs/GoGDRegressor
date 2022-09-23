package GDRegressor

import "github.com/sschrs/matrix"

type SGDRegressor struct {
	Iterations     int
	intercept, Eta float64
	coef           []float64
	fitted         bool
	theta, x, y    matrix.Matrix
}

package GDRegressor

import "github.com/sschrs/matrix"

type BGDRegressor struct {
	Iterations     int
	intercept, Eta float64
	coef           []float64
	fitted         bool
	theta, x, y    matrix.Matrix
}

func NewBGDRegressor() *BGDRegressor {
	return &BGDRegressor{
		Iterations: 500,
		Eta:        0.1,
	}
}

func (bgd *BGDRegressor) Fit(x, y [][]float64) *BGDRegressor {
	X := matrix.AsMatrix(x)
	Y := matrix.AsMatrix(y)

	X = X.JoinColumn(matrix.GenerateColumn(len(x), 1), 0)
	theta := matrix.GenerateRand(X.Shape()["cols"], 1).Divide(10)

	for i := 0; i < bgd.Iterations; i++ {
		gradient := X.T().Dot(X.Dot(theta).Subtract(Y)).Multiply(2.0 / float64(len(x)))
		theta = theta.Subtract(gradient.Multiply(bgd.Eta))
	}

	bgd.theta = theta
	bgd.intercept = float64(theta[0][0])

	var coef []float64
	for i := 1; i < len(theta); i++ {
		coef = append(coef, float64(theta[i][0]))
	}

	bgd.coef = coef
	bgd.fitted = true

	return bgd
}

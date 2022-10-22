package GDRegressor

import (
	"github.com/sschrs/matrix"
	"math/rand"
	"time"
)

type SGDRegressor struct {
	Iterations     int
	intercept, Eta float64
	coef           []float64
	fitted         bool
	theta, x, y    matrix.Matrix
}

func NewSGDRegressor() *SGDRegressor {
	return &SGDRegressor{
		Iterations: 1000,
		Eta:        0.1,
		fitted:     false,
	}
}

func learning_schedule(eta float64) float64 {
	return 5 / (50 + eta)
}

func (sgd *SGDRegressor) Fit(x, y [][]float64) *SGDRegressor {
	X := matrix.AsMatrix(x)
	Y := matrix.AsMatrix(y)

	X = X.JoinColumn(matrix.GenerateColumn(len(x), 1), 0)

	theta := matrix.GenerateRand(X.Shape()["cols"], 1).Divide(10)

	for i := 0; i < sgd.Iterations; i++ {
		for j := 0; j < len(sgd.x); j++ {
			rand.Seed(time.Now().UnixNano())
			random_index := rand.Intn(len(sgd.x))
			xi := X[random_index : random_index+1]
			yi := Y[random_index : random_index+1]

			gradient := xi.T().Dot(xi.Dot(theta).Subtract(yi)).Multiply(2)
			sgd.Eta = learning_schedule(sgd.Eta)
			theta = theta.Subtract(gradient.Multiply(sgd.Eta))
		}
	}
	sgd.fitted = true
	sgd.intercept = float64(theta[0][0])
	sgd.theta = theta
	var coef []float64
	for i := 1; i < len(sgd.theta); i++ {
		coef = append(coef, float64(sgd.theta[i][0]))
	}

	sgd.coef = coef
	return sgd
}

func (sgd *SGDRegressor) Coef() []float64 {
	return sgd.coef
}

func (sgd *SGDRegressor) Intercept() float64 {
	return sgd.intercept
}

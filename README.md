# Gradient Descent Regressor for GoLang
This package allows you to create simple linear models with the Batch and Stochastic Gradient Descent methods.

## Batch Gradient Descent

Sample usage for Batch Gradient Descent:
```go
import "github.com/sschrs/GDRegressor"

x := [][]float64{{1.51}, {1.97}, {0.13}, {0.08}, {0.57}, {0.24}, {1.07}, {1.04}, {0.76}, {1.35}}
y := [][]float64{{9.08}, {10.64}, {4.73}, {5.16}, {5.66}, {4.93}, {7.21}, {7.13}, {7.33}, {5.94}}

bgd := GDRegressor.NewBGDRegressor()
bgd.Iterations = 1000
bgd.Eta = 0.1
bgd.Fit(x, y)
```

Make predictions:
```go
predictedValues, err := bgd.Predict(x)
if err != nil {
    fmt.Println(err)
}
rmse := GDRegressor.RMSE(y, predictedValues)
```

Get intercept ant coefficients:
```go
intercept := bgd.Intercept()
coef := bgd.Coef()
```

## Stochastic Gradient Descent
Sample usage for Stochastic Gradient Descent:
```go
import "github.com/sschrs/GDRegressor"
x := [][]float64{{1.51}, {1.97}, {0.13}, {0.08}, {0.57}, {0.24}, {1.07}, {1.04}, {0.76}, {1.35}}
y := [][]float64{{9.08}, {10.64}, {4.73}, {5.16}, {5.66}, {4.93}, {7.21}, {7.13}, {7.33}, {5.94}}

sgd := GDRegressor.NewSGDRegressor()
sgd.Iterations = 1000
sgd.Eta = 0.1
sgd.Fit(x, y)
```

Make predictions:
```go
predictedValues, err := sgd.Predict(x)
if err != nil {
    fmt.Println(err)
}
rmse := GDRegressor.RMSE(y, predictedValues)
```

Get intercept ant coefficients:
```go
intercept := sgd.Intercept()
coef := sgd.Coef()
```
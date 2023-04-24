# Cerceptron

Simple perceptron in C.

## How to build

* with ascii plot: `clang *.c -fopenmp; ./a.out`
* without ascii plot and openmp: `clang *.c; ./a.out`

## Wait, what is a perceptron?

A perceptron is a machine learning model that performs well for classification against linearly separable data.

[If you didn't get that, this video explains it visually pretty well](https://www.youtube.com/watch?v=4Gac5I64LM4)

There are two types of perceptrons:

* __single-layer perceptron__ - what this code and the rest of the readme is about.
* __multi-layer perceptron__ - a fully-connected neural network. Can have more layers and more complex layer functions.

Usually when saying just "perceptron" it is inferred that it is a reference to a single-layer one.

### Hyperparameters

The power of machine learning is letting a model optimize its own parameters (training). However, there
are _hyperparameters_: defined by us to guide the model in its optimization process.

For perceptrons, there are two hyperparameters: the learning rate and error threshold.

The error threshold determines when training will stop and the learning rate scale the training steps.

### Layer

A perceptron layer is composed of:

* __neurons__ - which have weights that are fine-tuned using training data.
* __threshold functions__ - that maps an input _x_ to a single binary value.

__threshold functions__ usually look like this:

```
if( w * x + b > 0 ) {
    return 1;
} else {
    return 0;
}
```

where `w` is the weight vector and `b` is the bias.

Each layer takes an vector as input, performs a dot operation with it on the weights and maps it to a binary value using the
threshold function.

By the way, to compute the bias, you can just append a `1` element to the input vector and append another weight element.

The output can be send to the next layer or become the final result by being properly discretized.

### Training

update weights:

```
r = learning rate
y = true label
ŷ = predicted label
w[i] = w[i] + r * (y - ŷ) * x[i]
```

### Final Result

Perceptrons can do binary classification using just the __threshold function__:

* returning __1__ means one class.
* returning __0__ means another class.

## Multiclass Perceptron

Multiclass perceptrons can classify multiple classes. However, training and classification will be differ from the binary classifier Perceptron since each class will have it's own weight vector.

### Inference

1. Perform the dot product between the input and weight vector of every class
2. The inferred class will be the one in which the dot product obtained the highest score

### Training

Define `f(x, y)`, a function that returns every possible input/output pair. The training will iterate over these pairs.
0. Start with all weight vector filled with zeros
1. Predict class using current weights
```
y = argmax(w[i] * f(x, y))
```
2. If prediction is correct, nothing happens
3. If prediction is incorrect:
	1. Lower the score of wrong answer
	```
	w[wrong] = w[wrong] - f(x, y)
	```
	2. Raise the score of right answer
```
	w[correct] = w[correct] + f(x, y)
```

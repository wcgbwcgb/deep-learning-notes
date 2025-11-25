# Binary Classification
For example: identify a cat (1 cat, 0 non cat)  
Input: Red, Green, Blue  
feature vector: a vector with a size of $64*64*3$  
$n_{x}$ = n = 12288  
using x to predict y  

## notations:
- (x,y), where x is an $n_{x}$ dimensional feature vector, and y is the output
m training example: $\{(x^{(1)}, y^{(1)}),(x^{(2)},y^{(2)}),...,(x^{(m)},y^{(m)})\}$
- $m_{train}$ = number of training samples
- $m_{test}$ = number of test examples
- X = $\begin{bmatrix} {x}^{(1)}&{x}^{(2)}&{x}^{(3)}\cdots{x}^{(m)}\end{bmatrix}$  
shape = $(n_x, m)$
- Y = $\begin{bmatrix} {y}^{(1)}&{y}^{(2)}&{y}^{(3)}\cdots{y}^{(m)}\end{bmatrix}$  
shape = $(1, m)$

## logistic regression
sigmoid function:
$$
\hat{y} = \sigma(z), where z = {w}^Tx + b
$$

$\sigma(z)=\frac{1}{1+{e}^{-z}}$

If $z\to \infty$, then sigmoid function will be close to $\frac{1}{1+0}$

If $z\to -\infty$, then sigmoid function will be close to $\frac{1}{1+\infty}$, so it will be close to 0


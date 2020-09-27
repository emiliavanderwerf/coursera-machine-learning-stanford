function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% Implement cost function
hypothesis = sigmoid(X * theta);
summation = sum(-y .* log(hypothesis) - (1 - y) .* log(1 - hypothesis));
unregularizedJ = (1 / m) * summation;
J = unregularizedJ + (lambda / (2 * m)) * sum(theta(2 : end) .^ 2);

% Implement gradient descent
unregularizedGrad = (1 / m) * sum((hypothesis - y) .* X);
grad = zeros(size(theta)); % Initialize
grad(1) = unregularizedGrad(1); % Theta zero is not regularized
grad(2:end) = unregularizedGrad(:,2:end) + lambda / m * theta(2:end)';

grad = grad(:);

end

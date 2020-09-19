function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% Compute cost function
[unregularizedJ, unregularizedGrad] = costFunction(theta, X, y);
J = unregularizedJ + (lambda / (2 * m)) * sum(theta(2 : end) .^ 2);

% Compute gradient
grad = zeros(size(theta)); % Initialize
grad(1) = unregularizedGrad(1); % Theta zero is not regularized
grad(2:end) = unregularizedGrad(:,2:end) + lambda / m * theta(2:end)';

end

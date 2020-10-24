function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% Compute the regularization parameter
reg = (lambda / (2*m)) * sum(theta(2:end).^2);
% Compute hypothesis
h_theta = X * theta;
% Compute cost function, putting it all together
J = (1 / (2*m)) * sum((h_theta - y).^2) + reg;

% To vectorize the computation, declare a theta with 0's for the 1st column
thetaZero = [0; theta(2:end)];
% Compute gradient
grad = (1/m) * sum((h_theta - y) .* X) + (lambda / m) * thetaZero';

% =========================================================================

grad = grad(:);

end

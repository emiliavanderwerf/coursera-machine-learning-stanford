function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

m = length(y); % number of training examples

J = (1/(2*m))*sum((theta(1)*X(:,1)+theta(2)*X(:,2)-y).^2); % Un-vectorized

end

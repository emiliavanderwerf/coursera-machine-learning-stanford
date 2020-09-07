function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    % Compute the new theta0 and theta1 using simultaneous update
    hypothesis=theta(1)*X(:,1)+theta(2)*X(:,2);
    tempThetaZero=theta(1)-alpha*(1/m)*sum((hypothesis-y).*X(:,1));
    tempThetaOne=theta(2)-alpha*(1/m)*sum((hypothesis-y).*X(:,2));

    % Assign the newly computed values for theta0 and theta1
    theta=[tempThetaZero; tempThetaOne];
    
    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end

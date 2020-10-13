function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

% Add the bias unit
X = [ones(m, 1) X];

% Recode the labels as vectors containing only values of 0 and 1 to use in the
% cost function
y_recode = zeros(num_labels, size(y,1));
for i = 1 : size(y, 1),
  y_recode(y(i), i) = 1;
endfor

% Calculate h_theta of x(i) sub k, which is the activation (output value) of the
% kth output unit.
% Do forward propagation:
a_1 = X;
z_2 = Theta1 * a_1';
a_2 = [ones(m, 1) sigmoid(z_2)'];
h_theta = sigmoid(Theta2 * a_2');

% This is the unregularized cost; still need to regularize
J = (1 / m) * sum(sum(-y_recode .* log(h_theta) - (1-y_recode) .* log(1-h_theta)));

% Perform regularization
% Remove the bias unit, since we do not regularize it
Theta1_nobias = Theta1(:, 2:size(Theta1,2));
Theta2_nobias = Theta2(:, 2:size(Theta2,2));

% Regularization function
J = J + (lambda / (2*m)) * (sum(sum(Theta1_nobias.^2)) + sum(sum(Theta2_nobias.^2)));

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.

for t = 1 : m,

  % Step 1: Perform a feedforward pass, computing the activations for layers 2
  % and 3.
  a_1 = X(t,:); % Get the t-th row; already has the bias unit
  z_2 = Theta1 * a_1';
  a_2 = sigmoid(z_2);
  a_2 = [1; a_2]; % Add the bias unit
  z_3 = Theta2 * a_2;
  h_theta = sigmoid(z_3); % a_3 is the final activation layer

  % Step 2: Compute delta sub k for layer 3 (the output layer)
  delta_3 = h_theta - y_recode(:, t); % (10x1)
  
  % Step 3: Compute delta for hidden layer 2
  z_2 = [1; z_2]; % Add the bias unit
  delta_2 = (Theta2' * delta_3) .* sigmoidGradient(z_2);
  
  % Step 4: Accumulate the gradient; skip or remove delta sub zero of layer 2
  delta_2 = delta_2(2 : end);
  Theta2_grad = Theta2_grad + delta_3 * a_2';
  Theta1_grad = Theta1_grad + delta_2 * a_1;  

endfor

% Step 5: Obtain the unregularized gradient for the neural net cust function
% by dividing the accumulated gradients by 1/m
Theta1_grad = (1 / m) * Theta1_grad;
Theta2_grad = (1 / m) * Theta2_grad;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Regularize the gradient for j >= 1; note that when j = 0, no regularization
% is necessary
Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + ((lambda / m) * Theta1(:, 2:end));
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + ((lambda / m) * Theta2(:, 2:end));

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

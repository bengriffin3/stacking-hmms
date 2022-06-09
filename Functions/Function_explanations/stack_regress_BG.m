function [yhat_star,yhat,w,beta] = stack_regress_BG(X,y,epsilon)
%%% function to perform stacked ridge regression constraining the weights to be non-negative
%%% In ridge regression, the coefficients B are determined as the
%%% minimizers of:
%%% Sum_n (y_n - B*xn)^2 subject to sum_n B_m^2 = s (we use s = 1 here)
% Epsilon is lambda from the notes, which controls the strength of the
% penalty.
% Epsilon too small leads to overfitting
% Epsilon too large leads to underfitting

% Outputs will probably be the best prediction (yhat_star?), the coefficients that give
% you that prediction (beta_star?)

N = size(X{1},1) % Number of subjects
M = length(X) % Number of models we want to combine
% X is a no. models (learners) x 1 cell array, where each cell is made up of no.
% subjects x no. predictors (instead of a distance matrix)
% M = length X gives the longest dimension of the cell array X, which here
% is number of models (learners)

% y is the target

% initialise predictions (for each subject and each model (learner))
yhat = zeros(N,M);

% 'centering the variables' to give intercepts of models meaning
my = mean(y);
y = y-my;



for n = 1:N % for each subject

    % 1LOO - leave one out cross-validation
    ind = setdiff(1:N,n);
    
    for m = 1:M % for each model we make predictions
        % get the training data
        xn = X{m}(ind,:);
        yn = y(ind);

        % determine weights to combine predictors (input features in design matrix) (using ridge regression)
        b = (xn' * xn + epsilon * eye(size(xn,2))) \ (xn' * yn);

        % use weights to form predictions on the test data (since LOO-CV
        % the test data is just 1 subject)
        yhat(n,m) = X{m}(n,:) * b;

    end
end

% Now we use these predictions to find stacking weights

% stacking weights
% w = (yhat' * yhat) \ (yhat' * y);
opts1 = optimset('display','off');
w = lsqlin(yhat,y,[],[],ones(1,M),1,zeros(M,1),ones(M,1),[],opts1) % constrained linear least-squares is equivalent to penalising ridge regression.



%%% Now we can't just use the same weights and multiply by the predictions
%%% we just made, we need to train and test the model on different data
beta = cell(M,1);
%beta_star = [];
% so let's make new predictions
for m = 1:M

    % determine stacking weights
    beta{m} = (X{m}' * X{m} + epsilon * eye(size(X{m},2))) \ (X{m}' * y);

    % use stacking weights to make (new) predictions
    yhat(:,m) = X{m} * beta{m};

%     beta_star = [beta_star; beta{m} * w(m)]
end

yhat_star = yhat * w; % we multiply the predictions by the weights to get our superior predictions, y*



% % I think Diego was just checking to see if this gave the same output as
% % above (and it does)
% yhat_star2 = zeros(size(yhat_star));
% for m = 1:M
%     yhat_star2 = yhat_star2 + X{m} * beta{m} * w(m);
% end
%     
end



% subtracting mean from all y means we set the mean to 0
% Known as 'centering the variables' (this does not effect the
% coefficients found through ridge regression), but makes the intercept of
% the model we find meaningful.

% If we applied normalization, we would lose interpretability of our model,
% so we apply standardization (actually just centering the variables?)

% setdiff() returns the data in '1:N' that is not in 'n', with no
% repetitions i.e. this is getting the indices for LOO-CV

% This is the kind of cross-validation thing, where we leave 1 out of
% N each time we do it. So we then get the data for all except
% that 1 portion of the data next.
% We use level-one cross validation data to avoid overfitting.

% recall 'A\B' solves the system of linear equations A*x=B, so we are solving
% (xn' * xn + epsilon * eye(size(xn,2))) * x = (xn' * yn)
% Top of page 59 - solution is given by B = (X'X +
% epsilon*eye())^(-1)*X'y

% stacking weights
% lsqlin solve constrained linear least-squares problem, where we want to:
% minimise (1/2)|yhat*w - y|^2, or equivalently
% minimise (1/2)|y - w*yhat|^2
% subject to:
% sum of all w = 1
% 0 <= w =< 1
% (opts1 just sets the optimisation options i.e. 'display' 'off')
% w are the weights (i.e. coefficients) used to form a linear combination
% of our solutions.
% w = (X'X)^(-1)*X'y


% recall 'A\B' solves the system of linear equations A*x=B, so we are solving
% (X{m}' * X{m} + epsilon * eye(size(X{m},2))) * x = (X{m}' * y), or
% (X'X + epsilon * eye())^(-1)*X'y
function [yhat,W] = stacking_regress_ben(HMMs_to_stack,var_target,sim_params)
% Stacking function
rng('default') % set seed for reproducibility (CV-folds are random)

n_subjects = size(HMMs_to_stack,1);
n_models = size(HMMs_to_stack,2);
n_folds = 10;

% Divide data into test and train data
c = cvpartition(n_subjects,'KFold',n_folds);
yhat = NaN(n_subjects,1);

% for each fold
for i = 1:n_folds
    
    idxTrain = c.training(i);
    idxTest = c.test(i);

    %x = lsqlin(C,d,A,b)
    C = HMMs_to_stack(idxTrain,:);
    d = var_target(idxTrain);

    if strcmp(sim_params.stack_method, 'least_squares')
        % determine stacking weights
        opts1 = optimset('display','off');
        % W = lsqlin(predictors,targets,[],[], sum(W) = 1, 0 < w < 1)
        W = lsqlin(C,d,[],[],ones(1,n_models),1,zeros(n_models,1), ones(n_models,1),[],opts1);

        % Use stacking weights to form new predictions
        yhat(idxTest) = HMMs_to_stack(idxTest,:)*W; % make predictions

        % Make predictions using random forests
    elseif strcmp(sim_params.stack_method, 'random_forest')
        W = 0;
        Mdl_stack_rf = fitrensemble(C,d,'Method','Bag');
        yhat(idxTest) = predict(Mdl_stack_rf,HMMs_to_stack(idxTest,:));

    end
end

end
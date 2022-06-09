load('HMM_predictions.mat')
X = squeeze(vars_hat(:,1,:));
load('vars_target.mat')
Y = vars(:,1);

%%
% Cross validation (train: 70%, test: 30%)
cv = cvpartition(size(X,1),'HoldOut',0.3);
idx = cv.test;
X_test = X(idx,:);
X_train = X(~idx,:);
Y_test = Y(idx);
Y_train = Y(~idx);

% Here we use holdout to:
% train on 70% of data
Mdl3 = fitrensemble(X_train,Y_train,'Method','Bag')
% and test on 30% of data
Yfit = predict(Mdl3,X_test);
% Compare predictions with targets
[Y_test Yfit]

%%
% Here we use 10-fold (default for 'CrossVal','On') to train model
cvens = fitrensemble(X,Y,'Method','Bag','CrossVal','on');
% Then use the trained model to make 10-fold predictions
Yfit = kfoldPredict(cvens);
% Compare predictions with targets
[Y_test Yfit]



%%
Mdl3 = TreeBagger(100,X,Y,'Method','regression')

%%
figure;
for i = 1:15
    subplot(5,3,i)
    histogram(var_plot(:,i))
end
%%
figure;
for i = 1:15
    subplot(5,3,i)
    histogram(squared_error(:,i,1))
end







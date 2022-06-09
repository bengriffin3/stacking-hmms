%% Combining predictions
% In this script we combine the prediction using stacking (and metafeatures
% etc.). This assume that some of the variables made in PredictPhenotype
% have been stored after previously running the predictions. It allows us
% to test the stacking quickly.
% i.e. Stacking predictions once the original 5 have been made
build_regression_data_V2_0;
tic
% Load stuff for stacking
load('KLdistances_ICA50')
load('CV_data.mat')

DistHMM = DistMat;



% Set prediciton parameters
parameters_prediction = struct();
parameters_prediction.verbose = 0;
parameters_prediction.method = 'KRR';
parameters_prediction.alpha = [0.1 0.5 1.0 5];
parameters_prediction.sigmafact = [1/2 1 2];
repetitions = 5;

load HMMs_meta_data.mat
n_subjects = size(vars,1); % number of subjects
MF1 = ones(size(vars,1),repetitions);
MF2 = maxFO_all_reps;
MF3 = switchingRate_all_reps;
metafeatures = [MF1 MF3];% MF2 MF3];


% Initialise variables (stacking)
% explained_variance_ST_QUICK = NaN(size(vars,2),1);
% explained_variance_ST_reps_QUICK = NaN(size(vars,2),repetitions);
vars_hat_ST_QUICK = NaN(size(vars,1),size(vars,2),1);
vars_hat_ST_reps_QUICK = NaN(size(vars,1),size(vars,2),repetitions);
prediction_error = NaN(size(vars,2),1);
prediction_error_reps = NaN(size(vars,2),repetitions);
% prediction_error_FWLS = NaN(size(vars,2),1);
% prediction_error_FWLS_reps = NaN(size(vars,2),repetitions);

prediction_error_META = NaN(size(vars,2),1);
prediction_error_META_reps = NaN(size(vars,2),repetitions);


% % Initialise variables (stacking + metafeatures)
% explained_variance_ST_FWLS = NaN(size(vars,2),1);
% vars_hat_ST_FWLS = NaN(size(vars,1),size(vars,2),1);
% explained_variance_ST_reps_FWLS = NaN(size(vars,2),repetitions);
% vars_hat_ST_reps_FWLS = NaN(size(vars,1),size(vars,2),repetitions);

W_QUICK = NaN(repetitions,10,size(vars,2));% a weight for every repetition of the HMM, for every fold, for every intelligence feature
% W_FWLS = NaN(size(metafeatures,2),10,size(vars,2));% a weight for every repetition of the HMM, for every fold, for every intelligence feature


for j = [1:size(vars,2)]
        disp(['Vars ' num2str(j) ])
        
        y = vars(:,j); % here probably you need to remove subjects with missing values
        
        y_new = y;
        D_new = DistHMM;
        conf_new = conf;
        twins_new = twins;
        
        % BG code to remove subjects with missing values+
        non_nan_idx = find(~isnan(y));
        which_nan = isnan(y);
        if any(which_nan)
            y_new = y(~which_nan);
            D_new = DistHMM(~which_nan,~which_nan,:);
            conf_new = conf(~which_nan,:);
            twins_new = twins(~which_nan,~which_nan);
            warning('NaN found on Yin, will remove...')
        end
        
        D_new_rep= reshape(mat2cell(D_new, size(D_new,1), size(D_new,2), ones(1,size(D_new,3))),[repetitions 1 1]);
        %rng('default')
        size(y_new)
        [yhat_ST_QUICK,yhat_ST_reps_QUICK,w_QUICK] = stacking_predictions_BG(y_new,D_new_rep,QpredictedY_vars{j},QYin_vars{j},sigma_vars{j},alpha_vars{j},parameters_prediction,twins_new,conf_new);
        [yhat_ST_META,yhat_ST_reps_META,w_META] = stacking_predictions_METAFUNCTION(y_new,D_new_rep,QpredictedY_vars{j},QYin_vars{j},sigma_vars{j},alpha_vars{j},parameters_prediction,metafeatures,twins_new,conf_new);
    
        % Standardise the data
        yhat_ST_QUICK = (yhat_ST_QUICK - mean(yhat_ST_QUICK))/std(yhat_ST_QUICK);
        yhat_ST_reps_QUICK = (yhat_ST_reps_QUICK - mean(yhat_ST_reps_QUICK))./std(yhat_ST_reps_QUICK);
        y_new = (y_new- mean(y_new))/std(y_new);
        
        yhat_ST_META = (yhat_ST_META - mean(yhat_ST_META))/std(yhat_ST_META);
        yhat_ST_reps_META = (yhat_ST_reps_META - mean(yhat_ST_reps_META))./std(yhat_ST_reps_META);

        % Note predictions and determine explained variance
        prediction_error(j) = sum((y_new - yhat_ST_QUICK).^2);
        prediction_error_reps(j,:) = sum((y_new - yhat_ST_reps_QUICK).^2);

        % Note predictions and determine explained variance
        prediction_error_META(j) = sum((y_new - yhat_ST_META).^2);
        prediction_error_META_reps(j,:) = sum((y_new - yhat_ST_reps_META).^2);

%         explained_variance_ST_QUICK(j) = corr(squeeze(yhat_ST_QUICK),y_new).^2;
%         explained_variance_ST_reps_QUICK(j,:) = corr(squeeze(yhat_ST_reps_QUICK),y_new).^2;
%         vars_hat_ST_store_QUICK = NaN(size(vars,1),1);
%         vars_hat_ST_store_QUICK(non_nan_idx) = yhat_ST_QUICK;
%         vars_hat_ST_QUICK(:,j) = vars_hat_ST_store_QUICK;
% 
%         vars_hat_store_ST_reps_QUICK = NaN(size(vars,1),repetitions);
%         vars_hat_store_ST_reps_QUICK(non_nan_idx,:) = yhat_ST_reps_QUICK;
%         vars_hat_ST_reps_QUICK(:,j,:) = vars_hat_store_ST_reps_QUICK;

        % Note down the weights with which we combined predictors
%         W_QUICK(:,:,j) = w_QUICK;

        % Repeat for metafeatures
        %rng('default')
        %[yhat_ST_FWLS,yhat_ST_reps_FWLS,w_FWLS,w_QUICK_2,folds] = stacking_predictions_BG_mf(y_new,D_new_rep,QpredictedY_vars{j},QYin_vars{j},sigma_vars{j},alpha_vars{j},parameters_prediction,twins_new,conf_new,metafeatures);
% 
%         [yhat_ST_FWLS,yhat_ST_reps_FWLS,w_FWLS] = stacking_predictions_meta(y_new,D_new_rep,QpredictedY_vars{j},QYin_vars{j},sigma_vars{j},alpha_vars{j},parameters_prediction...
%                                                                             ,twins_new,conf_new,metafeatures,folds_store{j});
% 
%         
%                 % Standardise the data
%         yhat_ST_FWLS = (yhat_ST_FWLS - mean(yhat_ST_FWLS))/std(yhat_ST_FWLS);
%         yhat_ST_reps_FWLS = (yhat_ST_reps_FWLS - mean(yhat_ST_reps_FWLS))./std(yhat_ST_reps_FWLS);
%         %y_new = (y_new- mean(y_new))/std(y_new);
% 
%         % Note predictions and determine explained variance
%         prediction_error_FWLS(j) = sum((y_new - yhat_ST_FWLS).^2);
%         prediction_error_FWLS_reps(j,:) = sum((y_new - yhat_ST_reps_FWLS).^2);
        
        % 
%         % Note predictions and determine explained variance
%         explained_variance_ST_FWLS(j) = corr(squeeze(yhat_ST_FWLS),y_new).^2;
%         explained_variance_ST_reps_FWLS(j,:) = corr(squeeze(yhat_ST_reps_FWLS),y_new).^2;
%         vars_hat_ST_store_FWLS = NaN(size(vars,1),1);
%         vars_hat_ST_store_FWLS(non_nan_idx) = yhat_ST_FWLS;
%         vars_hat_ST_FWLS(:,j) = vars_hat_ST_store_FWLS;
% 
%         vars_hat_store_ST_reps_FWLS = NaN(size(vars,1),repetitions);
%         vars_hat_store_ST_reps_FWLS(non_nan_idx,:) = yhat_ST_reps_FWLS;
%         vars_hat_ST_reps_FWLS(:,j,:) = vars_hat_store_ST_reps_FWLS;
% 
%         % Note down the weights with which we combined metafeatures/predictors
%         W_FWLS(:,:,j) = w_FWLS;
%         %W_QUICK_2(:,:,j) = w_QUICK_2;
end

% save('HMM_predictions_stack_QUICK.mat','explained_variance_ST_QUICK','vars_hat_ST_QUICK','explained_variance_ST_reps_QUICK','vars_hat_ST_reps_QUICK','W_QUICK')
% save('HMM_predictions_stack_FWLS.mat','explained_variance_ST_FWLS','vars_hat_ST_FWLS','explained_variance_ST_reps_FWLS','vars_hat_ST_reps_FWLS','W_FWLS')

toc

%% Explore the data
% Here we look at how good at the predictions each repetition of the HMM is
% (each with a different number of states, K = 3, 8, 13, 18, 23)
figure()
bar(mean(explained_variance_ST_reps_QUICK,1)')
xlabel('No. states'); ylabel('Mean r^2')
title('Mean r^2 across intelligence variables by no. states of HMM runs')
K_vec = {'3','8','13','18','23'};
set(gca, 'XTick', 1:length(K_vec),'XTickLabel',K_vec);


figure()
bar([3 8 13 18 23], mean(squeeze(mean(W_QUICK,2)),2))
xlabel('No. states'); ylabel('Contribution of prediction');
title('Contribution of original HMM repetition to the stacked prediction')



%% Plot the original 5 repetitions and the new combined prediction
X = 1:size(vars,2);
figure
%scatter(X,explained_variance_ST_FWLS,'o','r') % plot my stacked value with metafeature
%hold on
scatter(X,explained_variance_ST_QUICK,'o','g') % plot Diego's stacked value
hold on
scatter(X,explained_variance_ST_reps_QUICK,'x','b') % plot the original 5 repetitions of HMM

% Format chart
xlabel('Intelligence features'); ylabel('r^2')
legend('FWLS Stacked predictor','Stacked predictor','Original HMM runs')
legend('Stacked predictor','Original HMM runs')

%% Plot the original 5 repetitions and the new combined prediction
X = 1:size(vars,2);
figure
scatter(X,log(log(prediction_error_META)),'o','r') % plot my stacked value with metafeature
hold on
scatter(X,log(log(prediction_error)),'o','g') % plot Diego's stacked value
hold on
scatter(X,log(log(prediction_error_reps)),'x','b') % plot the original 5 repetitions of HMM

% Format chart
xlabel('Intelligence features'); ylabel('log prediction error')
legend('Metafeatures prediction', 'Stacked predictor','Original HMM runs')




% Goal: we want to see if higher entropy means more accurate predictions
% for a given subject

% load meta data
load('HMMs_meta_data_subject_BEST_5_reps')
%load('HMMs_meta_data_subject')
MF2 = Entropy_subject;
MF3 = likelihood_subject;
standardized_MF2 = (MF2-mean(MF2))./std(MF2) + 1;
% Question: do we want to normalize?
% For e.g. entropy, the higher the state, the higher the entropy. Now we
% don't just want to emphasise the higher state predictions, so we need to
% normalize to get them on the same scale. Likelihood is very large so we
% scale just to help us read it a bit better
metafeature_check = standardized_MF2;
%normalized_MF2 = (MF2-min(MF2))./(max(MF2)-min(MF2));
%normalized_MF2_mean_1 = normalized_MF2 - mean(normalized_MF2) + 1;
%normalized_MF2_mean_1 = MF2-min(MF2);

load('vars_best_5_reps.mat')
load('HMM_predictions_stack_meta_BEST_5_reps.mat')
%load('vars_v4.mat')
%load('HMM_predictions_stack.mat')
for var = 1:34
    var;
y_new = vars(:,var);
yhat_ST = squeeze(vars_hat_ST(:,var,:));


[B,I] = sort(metafeature_check(:,1:3),2);
% rank from 1 to 3 about which repetition has highest entropy
% 1 is highest entropy, 3 is lowest entropy
% theoretically, higher entropy should be the best guess
r = repmat((1:3),length(y_new),1);

r_metafeature = NaN(length(y_new),3);
for i = 1:length(y_new)
    index_store = I(i,:);
    rank_store = r(i,:);
    rank_store(index_store) = rank_store;
    r_metafeature(i,:) = rank_store;
end
r_metafeature;

% Now, we find the accuracy of our predictions and rank them
prediction_accuracy = abs(y_new-yhat_ST(1:3)).^2;
[B_yhat,I_yhat] = sort(abs(y_new-yhat_ST(1:3)).^2,2);

r = repmat((1:3),length(y_new),1);

r_yhat = NaN(length(y_new),3);
for i = 1:length(y_new)
    index_store = I_yhat(i,:);
    rank_store = r(i,:);
    rank_store(index_store) = rank_store;
    r_yhat(i,:) = rank_store;
end
r_yhat;

% How many times was the highest entropy equal to the best prediction?
% Second highest equal to second best and so on.
sum(r_metafeature == r_yhat)

% How many rank positions out is the metafeature ranking and then
% predictions ranking across all subjects? (/mean per subject)
% rank_diff = r_metafeature - r_yhat;
% sum(sum(abs(rank_diff)))/1001
end

%%
% are the metafeatures correlated with the predictions in any way?
corr_mat = corr(standardized_MF2(:,1:3),squeeze(vars_hat_ST(:,1,1:3)))
diag(corr_mat) % take the diagonal to get columnwise correlation

%% Plots
% we want to plot the relationship between HMM runs' accuracy and their
% metafeatures

% load data
load('HMM_predictions.mat')
load('HMMs_meta_data_subject_r_1_27_40_63')
load('vars_target.mat')

% store data
Entropy_plot = Entropy_subject_all(:,1:27);
%Entropy_plot = Entropy_subject_all(:,[2 6 14 25 44]);
vars_target = vars;
vars_predictions = vars_hat(:,:,1:27);
%vars_predictions = vars_hat(:,:,[2 6 14 25 44]);

% calculate prediction accuracies
n_subjects = 1001;
squared_error = ((vars_target-vars_predictions).^2);%/n_subjects
mean_squared_error = squeeze(sum(squared_error,'omitnan')/nnz(~isnan(squared_error)));

% here we plot entropy vs prediction accuracy
for var = 1
    for rep = 1
        X = mean(Entropy_plot);
        Y = mean_squared_error(var,:);
        figure()
        scatter(X,Y,'x')
        %plot(X,Y)
        title('Relationship between entropy and prediction accuracy for 27 HMM runs')
        xlabel('Mean Entropy'); ylabel('Mean Squared Error')
        
    end
end

% %%
% % For a specific run, let's plot for each subject the 5 runs
% Entropy_plot = Entropy_subject_all(:,[2 6 14 25 44])
% vars_predictions = vars_hat(:,:,[2 6 14 25 44]);
% squared_error = ((vars_target-vars_predictions).^2);%/n_subjects
% 
% 
% for var = 1
%     squared_error_var = squeeze(squared_error(:,var,:));
%     figure()
%     for sub = 1:10
%         squared_error_sub = squared_error_var(sub,:);
%         X = Entropy_plot(sub,:);
%         Y = squared_error_sub;
%         plot(X,Y)
%         hold on
%     end
%     xlabel('Entropy'); ylabel('Squared Error');
% end

%%
% Let's simulate correlated data and then see if they improve the
% predictions
rng default  % For reproducibility

% let's begin with the predictions
% load data
load('HMM_predictions.mat')
%load('HMMs_meta_data_subject_r_1_27_40_63')
%load('vars_target.mat')

% Select the 'best' 5 predictions
vars_predictions = vars_hat(:,:,[2 6 14 25 44]);

for var = 1
    var_pred = squeeze(vars_predictions(:,var,:));
end
% this is the prediction we want to simulate metadata for
var_pred(:,1);
mean(var_pred(:,1:5))
% mean predictions for HMM runs = 29.0392   29.0544   29.0193   29.0263   29.0242
% std predictions for HMM runs = 0.1916    0.3922    0.3695    0.1528    0.1208

% now let's explore the current metadata, and see what we want our vector
% to look roughly like
load('HMMs_meta_data_subject_r_1_27_40_63')
Entropy = Entropy_subject_all(:,[2 6 14 25 44])
mean(Entropy)
% mean entropy for HMM runs = 1.2240    1.5170    2.2479    2.8217    3.3704
% std entropy for HMM runs = 0.2505    0.3139    0.3136    0.3587    0.3846
% %%
% rng default  % For reproducibility
% % r = pearsrnd(mu,sigma,skew,kurt,m,n)
% % mean mu, standard deviation sigma, skewness skew, and kurtosis kurt.
% %[p1, type1] = pearsrnd(0,1,-1,4,1000,1);
% %[p2, type2] = pearsrnd(0,1,0.75,3,1000,1);
% [p1, type1] = pearsrnd(0,1,0,3,1000,1); % type = 0 -> normal distribution
% [p2, type2] = pearsrnd(0,1,0,3,1000,1);
% figure
% scatterhist(p1,p2)
% 
% u = copularnd('Gaussian',-0.8,1000);
% % figure
% % scatterhist(u(:,1),u(:,2))
% [s1,i1] = sort(u(:,1));
% [s2,i2] = sort(u(:,2));
% x1 = zeros(size(s1));
% x2 = zeros(size(s2));
% x1(i1) = sort(p1);
% x2(i2) = sort(p2);
% % figure
% % scatterhist(x1,x2)
% copula_corr = corr(u,'Type','spearman')
%%
% convert this to a function if show Diego to clean it up
clear; clc;
rng default  % For reproducibility
tau = 0.5;
rho = copulaparam('Gaussian',tau);
% generate dependent random values

%figure
%scatterhist(u(:,1),u(:,2))
% 1001 subjects, 34 variables, 5 repetitions, for predictions &
% metafeatures
n_subjects = 1001;
n_var = 1; %34
n_repetitions = 5;
simulated_reps = NaN(n_subjects,2,n_repetitions,n_var);
%predictions_accuracy = NaN(n_variables,n_repetitions);
prediction_stack_accuracy = NaN(n_var);
prediction_FWLS_accuracy =  NaN(n_var);

var = 1;
for rep = 1:n_repetitions
    u_rep = copularnd('gaussian',rho,1001); % generate a two-column matrix of dependent random values
    b_rep = [betainv(u_rep(:,1),1,2), betainv(u_rep(:,2),1.5,2)]; %transform into random numbers from a beta distribution.
    simulated_reps(:,:,rep,var) = b_rep; % store simulated data
    %tau_sample = corr(b_rep,'type','kendall') % check correlations exist
end

% load predictions and targets
load('HMM_predictions.mat')
load('vars_target.mat')
Predictions = squeeze(vars_hat(:,var,[2 6 14 25 44]));
vars_target = vars(:,var);

% determine accuracy of predictions
Prediction_accuracy = (vars_target - Predictions).^2;


% load metadata
load('HMMs_meta_data_subject_r_1_27_40_63')
Entropy = Entropy_subject_all(:,[2 6 14 25 44]);

% Now we want to transform our random vectors into realistic entropy
% (same mean and standard deviation)
entropy_reps = squeeze(simulated_reps(:,2,:,:));
entropy_simu = entropy_reps - mean(entropy_reps); % translate mean to 0
entropy_simu = (entropy_simu./std(entropy_simu)).*std(Entropy);  % scale data to desired standard deviation
entropy_simu = entropy_simu + mean(Entropy); % translate to desired mean

% Now we translate our random vectors into realistic predictions
% (same mean and standard deviation)
% ACTUALLY, IT'S THE PREDICTION ACCURACY THAT NEEDS TO BE CORRELATED WITH
% THE ENTROPY, NOT THE PREDICTIONS THEMSELVES. AT THE MINUTE IT'ST JUST THE
% LARGER PREDICTION HAVE THE HIGHER ENTROPY, BUT WE WANT THE MORE ACCURATE
% PREDICITONS TO HAVE THE ENTROPY
prediction_errors_sim = squeeze(simulated_reps(:,1,:,:)); % the errors need to be in this form
predictions_simu = sqrt(prediction_errors_sim)+ mean(vars_target);
simulated_accuracy = (mean(vars_target) - predictions_simu).^2;

%pm = rand(1001,5); pm(pm>0.5) = 1; pm(pm<0.5) = -1; % randomise +- error % can't use this as it randomises correlation
%mean(sqrt(abs(predictions_simu_accuracy)))
%predictions_simu = (sqrt(abs(prediction_errors_sim)) - mean(sqrt(abs(prediction_errors_sim)))) + mean(vars_target);


%simulated_accuracy = (vars_target - predictions_simu).^2;
corr_test = [entropy_simu(:,1) simulated_accuracy(:,1)];
%corr_test = [entropy_simu(:,1) predictions_simu(:,1)];
corr(corr_test,'type','kendall')



% %predictions_accuracy(var,:) = sum((vars_target - predictions_reps(:,:,var)).^2)/1001;
% predictions_simu_accuracy = prediction_reps - mean(prediction_reps); % translate mean to 0
% predictions_simu_accuracy = (predictions_simu_accuracy./std(predictions_simu_accuracy)).*std(Prediction_accuracy);  % scale data to desired standard deviation
% predictions_simu_accuracy = predictions_simu_accuracy + mean(Prediction_accuracy); % translate to desired mean






% predictions_simu = prediction_reps - mean(prediction_reps); % translate mean to 0
% predictions_simu = (predictions_simu./std(predictions_simu)).*std(Predictions);  % scale data to desired standard deviation
% predictions_simu = predictions_simu + mean(Predictions); % translate to desired mean

% % store target variables
% load('vars_target.mat')
% vars_target = vars(:,var);
predictions_accuracy(var,:) = sum((vars_target - predictions_simu(:,:,var)).^2)/1001;




% metafeatures setup
% we first standardize the metafeatures
%entropy_simu_norm = (entropy_simu - mean(entropy_simu))./std(entropy_simu)+1;
entropy_simu_norm = (entropy_simu - min(entropy_simu))./(max(entropy_simu) - min(entropy_simu));
metafeature_array = [ones(1001,5) entropy_simu_norm];
n_metafeatures = size(metafeature_array,2)/n_repetitions;



% clear variables not needed
%clearvars -except predictions_simu entropy_simu vars_target var

% set up cross validation folds
n = 1001;
k = 10;
sub_partition = cvpartition(n,'KFold',k);
% idxTrain = training(sub_partition);
n_repetitions = 5;
prediction_stack = NaN(n,1);
prediction_FWLS = NaN(n,1);

for ifold = 1:10
    test_idx = test(sub_partition,ifold);
    training_idx = training(sub_partition,ifold);
    predictions_simu_test = predictions_simu(test_idx,:,var);
    predictions_simu_train = predictions_simu(training_idx,:,var);
    meta_test = metafeature_array(test_idx,:);
    meta_train = metafeature_array(training_idx,:);
    vars_test = vars_target(test_idx);
    vars_train = vars_target(training_idx);
    
    % set up metafeature_prediction_array
    A_train = NaN(sum(training_idx),n_metafeatures*n_repetitions);
    A_test = NaN(sum(test_idx),n_metafeatures*n_repetitions);
    for i = 1:n_metafeatures
        A_train(:,n_repetitions*i-n_repetitions+1:n_repetitions*i) = meta_train(:,n_repetitions*i-n_repetitions+1:n_repetitions*i).*predictions_simu_train;
        A_test(:,n_repetitions*i-n_repetitions+1:n_repetitions*i) = meta_test(:,n_repetitions*i-n_repetitions+1:n_repetitions*i).*predictions_simu_test;
    end
    
    % determine stacking weights
    opts1 = optimset('display','off');
    w_stack = lsqlin(predictions_simu_train,vars_train,[],[],ones(1,n_repetitions),1,zeros(n_repetitions,1),ones(n_repetitions,1),[],opts1);
    w_FWLS = lsqlin(A_train,vars_train,[],[],ones(1,n_repetitions*n_metafeatures),1,zeros(n_repetitions*n_metafeatures,1),ones(n_repetitions*n_metafeatures,1),[],opts1);
    
    % make stacked predictions
    prediction_stack(test_idx) = predictions_simu_test*w_stack;
    prediction_FWLS(test_idx) = A_test*w_FWLS;
end
prediction_stack_accuracy(var) = sum((vars_target - prediction_stack).^2)/n;
prediction_FWLS_accuracy(var) = sum((vars_target - prediction_FWLS).^2)/n;


% Plot the accuracy of the current predictions

X = 1; % no. variables
figure()
scatter(X,prediction_stack_accuracy,'o','r')
hold on
scatter(X,prediction_FWLS_accuracy,'o','g')
%scatter(X,predictions_accuracy,'x','b')
scatter(X,predictions_accuracy,'x','b')
legend('LSQLIN','FWLS','HMM reps')



% Need to  change correation stuff so mf are correlated to prediction accuracy


%%
% Then: mess about with p and method for obtaining stacking weights DONE
% Then: mess about with metafeatures simulation to see what works and what
% doesn't, e.g.
% - introduce a new metafeature 0.1 correlation doesn't do much for 1 metafeature, 
% but what about 2 metafeatures at 0.1 each?
% - can we introduce a nonlinear relationship and still see improvements?
% Might need a nonlinear method

clc; clear;
%rng('default') % set for reproducibility


% load predictions, targets, and metadata
load('HMM_predictions.mat')
load('vars_target.mat')
load('HMMs_meta_data_subject_r_1_27_40_63')

% Intialise variables
n_var = size(vars_hat,2); n_repetitions = 5;
explained_variance_ST = NaN(n_var,n_repetitions);
explained_variance_stack = NaN(n_var,1);
explained_variance_FWLS = NaN(n_var,1);

for var = 1:34
    var
    
    % Store data for variable
    Predictions = squeeze(vars_hat(:,var,[2 6 14 25 44]));
    vars_target = vars(:,var);
    Entropy = Entropy_subject_all(:,[2 6 14 25 44]);

    % Remove subjects with missing values
    which_nan = isnan(vars_target);
    if any(which_nan)
        vars_target = vars_target(~which_nan);
        Predictions = Predictions(~which_nan,:);
        warning('NaN found on Yin, will remove...')
    end
    
    % determine accuracy of predictions (we want our metafeature to be
    % correlated with this)
    Prediction_accuracy = (vars_target - Predictions).^2;
    
    % let x  be prediction accuracy
    x = Prediction_accuracy;
    
    % reverse: x = (s1 * u + m1)' to get u = randn(1, n);
    u = (x ./ std(x)) - mean(x);
    
    
    if any(which_nan)
        Entropy = Entropy(~which_nan,:);
    end
    
    % set parameters
    n = size(Entropy,1);
    m2 = mean(Entropy);
    s2 = std(Entropy);
    p = 0.1;
    v = randn(n, 1);
    
    %%%%%%%%%% METAFEATURE ARRAY SIMULATION %%%%%%%%%%%%%%
    % create a correlated metafeature array
    y = s2 .* squeeze((p * u + sqrt(1 - p^2) * v)) + m2;
    y = y + mean(Entropy) - mean(y);
    
    entropy_simu = y;
    
    % metafeatures setup
    % we first standardize the metafeatures
    entropy_simu_norm = (entropy_simu - mean(entropy_simu))+1;%./std(entropy_simu)+1; %entropy_simu_norm = (entropy_simu - min(entropy_simu))./(max(entropy_simu) - min(entropy_simu));
    metafeature_array = [squeeze(repmat(ones(n,5),1,1)) entropy_simu_norm];
    
    %%%%%%%%%% METAFEATURE ARRAY SIMULATION END %%%%%%%%%%%%%%
    
    % Make predictions
    [prediction_ST_accuracy,prediction_stack_accuracy,prediction_FWLS_accuracy,ev_ST,...
        ev_stack,ev_FWLS] = predictPhenotype_mf_simulation(Predictions,vars_target,metafeature_array);
    
    
    explained_variance_ST(var,:) = ev_ST;
    explained_variance_stack(var) = ev_stack;
    explained_variance_FWLS(var) = ev_FWLS;
end


%%
%%%%%%%%% PLOT SIMULATION RESULTS %%%%%%%%%%%%%%
X = 1:34; % no. variables
figure()
scatter(1:n_var,explained_variance_stack,'o','r')
hold on
scatter(1:n_var,explained_variance_FWLS,'o','g')
%scatter(X,predictions_accuracy,'x','b')
for rep = 1:n_repetitions
    scatter(1:n_var,explained_variance_ST(:,rep),'x','b')
end
legend('LSQLIN','FWLS','HMM reps')
title('Explained variance, rho = 0.1 LSQLIN')


% % Mean prediction error
% X = 1:34; % no. variables
% figure()
% scatter(1:n_var,prediction_stack_accuracy,'o','r')
% hold on
% scatter(1:n_var,prediction_FWLS_accuracy,'o','g')
% %scatter(X,predictions_accuracy,'x','b')
% for rep = 1:5
%     scatter(1:n_var,prediction_ST_accuracy,'x','b')
%     %scatter(1:n_var,squeeze(Prediction_accuracy_2(:,:,rep)),'x','b')
% end
% legend('LSQLIN','FWLS','HMM reps')
% title('Mean prediction error')
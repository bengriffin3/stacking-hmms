%% Assessing correlation between predictions (consistent no. of states)
% Load hmm predictions
clear;clc;
DirResults = ['Dokumenter/MATLAB/HMMMAR_BG/HMMMAR Results/'];
DirFC = ['FC_HMM_zeromean_1_covtype_full/'];
load([DirResults DirFC 'HMM_predictions.mat'])

% Clean hmm predictions (remove subject where we have NaNs in predictions)
%idx_nans = any(isnan(vars_hat(:,:,1)), 2);
varsKEEP = true(size(vars_hat,1),1);
% varsKEEP(idx_nans) = 0;
vars_hat_hmm = vars_hat(varsKEEP,:,:); 
explained_variance_hmm = explained_variance;

% % Load and clean hmm predictions (varying states)
% load FC_HMM_zeromean_1_covtype_full_vary_states/HMM_predictions.mat
% vars_hat_hmm_varying_states = vars_hat(varsKEEP,:,:);
% explained_variance_varying_states = explained_variance;

% Load and clean hmm stacked predictions
load FC_HMM_zeromean_1_covtype_full_stacked/HMM_predictions_stack.mat
vars_hat_stack = vars_hat_ST(varsKEEP,:,:);
vars_hat_stack_reps = vars_hat_ST_reps(varsKEEP,:,:);
explained_variance_stack = explained_variance_ST;
explained_variance_stack_reps = explained_variance_ST_reps;

%Load and clean hmm stacked predictions of varying states
load FC_HMM_zeromean_1_covtype_full_vary_states_3_repetitions/HMM_predictions_stack.mat
vars_hat_stack_states = vars_hat_ST(varsKEEP,:,:);
vars_hat_stack_states_reps = vars_hat_ST_reps(varsKEEP,:,:);
explained_variance_stack_states = explained_variance_ST;
explained_variance_stack_states_reps = explained_variance_ST_reps;

% Load features then store the intelligence ones
load('feature_groupings.mat')
feature_vec = feature_groupings(2)+1:feature_groupings(3); %[1:151];

% Extract intelligence features 
load 'Behavioural Variables/vars_clean.mat'
%vars_clean = vars_clean(~idx_nans,:);
vars_intelligence = vars_clean(:,feature_vec);
%vars_hat_hmm_intelligence = vars_hat_hmm(:,feature_vec,:);


%% Calculate p-values for hmm predictions
% Initialise variables
p_values_store = zeros(size([explained_variance_hmm explained_variance_varying_states]));
p_values_sig = zeros(size([explained_variance_hmm explained_variance_varying_states]));
explained_variance_store = [explained_variance_hmm explained_variance_varying_states];

% Calculate p-values
N = size(vars_hat,1);  % number of observations
for rep = 1:size(explained_variance_store,2)
    for var = 1:size(explained_variance_hmm,1)

        % Calculate p-values for the explained variance values
         r = explained_variance_store(var,rep); % do we need to square root?
         t = r*sqrt((N-2)/(1-r^2));
         p_values_store(var,rep) = 1 - tcdf(t,(N-2));

    end
end

% Convert p-values to asterisks (to add to MATLAB figures)
p_values_sig(p_values_store < 0.05) = 1;
p_value_all = replace(string(p_values_sig), {'1', '0'}, {'*', ''});


%% Calculate correlations between HMM runs (for intelligence features)


% Choose which HMM runs to plot
figure()
plot =  "hmm"; %"hmm_varying_states";


% Set variables based on which HMMs we want to plot
if plot == "hmm_varying_states"
    vars_hat_plot = vars_hat_hmm_varying_states;
    explained_variance_plot =  explained_variance_varying_states;
    p_value_plot = p_value_all(:,6:end);
    set(gcf, 'Name', 'FC-HMM: 5 HMM runs with varying K (3-15)')
elseif plot == "hmm"
    vars_hat_plot = vars_hat_hmm;
    explained_variance_plot =   explained_variance_hmm;
    p_value_plot = p_value_all(:,1:5);
    set(gcf, 'Name', 'FC-HMM: 5 HMM runs with 8 States')
end

p = 0;

for var = feature_vec
    p = p + 1;

    % Plot correlation between HMM runs (for intelligence variable)
    subplot(ceil(sqrt(length(feature_vec))),ceil(sqrt(length(feature_vec))),p)
    imagesc(corr(squeeze(vars_hat_plot(:,var,:))))

    % Note p-values and r^2 for figure labels
    p_var = p_value_plot(var,:);
    ev_var = explained_variance_plot(var,:);

    % Format chart
    colorbar()
    title(sprintf("Feature %d",var))
    set(gca,'yticklabel',{'r_1','r_2','r_3','r_4','r_5'});
    set(gca,'xticklabel',{strcat(sprintf("r_1=%.2f",ev_var(1)),p_var(1)),strcat(sprintf("r_2=%.2f",...
        ev_var(2)),p_var(2)),strcat(sprintf("r_3=%.2f",ev_var(3)),p_var(3)),...
        strcat(sprintf("r_4=%.2f",ev_var(4)),p_var(4)),strcat(sprintf("r_5=%.2f",ev_var(5)),p_var(5))});

    % Change axes font colours to highlight higher/lower correlations
    if sum(ev_var) > 0.3
        set(gca, 'XColor', [0 0.5 0],'YColor', [0 0.5 0]);
    elseif sum(ev_var) < 0.15
        set(gca, 'XColor', [1 0 0], 'YColor', [1 0 0]);
    else
        set(gca,'fontweight','bold','XColor', [0 0 0]);
    end
end

%% How good are our predictors?
% We can measure the efficacy of our predictors by looking at e.g. explained variance

% % We can also measure efficacy by looking at residual sum of squares (after standardising)
% vars_clean_intelligence_standardised = (vars_clean_intelligence - mean(vars_clean_intelligence))./std(vars_clean_intelligence);
% LSE_1 = sum(squeeze(sum((vars_clean_intelligence - vars_hat_hmm_intelligence).^2))) % calculate MSE
% % Now we could e.g. choose the best predicor
% [M,I] = min(LSE_1);
% I % Result: predictor 4 is the best predictor

%% Now we combine models using Ridge Regressione
%clear;clc;
% Load and store the variables we aim to predict
% load vars_clean.mat
% vars = vars_clean(:,11:44); % define which variables we want to predict

% Load and store our current predictions
%load FC_HMM_zeromean_1_covtype_full_stacked/HMM_predictions_stack.mat
%vars_hat = vars_hat_ST_reps;
%load FC_HMM_zeromean_1_covtype_full_vary_states_3_repetitions/HMM_predictions_stack.mat
%vars_hat = vars_hat_ST_reps(:,:,[1 4 7 10 13]); % note the current predictions (as produced by predictPhenotype)


% Set up metafunctions - just using the first (constant) metafeature is
% equivalent to using no metafeatures
load HMMs_meta_data.mat
n_subjects = size(vars,1); % number of subjects
MF1 = ones(n_subjects,1);
MF2 = mean(maxFO_all_reps,2)*3;%/10;
MF3 = mean(switchingRate_all_reps,2)*3;
metafeatures = [MF1 MF2 MF3];

% Perform stacked regression & stacked regression with metafeatures
[vars_hat_st,vars_hat_FWLS,explained_variance_st,explained_variance_FWLS] = stack_regress_metaf(vars,vars_hat,metafeatures);

% Plot the original 5 repetitions and the new combined prediction
X = 1:size(vars,2);
figure
scatter(X,explained_variance_st,'o','g'); hold on;
scatter(X,explained_variance_FWLS,'o','r')
%scatter(X,explained_variance_ST_reps,'x','b')
scatter(X,explained_variance,'x','b')

% Format chart
xlabel('Intelligence feature'); ylabel('r^2')
legend('Stacked predictor','FWLS predictor','Original HMM runs')


%% Exploring repetitions with a different number of states
% We now want to vary the number of states, so that our repetitions are
% more dissimilar to each other. However, because the HMM can sometimes
% product 'bad' results, we run the HMM for 3 times for each number of
% states, and keep the best
load FC_HMM_zeromean_1_covtype_full_vary_states_3_repetitions/HMM_predictions.mat
explained_variance_vary_states_3_reps = explained_variance;
K_vec = [3 3 3 5 5 5 8 8 8 13 13 13 15 15 15 18 18 18 23 23 23 25 25 25 28 28 28 33 33 33];
reshaped_exp_var = reshape(explained_variance_vary_states_3_reps,[34,3,10]);
vars_hat = vars_hat(varsKEEP,:,:);
vars_hat_vary_states_3_reps = reshape(vars_hat,[941,34,3,10]);


%% Choosing best repetition for each number of states
% For one particular state number, we plot the 3 repetitions for each
% intelligence feature and then choose the best repetition
x = 1:34;
figure
for rep = 1:3
    for k = 10
        y = reshaped_exp_var(:,rep,k);
        scatter(x,y)
        hold on
    end
end
% Format chart
title('r^2 for intelligence features, 3 HMM runs for 13 states');
xlabel('Intelligence features'); ylabel('r^2');
legend('Repetition 1','Repetition 2', 'Repetition 3')

% Repetition to keep:
% 3 states = rep 1; 5 states = rep 3; 8 states = rep 3; 13 states = rep 2
% 15 states = rep 1; 18 states = rep 3; 23 states = rep 3; 25 states = rep 2
% 28 states = rep 3; 33 states = rep 3;
best_rep = [1 3 3 2 1 3 3 2 3 3];
exp_var_best = zeros(34,10);
vars_hat_best = zeros(n_subjects,34,10);
for k = 1:10
    exp_var_best_iter = reshaped_exp_var(:,best_rep(k),k);
    exp_var_best(:,k) = exp_var_best_iter;
    vars_best_iter = vars_hat_vary_states_3_reps(:,:,best_rep(k),k);
    vars_hat_best(:,:,k) = vars_best_iter;
end
    
%% Now we combine the models using ridge regression
n_subjects = size(vars_hat_best,1); % number of subjects
n_var = size(vars_hat_best,2); % number of intelligence variables
model_plot_vec = [1:10];%[2 4 6 8 10]; % choose which of our 10 models we want to plot
n_models = length(model_plot_vec); % number models to combine

% Initialise variables
weights_vary_states = zeros(n_models,n_var);
yhat_star_vary_states = zeros(n_subjects,n_var);
explained_variance_combined_vary_states = zeros(n_var,1);

for m = 1:n_var

    % Save data for specific variable
    y = vars_clean_intelligence(:,m+1);
    %V = squeeze(vars_hat_best(:,m,1:n_models)); %yhat
    V = squeeze(vars_hat_best(:,m,model_plot_vec)); %yhat
    
    % Stacking weights (determined by least squares
    opts1 = optimset('display','off');
    weights_vary_states(:,m) = lsqlin(V,y,[],[],ones(1,n_models),1,zeros(n_models,1),ones(n_models,1),[],opts1);
    
    % Use the weights to form new predictions
    yhat_star_vary_states(:,m) = V * weights_vary_states(:,m);

    % Calculate new levels of explained variance (for combined model)
    explained_variance_combined_vary_states(m) = corr(yhat_star_vary_states(:,m),y).^2;

end

% Plot the original 10 repetitions and the new combination
X = 10:43;
Y = explained_variance_combined_vary_states;
figure
scatter(X,Y,'o','r')

hold on
for i = 1:n_models
    Y = exp_var_best(:,model_plot_vec(i));
    scatter(X,Y,'x','b')
    %scatter(X,Y)
end

% Format chart
title('r^2 for intelligence features: 10 HMM runs of varying states')
xlabel('Intelligence features'); ylabel('r^2');
legend('Combined predictor','Original HMM runs')
%legend('Combined predictor','1','2','3','4','5','6','7','8','9','10')
xlim([9 45])

%% Plot the total r^2 across all intelligence variables by no. states
figure()
bar(sum(exp_var_best))
xlabel('No. states'); ylabel('Total explained variance')
title('Total r^2 across intelligence variables by no. states of HMM runs')
set(gca,'xticklabel',{[K_vec(1:3:end)]})

%% Including meta-features
% Diego suggested: maximum fractional occupancy, free energy, number of
% states
% Let's begin by combining the metafeature data together across repetitions
% of the HMM
maxFO_all_reps = zeros(1001,5);
fehist_all_reps = zeros(100,5);
K_all_reps = zeros(5,1);
switchingRate_all_reps = zeros(1001,5);
% 
% K_vec = [3 3 3 5 5 5 8 8 8 13 13 13 15 15 15 18 18 18 23 23 23 25 25 25 28 28 28 33 33 33];
% 
for i = 1:5
    load (['HMMs_r' num2str(i) '_GROUP.mat'])
%     if length(fehist) == 99
%         fehist = [ fehist fehist(end)];
%     end
    
    maxFO_all_reps(:,i) = maxFO;
    fehist_all_reps(:,i) = fehist;
    K_all_reps(i) = hmm.K;
    switchingRate_all_reps(:,i) = switchingRate;
end
 save(['HMMs_meta_data.mat'],...
     'maxFO_all_reps','fehist_all_reps','K_all_reps','switchingRate_all_reps')
 %        'maxFO_all_reps','K_all_reps','switchingRate_all_reps')
          

% %
% load 'HMMMAR Results/FC_HMM_zeromean_1_covtype_full/HMMs_meta_data'
% fehist_FC_HMM = fehist_all_reps;
% load 'HMMMAR Results/Mean_FC_HMM_zeromean_0_covtype_full/HMMs_meta_data'
% fehist_Mean_FC_HMM = fehist_all_reps;
% load 'HMMMAR Results/Mean_HMM_zeromean_0_covtype_uniquefull/HMMs_meta_data'
% fehist_Mean_HMM = fehist_all_reps;
% load 'HMMMAR Results/FC_HMM_zeromean_1_covtype_full_vary_states_3_repetitions/HMMs_meta_data'
% fehist_FC_HMM_vary_states = fehist_all_reps;
% load 'HMMMAR Results/VAR_HMM_zeromean_1_covtype_diag/HMMs_meta_data'
% fehist_Var_HMM = fehist_all_reps;

% %% Incorporating metafeatures
% n_subjects = 1001; % number of subjects
% n_models = 5; % number models to combine
% n_var = 34;
% n_metafeatures = 2; % number of metafeatures
% 
% % Initialise variables
% weights_FWLS = NaN(n_models*n_metafeatures,n_var);
% yhat_star_FWLS = NaN(n_subjects,n_var);
% explained_variance_FWLS = NaN(n_var,1);
% 
% % Define metafeatures
% F1 = ones(n_subjects,1);
% F2 = mean(maxFO_all_reps,2)*3;
% %F2 = 2*F1;
% F = [F1 F2];
% 
% for v = 1:n_var
%     v
%     % Save data for specific variable
%     y = vars_intelligence(:,v); % intelligence variables
%     V = squeeze(vars_hat_stack_reps(:,v,:)); %yhat
% 
%     % BG code to remove subjects with missing values
%     non_nan_idx = find(~isnan(y));
%     y_new = y; V_new = V; F_new = F;
%     which_nan = isnan(y);
%     if any(which_nan)
%         y_new = y(~which_nan);
%         V_new = V(~which_nan,:);
%         F_new = F(~which_nan,:);
%         warning('NaN found on Yin, will remove...')
%     end
% 
%     % Create A matrix of learners and metafeatures
%     A = NaN(size(F1,1),n_metafeatures*size(V_new,2));
%     for i = 1:n_metafeatures % 2 metafeatures
%         A(non_nan_idx,n_models*i-4:n_models*i) = F_new(:,i).*V_new;
%     end
% 
%     % Stacking weights (determined by least squares
%     opts1 = optimset('display','off');
%     weights_FWLS(:,v) = lsqlin(A(non_nan_idx,:),y_new,[],[],ones(1,n_models*n_metafeatures),1,zeros(n_models*n_metafeatures,1),ones(n_models*n_metafeatures,1),[],opts1);
% 
% 
%     % Use the weights to form new predictions
%     yhat_star_FWLS(:,v) = A * weights_FWLS(:,v);
% 
%     % Calculate new levels of explained variance (for combined model)
%     explained_variance_combined(v) = corr(yhat_star_FWLS(non_nan_idx,v),y_new).^2;
% 
% end
% 
% 
% 
% 
% 
% X = 11:44;
% Y = explained_variance_combined;
% figure
% scatter(X,Y,'o','r')
% hold on
% Y = explained_variance_stack_reps;
% scatter(X,Y,'x','b')
% 
% 
% %%
% clc; clear;
% F1 = [1; 2; 1];
% F2 = 2*F1;
% F3 = 3*F1;
% F = [F1 F2 F3];
% n_metafeatures = 3;
% v_1 = [0.1 0.2 0.1];
% V = [v_1; 2*v_1; 3*v_1];
% 
% J = zeros(size(F1,1),n_metafeatures*size(V,1));
% for i = 1:size(F,1)
%     J(:,3*i-2:3*i) = [F(:,i).*V];
% end
% J


%% Now we combine models using Ridge Regression (OLD CODE!!!!!!!!!!!!!!)
% %clear;clc;
% repetitions = 5;%30;
% 
% n_subjects = size(vars_hat_stack_reps,1); % number of subjects
% n_var = size(vars_hat_stack_reps,2); % number of intelligence variables
% n_models = repetitions;%size(vars_hat_stack_reps,3); % number models to combine
% n_folds = 10; % number of folds for Cross-Validation scheme
% 
% % Initialise stacking variables
% weights = NaN(n_models,10,n_var); % 10 is 10-fold cross validation
% vars_hat_combined = NaN(n_subjects,n_var);
% explained_variance_combined = NaN(n_var,1);
% 
% 
% 
% % Set up metafunctions - just using the first (constant) metafeature is
% % equivalent to using no metafeatures
% load HMMs_meta_data.mat
% MF1 = ones(n_subjects,1);
% MF2 = mean(maxFO_all_reps,2)*3;%/10;
% MF3 = mean(switchingRate_all_reps,2)*3;
% metafeatures = [MF1 MF2 MF3];
% n_metafeatures = size(F_all,2); % number of metafeatures
% 
% % Initialise FWLS variables
% weights_FWLS = NaN(n_models*n_metafeatures,10,n_var);
% vars_hat_FWLS = NaN(n_subjects,n_var);
% explained_variance_FWLS = NaN(n_var,1);
% 
% 
% indices_all = crossvalind('Kfold',1:n_subjects,n_folds);
% 
% for m = 1:n_var
% 
%     % Save data for specific variable
%     y_all = vars_intelligence(:,m); % intelligence variables
%     %V_all = squeeze(vars_hat_stack_reps(:,m,:)); %yhat
%     V_all = squeeze(vars_hat_stack_states_reps(:,m,[1 7 13 19 25])); %yhat
%     % Varying number of states more increases improvement by metafeatures
% 
%     % Remove subjects with NaN values
%     y = y_all; V = V_all; F = metafeatures; indices = indices_all;
%     non_nan_idx = ~isnan(y);
%     which_nan = isnan(y);
%     if any(which_nan)
%         y = y_all(~which_nan);
%         V = V_all(~which_nan,:);
%         F = F_all(~which_nan,:);
%         indices = indices_all(~which_nan);
%         warning('NaN found on Yin, will remove...')
%     end
% 
%     yhat_star = NaN(sum(~which_nan),1);
%     yhat_star_FWLS = NaN(sum(~which_nan),1);
%     
%     for ifold = 1:10
% 
%         % Note indices of testing and training data
%         test = (indices == ifold);
%         train = ~test;
% 
%         % Split into train and test data
%         y_train = y(train); y_test = y(test);
%         V_train = V(train,:); V_test = V(test,:);
%         F_train = F(train,:); F_test = F(test,:);
% 
%         % Create A matrix of learners and metafeatures
%         A = NaN(size(y,1),n_metafeatures*n_models);
%         for i = 1:n_metafeatures % 2 metafeatures
%               A(:,n_models*i-n_models+1:n_models*i) = F(:,i).*V;
%         end
% 
%         % Stacking weights (determined by least squares)
%         opts1 = optimset('display','off');
%         weights(:,ifold,m) = lsqlin(V_train,y_train,[],[],ones(1,n_models),1,zeros(n_models,1),ones(n_models,1),[],opts1);
%         weights_FWLS(:,ifold,m) = lsqlin(A(train,:),y_train,[],[],ones(1,n_models*n_metafeatures),1,zeros(n_models*n_metafeatures,1),ones(n_models*n_metafeatures,1),[],opts1);
% 
%         % Use the weights to form new predictions
%         yhat_star(test) = V_test * weights(:,ifold,m); % stacking
%         yhat_star_FWLS(test) = A(test,:) * weights_FWLS(:,ifold,m); % FWLS
% 
% 
% 
%     end
%         % Store the new predictions
%         vars_hat_combined(~which_nan,m) = yhat_star;
%         vars_hat_FWLS(~which_nan,m) = yhat_star_FWLS;
% 
%     % Calculate new levels of explained variance (for combined model)
%     explained_variance_combined(m) = corr(vars_hat_combined(~which_nan,m),y).^2;
%     explained_variance_FWLS(m) = corr(vars_hat_FWLS(~which_nan,m),y).^2;
% 
% 
% end
% 
% [explained_variance_combined explained_variance_FWLS]
% A(test,:)
% % Plot the original 5 repetitions and the new combined prediction
% X = 11:44;
% %Y = explained_variance_stack;
% %Y = explained_variance_stack_states;
% Y = explained_variance_combined;
% figure
% scatter(X,Y,'o','g')
% hold on
% Y = explained_variance_FWLS;
% scatter(X,Y,'o','r')
% 
% Y = explained_variance_stack_reps;
% %Y = explained_variance_stack_states_reps;
% scatter(X,Y,'x','b')
% 
% % Format chart
% title('r^2 for intelligence features (5 HMM runs with 8 states)')
% xlabel('Intelligence feature'); ylabel('r^2')
% legend('Stacked predictor','FWLS prediction','Original HMM runs')
% % xlim([9 45])
% %xlim([1 151])





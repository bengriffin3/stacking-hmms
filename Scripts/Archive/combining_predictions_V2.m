%% Assessing correlation between predictions (consistent no. of states)
% Load hmm predictions
clear;clc;
DirResults = ['Dokumenter/MATLAB/HMMMAR/HMMMAR Results/'];
DirFC = ['FC_HMM_zeromean_1_covtype_full/'];
load([DirResults DirFC 'HMM_predictions.mat'])

% Clean hmm predictions (remove subject where we have NaNs in predictions)
idx_nans = any(isnan(vars_hat(:,:,1)), 2);
varsKEEP = true(size(vars_hat,1),1);
varsKEEP(idx_nans) = 0;
vars_hat_hmm = vars_hat(varsKEEP,:,:); 
explained_variance_hmm = explained_variance;

% % Load and clean static predictions
% load('FC_HMM_zeromean_1_covtype_full/staticFC_predictions.mat')
% explained_variance_static = explained_variance;
% vars_hat_static = vars_hat(varsKEEP, :);

% Load and clean hmm predictions
load FC_HMM_zeromean_1_covtype_full_vary_states/HMM_predictions.mat
vars_hat_hmm_varying_states = vars_hat(varsKEEP,:,:);
explained_variance_varying_states = explained_variance;


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
plot = "hmm"% "hmm_varying_states";

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

% We explore the correlation for intelligence features only (store in
% feature_groupings)
load('feature_groupings.mat')
for var = feature_groupings(2):feature_groupings(3)

    % Plot correlation between HMM runs (for intelligence variable)
    subplot(6,6,var - feature_groupings(2) + 1)
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

%% Combining predictors
% Now we want to see if we can combine predictors in order to make an even 
% better predictor. To do this, let's examine only the intelligence
% variables as these were the only variables for which we made good
% predictions.
load 'Behavioural Variables/vars_clean.mat'
vars_clean = vars_clean(~idx_nans,:);
vars_clean_intelligence = vars_clean(:,feature_groupings(2):feature_groupings(3));
vars_hat_hmm_intelligence = vars_hat_hmm(:,feature_groupings(2):feature_groupings(3),:);


%% Now we combine models using Ridge Regression
n_subjects = size(vars_hat_hmm_intelligence,1); % number of subjects
n_var = size(vars_hat_hmm_intelligence,2); % number of intelligence variables
n_models = size(vars_hat_hmm_intelligence,3); % number models to combine

% Initialise variables
weights = zeros(n_models,n_var);
yhat_star = zeros(n_subjects,n_var);
explained_variance_combined = zeros(n_var,1);

for m = 1:n_var
    % Save data for specific variable
    y = vars_clean_intelligence(:,m);
    V = squeeze(vars_hat_hmm_intelligence(:,m,:)); %yhat
    
    % Stacking weights (determined by least squares)
    weights(:,m) = lsqlin(V,y,[],[],ones(1,n_models),1,zeros(n_models,1),ones(n_models,1),[]);
    
    % Use the weights to form new predictions
    yhat_star(:,m) = V * weights(:,m);

    % Calculate new levels of explained variance (for combined model)
    explained_variance_combined(m) = corr(yhat_star(:,m),y).^2;

end

% Plot the original 5 repetitions and the new combined prediction
X = feature_groupings(2):feature_groupings(3);
Y = explained_variance_combined;
figure
scatter(X,Y,'o','r')
hold on
for i = 1:n_models
    Y = explained_variance_hmm(feature_groupings(2):feature_groupings(3),i);
    scatter(X,Y,'x','b')
end

% Format chart
title('r^2 for intelligence features (5 HMM runs with 8 states)')
xlabel('Intelligence feature'); ylabel('r^2')
legend('Combined predictor','Original HMM runs')
xlim([9 45])


%% Exploring repetitions with a different number of states
% We now want to vary the number of states, so that our repetitions are
% more dissimilar to each other. However, because the HMM can sometimes
% product 'bad' results, we run the HMM for 3 times for each number of
% states, and keep the best
load FC_HMM_zeromean_1_covtype_full_vary_states_3_repetitions/HMM_predictions.mat
explained_variance_vary_states_3_reps = explained_variance;
K_vec = [3 3 3 5 5 5 8 8 8 13 13 13 15 15 15 18 18 18 23 23 23 25 25 25 28 28 28 33 33 33];
reshaped_exp_var = reshape(explained_variance_vary_states_3_reps,[34,3,10]);
vars_hat(idx_nans,:,:) = [];
vars_hat_vary_states_3_reps = reshape(vars_hat,[941,34,3,10]);
%%
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
    
%%
n_models = 5; % number models to combine
n_var = size(vars_hat_best,2); % number of intelligence variables
n_subjects = size(vars_hat_best,3); % number of subjects

% Initialise variables
weights_vary_states = zeros(n_models,n_var);
yhat_star_vary_states = zeros(n_subjects,n_var);
explained_variance_combined_vary_states = zeros(n_var,1);


for m = 1:n_var

    % Save data for specific variable
    y = vars_clean_intelligence(:,m+1);
    V = squeeze(vars_hat_best(:,m,1:n_models)); %yhat
    
    % Stacking weights (determined by least squares)
    weights_vary_states(:,m) = lsqlin(V,y,[],[],ones(1,n_models),1,zeros(n_models,1),ones(n_models,1),[]);
    
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
    Y = exp_var_best(:,i);
    scatter(X,Y,'x','b')
end

% Format chart
title('r^2 for intelligence features: 5 HMM runs of varying states (3-15)')
xlabel('Intelligence features'); ylabel('r^2');
legend('Combined predictor','Original HMM runs')
xlim([9 45])



%% 
% % First though, let's exclude some of the variables which we weren't able to predict
% sig_explained_variance = average_explained_variance_hmm_5_reps<0.05
% % indices of variables we have predicted at least medium well done
% I = find(average_explained_variance_hmm_5_reps>0.05)% + 9 
% I+9
% explained_variance_combined(I) > average_explained_variance_hmm_5_reps(I)

% %%
% x = 1:35
% y = explained_variance_combine
% figure
% scatter(x,y,'x')
% title('r^2 for intelligence features')
% xlabel('Intelligence features'); ylabel('r^2');






% Second: assess best predictor using cross validation
% Note: this involves creating (e.g. for 10-fold validation) 10
% predictions, based on the other 9 sets of data, so I think it will take
% 10 times the amount of time? (which is too much?) 

% Let's say the entire dataset is the learning set, and so have a look at
% how our predictors have done across the dataset, then choose the best
% one (i.e. use v_k where k minimises sum_k (y_n - z_{kn})^2 )
 

% Basic ensemble
% Now we combine our predictors by finding the mean of them and check error
%vars_hat_combine = mean(vars_hat_hmm_intelligence,3);
%LSE_2 = sum(squeeze(sum((vars_clean_intelligence - vars_hat_combine).^2))) % calculate MSE
% Result: our predictor is slighty worse than the second best predictor


%%
% Weighted ensemble
% Now we randomly assign weights to our predictors to see if we can find a
% better combination (e.g. put a larger weight on the best predictor)
% w = [0.6 0.2 0.2];
% w_vec = [0:0.1:1];
% vars_hat_combine_weighted = (w(1)*vars_hat_hmm_intelligence(:,:,1) + w(2)*vars_hat_hmm_intelligence(:,:,2) + w(3)*vars_hat_hmm_intelligence(:,:,3));
% LSE_3 = sum(squeeze(sum((vars_clean_intelligence - vars_hat_combine_weighted).^2))); % calculate MSE
% LSE_store = zeros(10,10,10);
% for i = 1:11
%     for j = 1:11
%         for k = 1:11
%             w(1) = w_vec(i);
%             w(2) = w_vec(j);
%             w(3) = w_vec(k);
%             vars_hat_combine_weighted = (w(1)*vars_hat_hmm_clean_intelligence(:,:,1) + w(2)*vars_hat_hmm_clean_intelligence(:,:,2) + w(3)*vars_hat_hmm_clean_intelligence(:,:,3));
%             LSE_store(i,j,k) = sum(squeeze(sum((vars_clean_intelligence - vars_hat_combine_weighted).^2)));
%         end
%     end
% end
% LSE_store

% Here we have specified weights but how can we determine the best weights?
% We can use least squares by comparing the predicted values with the true
% values from the learning set, so
% vars_hat are the predictions (v)
% vars are the original values (y)
% vars are the rest of the variables except the one we are looking at (x)

%%
% clc
% load vars_clean.mat
% y = vars_clean(~idx,:); % the actual variables that we tried to predict
% % This is definitely y, but do we feed into stack_regress 1 variable at a
% % time (for all subjects?) and so use "for v = 1:151; y = vars_clean(:,v); end"
% 
% X = squeeze(num2cell(vars_hat_hmm_clean,[2 1]));
% epsilon = 1;
% y = y(1,:)'
% 
% % X = squeeze(num2cell(vars_hat_hmm_clean,[1 2]));
% % epsilon = 10;
% % y = y(:,1)';
% stack_regress_BG(X,y,epsilon)
% 



% %% Calculate correlations between HMM runs while varying the number of states
% load FC_HMM_zeromean_1_covtype_full_vary_states/HMM_predictions.mat
% %idx_nans = any(isnan(vars_hat(:,:,1)), 2);
% %varsKEEP = true(size(vars_hat,1),1);
% %varsKEEP(idx_nans) = 0;
% vars_hat_hmm_varying_states = vars_hat(varsKEEP,:,:);
% explained_variance_varying_states = explained_variance;
% 
% figure()
% for var = feature_groupings(2):feature_groupings(3)
% 
%     % Plot correlation between HMM runs with varied states (for intelligence variables)
%     subplot(6,6,var - feature_groupings(2) + 1)
%     imagesc(corr(squeeze(vars_hat_hmm_varying_states(:,var,:))))
%     
%     % Note p-values and r^2 for figure labels
%     ev_var = explained_variance_varying_states(var,:);
% 
%     % Format chart
%     colorbar()
%     set(gca,'xticklabel',{'r_1','r_2','r_3','r_4','r_5'});
%     set(gca,'yticklabel',{'r_1','r_2','r_3','r_4','r_5'});
%     set(gcf, 'Name', 'FC-HMM: 5 HMM runs with varying K (3-15)')
%     title(sprintf("Feature %d",i))
%     set(gca,'xticklabel',{sprintf("r_1=%.2f",ev_var(1)),sprintf("r_2=%.2f",...
%         ev_var(2)),sprintf("r_3=%.2f",ev_var(3)),sprintf("r_4=%.2f",ev_var(4)),sprintf("r_5=%.2f",ev_var(5))});
% 
%     % Change axes font colours to highlight higher/lower correlations
%     if sum(ev_var) > 0.3
%         set(gca, 'XColor', [0 0.5 0],'YColor', [0 0.5 0]);
%     elseif sum(ev_var) < 0.15
%         set(gca, 'XColor', [1 0 0], 'YColor', [1 0 0]);
%     else 
%         set(gca,'fontweight','bold','XColor', [0 0 0]);
%     end
% end







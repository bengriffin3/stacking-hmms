%% Assessing correlation between predictions (consistent no. of states)
% Load hmm predictions
clear;clc;
DirResults = ['Dokumenter/MATLAB/HMMMAR/HMMMAR Results/'];
DirFC = ['FC_HMM_zeromean_1_covtype_full/'];
load([DirResults DirFC 'HMM_predictions.mat'])
explained_variance_hmm = explained_variance;

% Clean hmm predictions (remove subject where we have NaNs in predictions)
vars_store = cell(3,1);
for rep = 1:size(explained_variance_hmm,2)
    X = vars_hat(:,:,rep);
    idx_nans = any(isnan(X), 2);
    X(idx_nans, :) = [];
    vars_store{rep} = X;
end
vars_hat_hmm = cell2mat(reshape(vars_store,1,1,[]));

% Load and clean static predictions
load('FC_HMM_zeromean_1_covtype_full/staticFC_predictions.mat')
explained_variance_static = explained_variance;
vars_hat(idx_nans, :) = [];
vars_hat_static = vars_hat;

%% Calculate p-values for hmm predictions
% Initialise variables
p_values = zeros(size(explained_variance_hmm));
p_values_sig = zeros(size(explained_variance_hmm));

% Calculate p-values
N = size(vars_hat,1);  % number of observations
for rep = 1:3
    for i = 1:length(explained_variance_hmm)
        r = explained_variance_hmm(i,rep); % do we need to square root?
        t = r*sqrt((N-2)/(1-r^2));
        p_values(i,rep) = 1 - tcdf(t,(N-2));
    end
end

% Convert p-values to asterisks (to add to MATLAB figures)
p_values_sig(p_values < 0.05) = 1;
p_values_sig(p_values > 0.05) = 0;
s1 =string(p_values_sig);
s2 = strrep(s1, '1', '*');
p_value_sig = strrep(s2, '0', '');


%% Calculate correlations between HMM runs (for intelligence features)
% Note indices of intelligence features
load('type_beh.mat')
load('feature_groupings.mat')

% Plot correlation between HMM runs (for intelligence variables)
figure()
for i = feature_groupings(2):feature_groupings(3)
    subplot(6,6,i - feature_groupings(2) + 1)
    X = squeeze(vars_hat_hmm(:,i,:));
    X(idx_nans,:) = [];
    imagesc(corr(X))

    % Note p-values and r^2 for figure labels
    p_value_string = p_value_sig(i,:);
    X_exp_var = explained_variance_hmm(i,:);

    % Format chart
    colorbar()
    title(sprintf("Feature %d",i))
    set(gca,'yticklabel',{'r_1','r_2','r_3'});
    set(gcf, 'Name', 'FC-HMM: 8 States')
    set(gca,'xticklabel',{strcat(sprintf("r_1=%.2f",X_exp_var(1)),p_value_string(1)),strcat(sprintf("r_2=%.2f",...
        X_exp_var(2)),p_value_string(2)),strcat(sprintf("r_3=%.2f",X_exp_var(3)),p_value_string(3))});

    % Change axes font colours to highlight higher/lower correlations
    if sum(X_exp_var) > 0.3
        set(gca, 'XColor', [0 0.5 0],'YColor', [0 0.5 0]);
    elseif sum(X_exp_var) < 0.15
        set(gca, 'XColor', [1 0 0], 'YColor', [1 0 0]);
    else set(gca,'fontweight','bold','XColor', [0 0 0]);%[0.8500, 0.3250, 0.0980])
    end

end




%% Checking correlation for each variable/trait (HMM COMP) VARYING STATES
load FC_HMM_zeromean_1_covtype_full_vary_states/HMM_predictions.mat
vars_hat_hmm_varying_states = vars_hat;
explained_variance_varying_states = explained_variance;

% Note indices of intelligence features
load('type_beh.mat')
[GC,GR] = groupcounts(type_beh');
feature_groupings = cumsum(GC);

% Plot correlation between HMM runs with varied states (for intelligence variables)
figure()
for i = feature_groupings(1):feature_groupings(2)
    subplot(6,6,i - feature_groupings(1) + 1)

    X = squeeze(vars_hat_hmm_varying_states(:,i,:));
    X(any(isnan(X), 2), :) = [];
    X_exp_var = explained_variance_varying_states(i,:);
    imagesc(corr(X))

    % Format chart
    colorbar()
    set(gca,'xticklabel',{'r_1','r_2','r_3'});
    set(gca,'yticklabel',{'r_1','r_2','r_3'});
    set(gcf, 'Name', 'FC-HMM: Varying States')
    title(sprintf("Feature %d",i))
    set(gca,'xticklabel',{sprintf("r_1=%.2f",X_exp_var(1)),sprintf("r_2=%.2f",...
        X_exp_var(2)),sprintf("r_3=%.2f",X_exp_var(3))});

    % Change axes font colours to highlight higher/lower correlations
    if sum(X_exp_var) > 0.3
        set(gca, 'XColor', [0 0.5 0],'YColor', [0 0.5 0]);
    elseif sum(X_exp_var) < 0.15
        set(gca, 'XColor', [1 0 0], 'YColor', [1 0 0]);
    else 
        set(gca,'fontweight','bold','XColor', [0 0 0]);
    end
end

%% Combining predictors
% Now we want to see if we can combine predictors in order to make an even 
% better predictor. To do this, let's examine only the intelligence
% variables
load 'Behavioural Variables/vars_clean.mat'
vars_clean_2 = vars_clean(~idx_nans,:);
vars_clean_intelligence = vars_clean_2(:,feature_groupings(1):feature_groupings(2));

vars_hat_hmm_intelligence = vars_hat_hmm(:,feature_groupings(1):feature_groupings(2),:);
vars_hat_static_intelligence = vars_hat_static(:,feature_groupings(1):feature_groupings(2),:);


%% Firstly, we could choose the best predictors
% First: assess best predictor across entire dataset
LSE_1 = sum(squeeze(sum((vars_clean_intelligence - vars_hat_hmm_intelligence).^2))) % calculate MSE
[M,I] = min(LSE_1);
% Result: predictor 1 is the best predictor


% Second: assess best predictor using cross validation
% Note: this involves creating (e.g. for 10-fold validation) 10
% predictions, based on the other 9 sets of data, so I think it will take
% 10 times the amount of time? (which is too much?) 

% Let's say the entire dataset is the learning set, and so have a look at
% how our predictors have done across the dataset, then choose the best
% one (i.e. use v_k where k minimises sum_k (y_n - z_{kn})^2 )
 

% Basic ensemble
% Now we combine our predictors by finding the mean of them and check error
vars_hat_combine = mean(vars_hat_hmm_intelligence,3);
LSE_2 = sum(squeeze(sum((vars_clean_intelligence - vars_hat_combine).^2))) % calculate MSE
% Result: our predictor is slighty worse than the second best predictor


%%
% Weighted ensemble
% Now we randomly assign weights to our predictors to see if we can find a
% better combination (e.g. put a larger weight on the best predictor)
w = [0.6 0.2 0.2];
w_vec = [0:0.1:1];
vars_hat_combine_weighted = (w(1)*vars_hat_hmm_intelligence(:,:,1) + w(2)*vars_hat_hmm_intelligence(:,:,2) + w(3)*vars_hat_hmm_intelligence(:,:,3));
LSE_3 = sum(squeeze(sum((vars_clean_intelligence - vars_hat_combine_weighted).^2))); % calculate MSE
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




















%% ARCHIVED CODE



% figure
% imagesc(corr(X))
% colorbar
% figure
% imagesc(corr(squeeze(vars_hat_hmm(1,:,:)))); % test for 1 subject
% correlation between static and hmm

% figure
% imagesc(corr(Y))
% colorbar()
% % Result: Still correlated

% %% Checking correlation for intelligence variables/traits (INC STATIC)
% % We show here that for the intelligence variables, the static predictions
% % are relatively different to the hmm predictions (note that the hmm
% % preditions are more frequently similar but sometimes are quite different)
% vars_hat_all = cat(3,vars_hat_hmm_clean, vars_hat_static_clean); % concatenate HMM and static predictions
% explained_variance_all = cat(2,round(explained_variance_hmm,2), round(explained_variance_static,2))
% 
% % Get the indices of the intelligence variables
% load('type_beh.mat')
% [GC,GR] = groupcounts(type_beh');
% cumsum_groups = cumsum(GC);
% 
% % Plot, for each intelligence variable, the correlation between each set of
% % predictions (averaged out over all subjects)
% figure()
% for i = cumsum_groups(1):cumsum_groups(2)
%     subplot(6,6,i - cumsum_groups(1) + 1)
%     X = squeeze(vars_hat_all(:,i,:));
%     X_exp_var = explained_variance_all(i,:)
%     X(any(isnan(X), 2), :) = [];
%     imagesc(corr(X))
%     colorbar()
%     %set(gca,'xticklabel',{'hmm_1','hmm_2','hmm_3','hmm_4','hmm_5','Static'});
%     %set(gca,'yticklabel',{'hmm_1','hmm_2','hmm_3','hmm_4','hmm_5','Static'});
%     set(gca,'xticklabel',{sprintf("%f",X_exp_var(1)),sprintf("%f",X_exp_var(2)),sprintf("%f",X_exp_var(3)),sprintf("%f",X_exp_var(4))});
%     set(gca,'yticklabel',{'hmm_1','hmm_2','hmm_3','Static'});
%     title(sprintf("Feature %d",i))
% end
%
% figure()
% for i = cumsum_groups(1):cumsum_groups(2)
%     subplot(6,6,i - cumsum_groups(1) + 1)
%     X = squeeze(Y(:,i,:));
%     X(any(isnan(X), 2), :) = [];
%     corrplot(X)
% end
% Assessing correlation between predictions (varying states)
% Average across all subjects

% % Correlation between the 5 predictions of 20 subjects
% figure
% load HMM_predictions_20_subjects_varying_states.mat
% X = squeeze(mean(vars_hat_hmm,1));
% X(any(isnan(X), 2), :) = [];
% imagesc(corr(X))
% colorbar()
% % Result: Extremely correlated

%% correlation between static and hmm
% figure
% load staticFC_predictions_20_subjects_varying_states.mat
% Y = cat(3,vars_hat_hmm, vars_hat_static);
% Y = squeeze(mean(Y,1));
% Y(any(isnan(Y), 2), :) = [];
% imagesc(corr(Y))
% colorbar()
% % Result: Still correlated

% %% Checking correlation for each variable/trait (INC STATIC)
% load FC_HMM_zeromean_1_covtype_full_vary_states/HMM_predictions.mat
% vars_hat_hmm_varying_states = vars_hat;
% 
% %Y = cat(3,vars_hat_hmm_varying_states, vars_hat_static); % concatenate HMM and static predictions
% Y = cat(3,vars_hat_hmm_varying_states, vars_hat_static); % concatenate HMM and static predictions
% vars_hat_all = cat(3,vars_hat_hmm, vars_hat_static); % concatenate HMM and static predictions
% explained_variance_all = cat(2,round(explained_variance_hmm,2), round(explained_variance_static,2));
% % Get the indices of the intelligence variables
% % load('type_beh.mat')
% % [GC,GR] = groupcounts(type_beh');
% % cumsum_groups = cumsum(GC);
% 
% % Plot, for each intelligence variable, the correlation between each set of
% % predictions (averaged out over all subjects) but this time we have varied
% % states
% figure()
% for i = cumsum_groups(1):cumsum_groups(2)
%     subplot(6,6,i - cumsum_groups(1) + 1)
% 
%     X = squeeze(Y(:,i,:));
%     X(any(isnan(X), 2), :) = [];
%     imagesc(corr(X))
%     colorbar()
%     %set(gca,'xticklabel',{'hmm_1','hmm_2','hmm_3','hmm_4','hmm_5','Static'});
%     %set(gca,'yticklabel',{'hmm_1','hmm_2','hmm_3','hmm_4','hmm_5','Static'});
%     set(gca,'xticklabel',{'hmm_1','hmm_2','hmm_3','Static'});
%     set(gca,'yticklabel',{'hmm_1','hmm_2','hmm_3','Static'});
% end

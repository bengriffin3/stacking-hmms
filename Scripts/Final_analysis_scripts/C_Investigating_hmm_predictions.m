%% Assessing correlation between predictions (consistent no. of states)
% Load hmm predictions
clear;clc;

DirResults = 'C:\Users\au699373\OneDrive - Aarhus Universitet\Dokumenter\MATLAB\HMMMAR_BG\HMMMAR Results\Changing_models\';
DirSame = 'FC_HMM_zeromean_1_covtype_full\'; % set directory for same states repetitions
DirVarying = 'zeromean_1_covtype_full_vary_states/'; % set directory for varying states repetitions
load([DirResults DirSame 'HMM_predictions.mat'])

% Clean hmm predictions (remove subject where we have NaNs in predictions)
idx_nans = any(isnan(vars_hat(:,:,1)), 2);
subKEEP = true(size(vars_hat,1),1);
subKEEP(idx_nans) = 0;
% select variables to explore
varsKEEP = 11:44;
vars_hat_hmm = vars_hat(subKEEP,varsKEEP,:); 
explained_variance_hmm = explained_variance(varsKEEP,:);

% Load and clean hmm predictions (varying states)
load([DirResults DirVarying 'HMM_predictions.mat'])
vars_hat_hmm_varying_states = vars_hat(subKEEP,varsKEEP,:);
explained_variance_varying_states = explained_variance(varsKEEP,:);

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
    for v = 1:size(explained_variance_hmm,1)

        % Calculate p-values for the explained variance values
         r = explained_variance_store(v,rep); % do we need to square root?
         t = r*sqrt((N-2)/(1-r^2));
         p_values_store(v,rep) = 1 - tcdf(t,(N-2));

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

for v = 1:34%feature_vec
    p = p + 1;

    % Plot correlation between HMM runs (for intelligence variable)
    subplot(ceil(sqrt(length(feature_vec))),ceil(sqrt(length(feature_vec))),p)
    imagesc(corr(squeeze(vars_hat_plot(:,v,:))))
    %sgtitle('FC-HMM: 5 HMM runs with varying States')

    % Note p-values and r^2 for figure labels
    p_var = p_value_plot(v,:);
    ev_var = explained_variance_plot(v,:);

    % Format chart
    colorbar()
    title(sprintf("Feature %d",v))
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



%% Plot the total r^2 across all intelligence variables by no. states
DirMultiRep = 'zeromean_1_covtype_full_vary_states_3_reps\'; % set directory for multiple reps of varying states repetitions
load([DirResults DirMultiRep 'HMM_predictions.mat'])
figure()
bar(mean(explained_variance,1)')
xlabel('No. states'); ylabel('Mean r^2 (across 34 variables)')
title('Mean r^2 across intelligence variables by no. states of HMM runs')
%K_vec = {'3','3','3','4','4','4','5','5','5','6','6','6','7','7','7','8','8','8','9','9','9','10','10','10','11','11','11','12','12','12','13','13','13','14','14','14','15','15','15','16','16','16','17','17','17'};
K_vec = {'3','3','3','5','5','5','8','8','8','13','13','13','15','15','15','18','18','18','23','23','23','25','25','25','28','28','28','33','33','33'};
set(gca, 'XTick', 1:length(K_vec),'XTickLabel',K_vec);
 

%% Plot the total r^2 across all intelligence variables by no. states (mean across repetitions)
error_vec_plot = NaN(10,1);
mean_ev_across_vars = mean(explained_variance(:,1:end));
for i = 1:10
    error_vec_plot(i) = mean(mean_ev_across_vars(3*i-2:3*i));
end
K_vec_plot = [3 5 8 13 15 18 23 25 28 33];
figure()
bar(K_vec_plot,error_vec_plot)
xlabel('Number of states'); ylabel('Mean explained variance');
title('Mean explained variance by number of states (across 34 variables)')





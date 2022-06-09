%% Plot dirichlet_diag investigation figures
DirOut = 'C:\Users\au699373\OneDrive - Aarhus Universitet\Dokumenter\MATLAB\HMMMAR_BG\HMMMAR Results\FEB_2022\K_prior_grid_search_4\';  
load([DirOut 'HMM_predictions'])
hmm_train_dirichlet_diag_vec = [10 10000 100000000 100000000000 100000000000000 100000000000000000 100000000000000000000 100000000000000000000000];

% DirOut = 'C:\Users\au699373\OneDrive - Aarhus Universitet\Dokumenter\MATLAB\HMMMAR_BG\HMMMAR Results\FEB_2022\TDE_runs\zeromean_1\9_20\';
% load([DirOut 'HMM_predictions_9_20.mat'])
% lags = [3 7 9 15];
% explained_variance = explained_variance_9_20(:,3:5,:);
% vars_hat = vars_hat_9_20(:,:,3:5,:);
n_vars = size(explained_variance,1);
n_dif_states = size(explained_variance,2);
n_dif_lags = size(explained_variance,3);
n_HMMs = n_dif_lags*n_dif_states;

%%% Explained variance for each intelligence variable as dirichlet_diag varies
figure;
for i = 1:length(hmm_train_dirichlet_diag_vec) %length(lags) 
    X = 1:n_vars;
    Y = explained_variance(:,:,i);
    subplot(3,3,i);
    scatter(X,Y)
    xlabel('Intelligence Variable'); ylabel('Explained variance, r^2')
    title(sprintf('Dirichlet diag = %i', hmm_train_dirichlet_diag_vec(i)))
    %title(sprintf('Lags = %i', lags(i)))
    %legend('3 states', '5 states', '7 states', '9 states', '11 states', '13 states', '15 states')
    legend('3 states', '9 states', '15 states')
end
sgtitle('How do predictions change as we change dirichlet diag?')
%sgtitle('How do predictions change as we change number of lags?')


%%% Bar chart of total explained variance for changing states and dirichlet_diag
figure;
X = 1:3;
Y = squeeze(sum(explained_variance,1));
bar(X,Y/34)
xlabel('Number of states'), ylabel('Mean explained variance')
legend('DD 10','DD 100','DD 1000','DD 10000', 'DD 100000000', 'DD 10000000000', 'DD 10000000000000', 'DD 100000000000000000')
%legend('Lags 3','Lags 7','Lags 9','Lags 15')
title('Mean explained variance across 34 variables for changing dirichlet diag')
%title('Mean explained variance across 34 variables for changing number of lags')
%xticklabels({'3','6','7','9','11','13','15'})
xticklabels({'3','9','15'})



%% Correlation between repetitions (as shown by largest eigenvalues) and accuracy of predictions
e=zeros(n_vars,1);
for i=1:n_vars
    vars_hat_reshape = reshape(vars_hat(:,i,:,:),[1001  n_HMMs]);
    %vars_hat_reshape = vars_hat(:,:,1:19);
    [~,~,eig_vars_hat] = pca(vars_hat_reshape);
    cumsum_eig = (cumsum(eig_vars_hat)/sum(eig_vars_hat))';
    e(i) = cumsum_eig(1);
end
fig = figure;
left_color = [.5 .5 0];
right_color = [0 .5 .5];
set(fig,'defaultAxesColorOrder',[[0 0.4470 0.7410]; [0 0 0]]);

yyaxis left
eig_vars_hat = bar(e);
eig_vars_hat.FaceColor = [0.3010 0.7450 0.9330];
ylabel('Var explained by largest eig (measure of corr between runs of HMM')
yyaxis right
scatter(1:n_vars,mean(reshape(explained_variance,[n_vars n_HMMs]),2),'x');
%scatter(1:n_vars,mean(explained_variance(:,19),2),'k','x');
[B,I] = sort(e);
xlabel('Intelligence variable')
ylabel('Accuracy of predictions')
title('Corr between repetitions vs accuracy of predictions')


%%
clc
for i=1:n_vars
    vars_hat_reshape = reshape(vars_hat(:,i,:,:),[1001  n_HMMs]);
    %vars_hat_reshape = vars_hat(:,:,19);
    corr_vars_hat_reshape = corr(vars_hat_reshape,'rows','complete');
    X = mean(corr_vars_hat_reshape);
    exp_var = reshape(explained_variance,[n_vars n_HMMs]);
    %exp_var = explained_variance(:,19);
    Y = exp_var(i,:);
    
    %%% Accuracy of rep x vs correlation of rep x with all other reps
    figure(4);
    subplot(6,6,i)
    scatter(X,Y)
    xlabel('mean corr rep x and all over reps')
    ylabel('Accuracy of repetitions')
    hold on
    sgtitle('Accuracy of rep x vs correlation of rep x with all other reps')
    plot([min(X),max(X)],[min(Y),max(Y)])
    
    %%% Correlation between HMM repetitions for 34 intelligence variables
    figure(5);
    subplot(6,6,i)
    imagesc(corr_vars_hat_reshape); colorbar;
    xlabel('Repetition i')
    ylabel('Repetition j')
    sgtitle('Correlation between HMM repetitions for 34 intelligence variables')

    
end


%% Let's plot the gamma's to see if changing the dirichlet diag has made the subjects stay in states much longer
% Dirichlet_diag only (because these plots won't change based on lags)
DirOut = 'C:\Users\au699373\OneDrive - Aarhus Universitet\Dokumenter\MATLAB\HMMMAR_BG\HMMMAR Results\FEB_2022\K_prior_grid_search_4\'; 
r = 2;

figure;
for d = 1:8
    load([DirOut 'HMMs_r' num2str(r) '_d' num2str(d) '_GROUP.mat'],'Gamma')
    subplot(8,1,d); area(Gamma(1:10000,:))
    title(sprintf('DD = %i',hmm_train_dirichlet_diag_vec(d)))
end


%% Plot lags investigation figures
DirOut = 'C:\Users\au699373\OneDrive - Aarhus Universitet\Dokumenter\MATLAB\HMMMAR_BG\HMMMAR Results\FEB_2022\TDE_runs\zeromean_0\';  
load([DirOut 'HMM_predictions'])
lags = [3 7 9 15];

n_vars = size(explained_variance,1);
n_dif_states = size(explained_variance,2);
n_dif_lags = size(explained_variance,3);
n_HMMs = n_dif_lags*n_dif_states;

%%% Explained variance for each intelligence variable as dirichlet_diag varies
figure;
for i = 1:length(lags) 
    X = 1:n_vars;
    Y = explained_variance(:,:,i);
    subplot(2,2,i);
    scatter(X,Y)
    xlabel('Intelligence Variable'); ylabel('Explained variance, r^2')
    title(sprintf('Lags = %i', lags(i)))
    legend('3 states','6 states','9 states', '12 states', '15 states')
end
sgtitle('How do predictions change as we change number of lags?')


%%% Bar chart of total explained variance for changing states and dirichlet_diag
figure;
X = 1:3;
Y = squeeze(sum(explained_variance,1));
bar(X,Y/34)
xlabel('Number of states'), ylabel('Mean explained variance')
legend('Lags 3','Lags 7','Lags 9','Lags 15')
title('Mean explained variance across 34 variables for changing number of lags')
xticklabels({'3','6','9','12','15'})



%% Correlation between repetitions (as shown by largest eigenvalues) and accuracy of predictions
e=zeros(n_vars,1);
for i=1:n_vars
    vars_hat_reshape = reshape(vars_hat(:,i,:,:),[1001  n_HMMs]);
    %vars_hat_reshape = vars_hat(:,:,1:19);
    [~,~,eig_vars_hat] = pca(vars_hat_reshape);
    cumsum_eig = (cumsum(eig_vars_hat)/sum(eig_vars_hat))';
    e(i) = cumsum_eig(1);
end
fig = figure;
left_color = [.5 .5 0];
right_color = [0 .5 .5];
set(fig,'defaultAxesColorOrder',[[0 0.4470 0.7410]; [0 0 0]]);

yyaxis left
eig_vars_hat = bar(e);
eig_vars_hat.FaceColor = [0.3010 0.7450 0.9330];
ylabel('Var explained by largest eig (measure of corr between runs of HMM')
yyaxis right
scatter(1:n_vars,mean(reshape(explained_variance,[n_vars n_HMMs]),2),'x');
%scatter(1:n_vars,mean(explained_variance(:,19),2),'k','x');
[B,I] = sort(e);
xlabel('Intelligence variable')
ylabel('Accuracy of predictions')
title('Corr between repetitions vs accuracy of predictions')


%%
clc
for i=1:n_vars
    vars_hat_reshape = reshape(vars_hat(:,i,:,:),[1001  n_HMMs]);
    %vars_hat_reshape = vars_hat(:,:,19);
    corr_vars_hat_reshape = corr(vars_hat_reshape,'rows','complete');
    X = mean(corr_vars_hat_reshape);
    exp_var = reshape(explained_variance,[n_vars n_HMMs]);
    %exp_var = explained_variance(:,19);
    Y = exp_var(i,:);
    
    %%% Accuracy of rep x vs correlation of rep x with all other reps
    figure(4);
    subplot(6,6,i)
    scatter(X,Y)
    xlabel('mean corr rep x and all over reps')
    ylabel('Accuracy of repetitions')
    hold on
    sgtitle('Accuracy of rep x vs correlation of rep x with all other reps')
    plot([min(X),max(X)],[min(Y),max(Y)])
    
    %%% Correlation between HMM repetitions for 34 intelligence variables
    figure(5);
    subplot(6,6,i)
    imagesc(corr_vars_hat_reshape); colorbar;
    xlabel('Repetition i')
    ylabel('Repetition j')
    sgtitle('Correlation between HMM repetitions for 34 intelligence variables')

    
end


%% Let's plot the gamma's to see if changing the dirichlet diag has made the subjects stay in states much longer
% Dirichlet_diag only (because these plots won't change based on lags)
DirOut = 'C:\Users\au699373\OneDrive - Aarhus Universitet\Dokumenter\MATLAB\HMMMAR_BG\HMMMAR Results\FEB_2022\K_prior_grid_search_4\'; 
r = 2;

figure;
for d = 1:8
    load([DirOut 'HMMs_r' num2str(r) '_d' num2str(d) '_GROUP.mat'],'Gamma')
    subplot(8,1,d); area(Gamma(1:10000,:))
    title(sprintf('DD = %i',hmm_train_dirichlet_diag_vec(d)))
end

%%








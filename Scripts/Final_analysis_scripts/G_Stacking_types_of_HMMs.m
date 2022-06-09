%% Stacking weight heatmaps
DirOut = 'C:\Users\au699373\OneDrive - Aarhus Universitet\Dokumenter\MATLAB\HMMMAR_BG\HMMMAR Results\FEB_2022\TDE_runs\';
load([DirOut 'HMM_predictions_stack_all.mat'])
n_HMMs = 40;%size(explained_variance,2)*size(explained_variance,3);
% Least squared heat map of weights
figure;
imagesc((mean(W_stack_ls,3))'); colorbar;
xlabel('Intelligence Variable'); ylabel('zeromean 0 <- HMM repetition -> zeromean 1');
yticklabels({'K = 3, l = 1','K = 3, l = 3','K = 3, l = 9','K = 3, l = 15','K = 6, l = 1','K = 6, l = 3','K = 6, l = 9','K = 6, l = 15'...
    'K = 9, l = 1','K = 9, l = 3','K = 9, l = 9','K = 9, l = 15','K = 12, l = 1','K = 12, l = 3','K = 12, l = 9','K = 12, l = 15','K = 15, l = 1','K = 15, l = 3','K = 15, l = 9','K = 15, l = 15'...
    ,'K = 3, l = 1','K = 3, l = 3','K = 3, l = 9','K = 3, l = 15','K = 6, l = 1','K = 6, l = 3','K = 6, l = 9','K = 6, l = 15'...
    'K = 9, l = 1','K = 9, l = 3','K = 9, l = 9','K = 9, l = 15','K = 12, l = 1','K = 12, l = 3','K = 12, l = 9','K = 12, l = 15','K = 15, l = 1','K = 15, l = 3','K = 15, l = 9','K = 15, l = 15'})
ytickangle(45)
set(gca,'ytick',1:n_HMMs);
set(gca,'fontsize',20);

% Ridge Regression heat map of weights
figure;
imagesc((mean(W_stack_rdg,3))'); colorbar; caxis([-1 1]);
xlabel('Intelligence Variable'); ylabel('zeromean 0 <- HMM repetition -> zeromean 1');
yticklabels({'K = 3, l = 1','K = 3, l = 3','K = 3, l = 9','K = 3, l = 15','K = 6, l = 1','K = 6, l = 3','K = 6, l = 9','K = 6, l = 15'...
    'K = 9, l = 1','K = 9, l = 3','K = 9, l = 9','K = 9, l = 15','K = 12, l = 1','K = 12, l = 3','K = 12, l = 9','K = 12, l = 15','K = 15, l = 1','K = 15, l = 3','K = 15, l = 9','K = 15, l = 15'...
    ,'K = 3, l = 1','K = 3, l = 3','K = 3, l = 9','K = 3, l = 15','K = 6, l = 1','K = 6, l = 3','K = 6, l = 9','K = 6, l = 15'...
    'K = 9, l = 1','K = 9, l = 3','K = 9, l = 9','K = 9, l = 15','K = 12, l = 1','K = 12, l = 3','K = 12, l = 9','K = 12, l = 15','K = 15, l = 1','K = 15, l = 3','K = 15, l = 9','K = 15, l = 15'})
ytickangle(45)
set(gca,'ytick',1:n_HMMs);
set(gca,'fontsize',20);





%% Plot dirichlet_diag investigation figures
clear; clc;

% Let's plot the gamma's to see if changing the dirichlet diag has made the subjects stay in states much longer
% Dirichlet_diag only (because these plots won't change based on lags)
DirOut = 'C:\Users\au699373\OneDrive - Aarhus Universitet\Dokumenter\MATLAB\HMMMAR_BG\HMMMAR Results\FEB_2022\K_prior_grid_search_5_with_folds\Zeromean_0\'; 
hmm_train_dirichlet_diag_vec = [10 10000000 1.0000e+10 1.0000e+13];
%r = 3; % change to explore different number of states
figure;
reps_explore = 1:4; % choose which reps to explore (1-4 -> K = 3, 5-8 -> K = 3, 9-12 -> K = 3, 13-16 -> K = 3,
for d = reps_explore
    r = d;
    load([DirOut 'HMMs_r' num2str(r) '_d' num2str(d) '_GROUP.mat'],'Gamma')
    subplot(4,1,d); area(Gamma(1:10000,:))
    title(sprintf('DD = %i',hmm_train_dirichlet_diag_vec(d)))
end


%% Mean and per trait explained variance by state and delays/DD 
%DirOut = 'C:\Users\au699373\OneDrive - Aarhus Universitet\Dokumenter\MATLAB\HMMMAR_BG\HMMMAR Results\FEB_2022\TDE_runs\zeromean_1\';
DirOut = 'C:\Users\au699373\OneDrive - Aarhus Universitet\Dokumenter\MATLAB\HMMMAR_BG\HMMMAR Results\FEB_2022\K_prior_grid_search_5_with_folds\Zeromean_1\'; 
load([DirOut 'HMM_predictions'])
states = [3 7 11 15];

n_vars = size(explained_variance,1);
n_dif_states = size(explained_variance,2);
n_dif_lags = size(explained_variance,3);
n_HMMs = n_dif_lags*n_dif_states;

%%% Explained variance for each intelligence variable as dirichlet_diag varies
figure;
for i = 1:length(hmm_train_dirichlet_diag_vec)
    X = 1:n_vars;
    Y = explained_variance(:,:,i);
    subplot(3,3,i);
    scatter(X,Y)
    xlabel('Intelligence Variable'); ylabel('Explained variance, r^2')
    title(sprintf('Dirichlet diag = %i', hmm_train_dirichlet_diag_vec(i)))
    legend('3 states', '9 states', '15 states')
end
sgtitle('How do predictions change as we change dirichlet diag?')


%%% Bar chart of total explained variance for changing states and dirichlet_diag
figure;
X = 1:length(states);
Y = squeeze(sum(explained_variance,1));
bar(X,Y/34)
xlabel('Number of states'), ylabel('Mean explained variance')
legend('DD 10', 'DD 10000', 'DD 100000000', 'DD 100000000000', 'DD 100000000000000', 'DD 100000000000000000', 'DD 100000000000000000000', 'DD 100000000000000000000000')
title('Mean explained variance across 34 variables for changing dirichlet diag')
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
    
%     %%% Accuracy of rep x vs correlation of rep x with all other reps
%     figure(100);
%     subplot(6,6,i)
%     scatter(X,Y)
%     xlabel('mean corr rep x and all over reps')
%     ylabel('Accuracy of repetitions')
%     hold on
%     sgtitle('Accuracy of rep x vs correlation of rep x with all other reps')
%     %plot([min(X),max(X)],[min(Y),max(Y)])
%     plot([0,1],[0,0.2])
%     xlim([0 1])
%     ylim([0 0.2])

    %%% Correlation between HMM repetitions for 34 intelligence variables
    figure(101);
    subplot(6,6,i)
    imagesc(corr_vars_hat_reshape); colorbar;
    xlabel('Repetition i')
    ylabel('Repetition j')
    sgtitle('Correlation between HMM repetitions for 34 intelligence variables')

    
end

%% Let's repeat this but for the best 5 predictions

for i = 1:34
    
    vars_hat_reshape = reshape(vars_hat(:,i,:,:),[1001  n_HMMs]);
    [M,I] = maxk(exp_var(i,:)',5);
    corr_vars_hat_reshape = corr(vars_hat_reshape(:,I),'rows','complete');
    X = mean(corr_vars_hat_reshape);
    exp_var = reshape(explained_variance,[n_vars n_HMMs]);
    Y = exp_var(i,I);

    %%% Accuracy of rep x vs correlation of rep x with all other reps
    figure(200);
    subplot(6,6,i)
    scatter(X,Y)
    xlabel('mean corr rep x and all over reps')
    ylabel('Accuracy of repetitions')
    hold on
    sgtitle('Accuracy of rep x vs correlation of rep x with all other reps')
    plot([0,1],[0,0.2])
    xlim([0 1])
    ylim([0 0.2])

    %%% Correlation between HMM repetitions for 34 intelligence variables
    figure(201);
    subplot(6,6,i)
    imagesc(corr_vars_hat_reshape); colorbar
    sgtitle('Correlation between{\it best} 5 HMM repetitions for 34 intelligence variables')

    ax=gca;
    ax.XTickLabel = num2cell(round(M,3));
    ax.YTickLabel = num2cell(round(M,3));
    xtickangle(45)
    ytickangle(45)
    set(gca,'FontSize',8)
    title(sprintf('Intelligence Variable %i',i))
    xlabel('Explained Variance of repetition')
    ylabel('Explained Variance of repetition')

end



%% Plot lags investigation figures
DirOut = 'C:\Users\au699373\OneDrive - Aarhus Universitet\Dokumenter\MATLAB\HMMMAR_BG\HMMMAR Results\FEB_2022\TDE_runs\'; % Test folder
load([DirOut 'zeromean_1\HMM_predictions.mat'],'vars_hat','explained_variance','folds_all');
explained_variance_all = explained_variance;
vars_hat_all = vars_hat;
load([DirOut 'zeromean_0\\HMM_predictions.mat'],'vars_hat','explained_variance','folds_all');
explained_variance = cat(3,explained_variance_all,explained_variance);
vars_hat = cat(4,vars_hat_all,vars_hat);

%load([DirOut 'HMM_predictions_stack_all.mat'],'vars_hat_stack_ls', 'vars_hat_stack_rdg', 'vars_hat_stack_rf')
states = [3 6 9 12 15];
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
X = 1:length(states);
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
    
%     %%% Accuracy of rep x vs correlation of rep x with all other reps
%     figure(300);
%     subplot(6,6,i)
%     scatter(X,Y)
%     xlabel('mean corr rep x and all over reps')
%     ylabel('Accuracy of repetitions')
%     hold on
%     sgtitle('Accuracy of rep x vs correlation of rep x with all other reps')
%     %plot([min(X),max(X)],[min(Y),max(Y)])
%     plot([0,1],[0,0.2])
%     xlim([0 1])
%     ylim([0 0.2])
% 
%     %%% Correlation between HMM repetitions for 34 intelligence variables
%     figure(301);
%     subplot(6,6,i)
%     imagesc(corr_vars_hat_reshape); colorbar;
%     xlabel('Repetition i')
%     ylabel('Repetition j')
%     sgtitle('Correlation between HMM repetitions for 34 intelligence variables')

    
end

%% Let's repeat this but for the best 5 predictions

for i = 1:34
    
    vars_hat_reshape = reshape(vars_hat(:,i,:,:),[1001  n_HMMs]);
    [M,I] = maxk(exp_var(i,:)',10);
    corr_vars_hat_reshape = corr(vars_hat_reshape(:,I),'rows','complete');
    X = mean(corr_vars_hat_reshape);
    exp_var = reshape(explained_variance,[n_vars n_HMMs]);
    Y = exp_var(i,I);

    %%% Accuracy of rep x vs correlation of rep x with all other reps
    figure(400);
    subplot(6,6,i)
    scatter(X,Y)
    xlabel('mean corr rep x and all over reps')
    ylabel('Accuracy of repetitions')
    hold on
    sgtitle('Accuracy of rep x vs correlation of rep x with all other reps')
    plot([0,1],[0,0.2])
    xlim([0 1])
    ylim([0 0.2])

    %%% Correlation between HMM repetitions for 34 intelligence variables
    figure(401);
    subplot(6,6,i)
    imagesc(corr_vars_hat_reshape); colorbar
    sgtitle('Correlation between{\it best} 5 HMM repetitions for 34 intelligence variables')

    ax=gca;
    ax.XTickLabel = num2cell(round(M,3));
    ax.YTickLabel = num2cell(round(M,3));
    xtickangle(45)
    ytickangle(45)
    set(gca,'FontSize',8)
    title(sprintf('Intelligence Variable %i',i))
    xlabel('Explained Variance of repetition')
    ylabel('Explained Variance of repetition')

end



%% Let's plot the gamma's to see if changing the dirichlet diag has made the subjects stay in states much longer

DirOut = 'C:\Users\au699373\OneDrive - Aarhus Universitet\Dokumenter\MATLAB\HMMMAR_BG\HMMMAR Results\FEB_2022\TDE_runs\zeromean_1\'; 
lags = [3 7 9 15 3 7 9 15 3 7 9 15 3 7 9 15 3 7 9 15];


for i = 5%1:5
    j = (i*4)-3;
    figure;
    for k = j:j+3
        load([DirOut 'HMMs_r' num2str(k) '_d' num2str(k) '_GROUP.mat'],'Gamma')
        subplot(4,1,k-(4*(i-1))); area(Gamma(1:10000,:))
        title(sprintf('Lags = %i',lags(k)))
        sgtitle(sprintf('States %i',states(i)))
    end
end










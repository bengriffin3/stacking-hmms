%% Prediction Analysis
% %DirOut = 'C:\Users\au699373\OneDrive - Aarhus Universitet\Dokumenter\MATLAB\HMMMAR_BG\HMMMAR Results\FEB_2022\K_prior_grid_search_4_SC\'; % Test folder
% %DirOut = 'C:\Users\au699373\OneDrive - Aarhus Universitet\Dokumenter\MATLAB\HMMMAR_BG\HMMMAR Results\FEB_2022\TDE_runs\zeromean_1\Supercomputer_fullset\'; % Test folder
% DirOut = 'C:\Users\au699373\OneDrive - Aarhus Universitet\Dokumenter\MATLAB\HMMMAR_BG\HMMMAR Results\FEB_2022\TDE_runs\zeromean_1\Supercomputer_fullset\'; % Test folder
% load([DirOut 'HMM_predictions.mat'],'vars_hat','explained_variance','mean_squared_error');
% Load predictions to stack
DirOut = 'C:\Users\au699373\OneDrive - Aarhus Universitet\Dokumenter\MATLAB\HMMMAR_BG\HMMMAR Results\FEB_2022\TDE_runs\'; % Test folder
load([DirOut 'zeromean_1\HMM_predictions.mat'],'vars_hat','explained_variance','folds_all');
explained_variance_all = explained_variance;
vars_hat_all = vars_hat;
load([DirOut 'zeromean_0\\HMM_predictions.mat'],'vars_hat','explained_variance','folds_all');
explained_variance = cat(3,explained_variance_all,explained_variance);
vars_hat = cat(4,vars_hat_all,vars_hat);

load([DirOut 'HMM_predictions_stack_all.mat'],'vars_hat_stack_ls', 'vars_hat_stack_rdg', 'vars_hat_stack_rf')


load('vars_target.mat')
n_vars = size(explained_variance,1);
n_subjects = 1001;
n_dif_states = size(explained_variance,2);
n_dif_dirichlet_diag = size(explained_variance,3);
n_HMMs = n_dif_states*n_dif_dirichlet_diag;

explained_variance_ls = NaN(n_vars,1);
explained_variance_rdg = NaN(n_vars,1);
explained_variance_rf = NaN(n_vars,1);
mean_squared_error_ls = NaN(n_vars,1);
mean_squared_error_rdg = NaN(n_vars,1);
mean_squared_error_rf = NaN(n_vars,1);
coefficient_of_determination_ls = NaN(n_vars,1);
coefficient_of_determination_rdg = NaN(n_vars,1);
coefficient_of_determination_rf = NaN(n_vars,1);


for j = 1:size(vars,2)
    % Calculate explained variance
    explained_variance_ls(j) = corr(vars(:,j),vars_hat_stack_ls(:,j),'rows','complete').^2;
    explained_variance_rdg(j) = corr(vars(:,j),vars_hat_stack_rdg(:,j),'rows','complete').^2;
    explained_variance_rf (j)= corr(vars(:,j),vars_hat_stack_rf(:,j),'rows','complete').^2;

    % Calculate mean squared error
    mean_squared_error_ls(j) = sum((vars(:,j) - vars_hat_stack_ls(:,j)).^2,'omitnan')/n_subjects;
    mean_squared_error_rdg(j) = sum((vars(:,j) - vars_hat_stack_rdg(:,j)).^2,'omitnan')/n_subjects;
    mean_squared_error_rf(j) = sum((vars(:,j) - vars_hat_stack_rf(:,j)).^2,'omitnan')/n_subjects;

    % Calculate coefficient of determination
    coefficient_of_determination_ls(j) = 1 - sum((vars(:,j) - vars_hat_stack_ls(:,j)).^2,'omitnan') / sum((vars(:,j) - mean(vars(:,j),'omitnan')).^2,'omitnan');
    coefficient_of_determination_rdg(j) = 1 - sum((vars(:,j) - vars_hat_stack_rdg(:,j)).^2,'omitnan') / sum((vars(:,j) - mean(vars(:,j),'omitnan')).^2,'omitnan');
    coefficient_of_determination_rf(j) = 1 - sum((vars(:,j) - vars_hat_stack_rdg(:,j)).^2,'omitnan') / sum((vars(:,j) - mean(vars(:,j),'omitnan')).^2,'omitnan');
    

end

%% Plot analysis
% Plot explained variance
figure; X = 1:n_vars;
scatter(X,explained_variance_ls,'r'); hold on
scatter(X,explained_variance_rdg,'k');
scatter(X,explained_variance_rf,'g');

% here we plot the original predictions
%hmm_train_dirichlet_diag_vec = [10 10000 10000000 10000000000 10000000000000 10000000000000000 10000000000000000000 10000000000000000000000]; c = linspace(min(hmm_train_dirichlet_diag_vec), max(hmm_train_dirichlet_diag_vec),n_HMMs);
lags = [3 7 9 15]; c = linspace(min(lags), max(lags),n_HMMs); %%% THIS IS FOR NUMBER OF LAGS

explained_variance_plot = reshape(explained_variance,[n_vars n_HMMs]);
for j = 1:n_vars; scatter((j*ones(n_HMMs,1))',explained_variance_plot(j,:),[],c,'x'); hold on; end; colormap winter;
%h = colorbar; ylabel(h, 'Dirichlet diag')
h = colorbar; ylabel(h, 'Number of lags') %%% THIS IS FOR NUMBER OF LAGS
title('Stacking HMM-TDE predictions with varying lags')
legend('Least Squares','Ridge Regression','Random Forests','Original HMMs');
xlabel('Intelligence Feature'); ylabel('Explained Variance, r^2')

% % Plot mean squared error
% figure; X = 1:n_vars;
% scatter(X,mean_squared_error_ls,'r'); hold on
% scatter(X,mean_squared_error_rdg,'k')
% scatter(X,mean_squared_error_rf,'g')
% 
% % here we plot the original predictions
% c = linspace(min(hmm_train_dirichlet_diag_vec), max(hmm_train_dirichlet_diag_vec),n_HMMs);
% mean_squared_error_plot = reshape(mean_squared_error,[n_vars n_HMMs]);
% for j = 1:n_vars; scatter((j*ones(n_HMMs,1))',mean_squared_error_plot(j,:),[],c,'x'); hold on; end; colormap winter;
% h = colorbar; ylabel(h, 'Dirichlet diag')
% legend('Least Squares','Ridge Regression','Random Forests','Original HMMs');
% xlabel('Intelligence Feature'); ylabel('Mean_squared_error, r^2')

%% Stacking weights of models across folds
load([DirOut 'zeromean_1\HMM_predictions_stack.mat']);

for i = 1:9
    figure(500);
    subplot(3,3,i)  
    imagesc(squeeze(W_stack_ls(i,:,:))); colorbar;
    xlabel('Fold Number'); ylabel('HMM repetition');
    sgtitle('Stacking (LS) weights of models, Zeromean = 1');
    title(sprintf('Intelligence Feature %i',i))
    
    figure(501);
    subplot(3,3,i)  
    imagesc(squeeze(W_stack_rdg(i,:,:))); colorbar;
    xlabel('Fold Number'); ylabel('HMM repetition');
    sgtitle('Stacking (RDG) weights of models, Zeromean = 1');
    title(sprintf('Intelligence Feature %i',i))
end
%%
% We can see that the weights are generally quite consistent across models.
% So, let's take the mean across folds so we can compare across
% intelligence featres
W_stack_ls_mean_folds = mean(W_stack_ls,3);
W_stack_rdg_mean_folds = mean(W_stack_rdg,3);
figure;
subplot(1,2,1)
imagesc(W_stack_ls_mean_folds); colorbar;
xlabel('HMM repetition'); ylabel('Intelligence Feature');
sgtitle('Stacking weights of models, Zeromean = 1');
title(sprintf('Least Squares'))
subplot(1,2,2)
imagesc(W_stack_rdg_mean_folds); colorbar;
xlabel('HMM repetition'); ylabel('Intelligence Feature');
title(sprintf('Ridge Regression'))
sgtitle('Stacking weights of models across features');
    
%%

% for i = 1:3
%     figure
%     subplot(3,1,1)
%     bar(W_stack_ls_mean_folds(i,:))
%     subplot(3,1,2)
%     bar(W_stack_rdg_mean_folds(i,:))
%     %subplot(3,1,3)
%     %bar(W_stack_rf_mean_folds(i,:))
% 
% end

%% Master Stack

% See C:\Users\au699373\OneDrive - Aarhus
% Universitet\Dokumenter\MATLAB\HMMMAR_BG\Scripts\NeuroImage2021_stacking_March_2022_V2.m
% for possible different / similar stuff




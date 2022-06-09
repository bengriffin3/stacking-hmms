% % Goal: we want to see if higher entropy means more accurate predictions
% % for a given subject


% Linegraphs, just like scatterplots, display the relationship between two numerical 
% variables. However, it is preferred to use linegraphs over scatterplots when the variable 
% on the x-axis (i.e., the explanatory variable) has an inherent ordering, such as some notion of time.



% 
%% PLOT ACTUAL RELATIONSHIPS BETWEEN PREDICTIVE ACCURACY AND METAFEATURES
% load predictions and metadata
clear; clc;
%DirResults = '/Users/au699373/OneDrive - Aarhus Universitet/Dokumenter/MATLAB/HMMMAR_BG/HMMMAR Results/FC_HMM_zeromean_1_covtype_full_vary_states_63_reps/'; % Test folder
DirResults = 'C:\Users\au699373\OneDrive - Aarhus Universitet\Dokumenter\MATLAB\HMMMAR_BG\HMMMAR Results\FEB_2022\Same_states\';
load([DirResults 'HMM_predictions.mat']) % load predictions
load([DirResults  'HMMs_meta_data_subject']) % load metadata
load('vars_target.mat') % learn targets to predict
rep_selection_vec = [1 2 3 4 5];
K_vec = [3 3 3 6 6 6 9 9 9 12 12 12 15 15 15];
repetitions = length(rep_selection_vec);
Entropy = Entropy_subject_ln(:,rep_selection_vec);
Likelihood = likelihood_subject(:,rep_selection_vec);
% load Gaussianized metadata
load('HMMs_meta_data_subject_gaussianized')


Entropy_gauss = Entropy_gaussianized(:,rep_selection_vec);
Likelihood_gauss = Likelihood_gaussianized(:,rep_selection_vec);

% check our data has been Gaussianized (in R)
for rep = 1:repetitions
    figure()
    subplot(2,1,1)
    histogram(Entropy_subject_ln(:,rep))
    xlabel('Entropy'); ylabel('Frequency');
    title('Entropy')
    subplot(2,1,2)
    histogram(Entropy_gaussianized(:,rep))
    title('Entropy Gaussianized')
    xlabel('Entropy Gaussianized'); ylabel('Frequency');
    sgtitle(sprintf('States = %i',K_vec(rep)))
end

%%
pearson_error_entropy = NaN(size(vars,2),size(Entropy,2));
pvalue_error_entropy = NaN(size(vars,2),size(Entropy,2));
pearson_error_likelihood = NaN(size(vars,2),size(Entropy,2));
pvalue_error_likelihood = NaN(size(vars,2),size(Entropy,2));
spearman_error_entropy = NaN(size(vars,2),size(Entropy,2));
spearman_error_likelihood = NaN(size(vars,2),size(Entropy,2));

pearson_error_entropy_norm = NaN(size(vars,2),size(Entropy,2));
pearson_error_likelihood_norm = NaN(size(vars,2),size(Entropy,2));
spearman_error_entropy_norm = NaN(size(vars,2),size(Entropy,2));
spearman_error_likelihood_norm = NaN(size(vars,2),size(Entropy,2));


pearson_error_entropy_gauss = NaN(size(vars,2),size(Entropy,2));
pearson_error_likelihood_gauss = NaN(size(vars,2),size(Entropy,2));
spearman_error_entropy_gauss = NaN(size(vars,2),size(Entropy,2));
spearman_error_likelihood_gauss = NaN(size(vars,2),size(Entropy,2));

pearson_error_entropy_gauss_norm = NaN(size(vars,2),size(Entropy,2));
pearson_error_likelihood_gauss_norm = NaN(size(vars,2),size(Entropy,2));
spearman_error_entropy_gauss_norm = NaN(size(vars,2),size(Entropy,2));
spearman_error_likelihood_gauss_norm = NaN(size(vars,2),size(Entropy,2));

vars = (vars-min(vars))./(max(vars)-min(vars));
vars_hat = (vars_hat-min(vars_hat))./(max(vars_hat)-min(vars_hat));

figure()
for var = 1:size(vars,2)
    % Remove NaN
    which_nan = isnan(vars(:,var));
    if any(which_nan)
        vars_hat = vars_hat(~which_nan,:,:);
        vars = vars(~which_nan,:);
        Entropy = Entropy(~which_nan,:);
        Entropy_gauss = Entropy_gauss(~which_nan,:);
        Likelihood = Likelihood(~which_nan,:);
        Likelihood_gauss = Likelihood_gauss(~which_nan,:);
        warning('NaN found on Yin, will remove...')
    end
    
    

    Predictions = squeeze(vars_hat(:,var,rep_selection_vec));
    vars_target = vars(:,var);
    Prediction_accuracy = (vars_target - Predictions).^2;
    %explained_variance = corr(Predictions,vars_target).^2;
    vars_norm = (vars-min(vars))./(max(vars)-min(vars));
    vars_hat_norm = (vars_hat-min(vars_hat))./(max(vars_hat)-min(vars_hat));
    Predictions_norm = squeeze(vars_hat_norm(:,var,rep_selection_vec));
    vars_target_norm = vars_norm(:,var);
    Prediction_accuracy_norm = (vars_target_norm - Predictions_norm).^2;
    
    for rep = 1:size(Entropy,2)
        [pearson_error_entropy(var,rep), pvalue_error_entropy(var,rep)] = corr(Prediction_accuracy(:,rep),Entropy(:,rep));
        [pearson_error_likelihood(var,rep), pvalue_error_likelihood(var,rep)] = corr(Prediction_accuracy(:,rep),Likelihood(:,rep));
        [spearman_error_entropy(var,rep),~] = corr(Prediction_accuracy(:,rep),Entropy(:,rep),'Type','Spearman');
        [spearman_error_likelihood(var,rep),~] = corr(Prediction_accuracy(:,rep),Likelihood(:,rep),'Type','Spearman');
        
        [pearson_error_entropy_gauss(var,rep), ~] = corr(Prediction_accuracy(:,rep),Entropy_gauss(:,rep));
        [pearson_error_likelihood_gauss(var,rep),~] = corr(Prediction_accuracy(:,rep),Likelihood_gauss(:,rep));
        [spearman_error_entropy_gauss(var,rep),~] = corr(Prediction_accuracy(:,rep),Entropy_gauss(:,rep),'Type','Spearman');
        [spearman_error_likelihood_gauss(var,rep),~] = corr(Prediction_accuracy(:,rep),Likelihood_gauss(:,rep),'Type','Spearman');
       
        [pearson_error_entropy_norm(var,rep), ~] = corr(Prediction_accuracy_norm(:,rep),Entropy(:,rep));
        [pearson_error_likelihood_norm(var,rep), ~] = corr(Prediction_accuracy_norm(:,rep),Likelihood(:,rep));
        [spearman_error_entropy_norm(var,rep),~] = corr(Prediction_accuracy_norm(:,rep),Entropy(:,rep),'Type','Spearman');
        [spearman_error_likelihood_norm(var,rep),~] = corr(Prediction_accuracy_norm(:,rep),Likelihood(:,rep),'Type','Spearman');
        
        [pearson_error_entropy_gauss_norm(var,rep), ~] = corr(Prediction_accuracy_norm(:,rep),Entropy_gauss(:,rep));
        [pearson_error_likelihood_gauss_norm(var,rep),~] = corr(Prediction_accuracy_norm(:,rep),Likelihood_gauss(:,rep));
        [spearman_error_entropy_gauss_norm(var,rep),~] = corr(Prediction_accuracy_norm(:,rep),Entropy_gauss(:,rep),'Type','Spearman');
        [spearman_error_likelihood_gauss_norm(var,rep),~] = corr(Prediction_accuracy_norm(:,rep),Likelihood_gauss(:,rep),'Type','Spearman');
       
        Entropy_plot = Entropy(:,rep);
        Prediction_accuracy_plot = Prediction_accuracy(:,rep);
        Likelihood_plot = Likelihood(:,rep);
        subplot(6,6,var)
        %scatter(Entropy_plot,Prediction_accuracy_plot);
        scatter(Likelihood_plot,Prediction_accuracy_plot);
        hold on
        %xlim([0 15])
    end
    xlabel('Likelihood')%xlabel('Entropy');%
    ylabel('Prediction error')
    sgtitle('HMM Likelihood to prediction accuracy relationship') 
    title(sprintf('Variable #%d', var));
    %legend('HMM rep 1', 'HMM rep 2', 'HMM rep 3', 'HMM rep 4', 'HMM rep 5')

end
subplot(6,6,35)
plot([0 0 0 0 0 0],[0 0 0 0 0 0; 0 0 0 0 0 0; 0 0 0 0 0 0; 0 0 0 0 0 0; 0 0 0 0 0 0]) %make dummy plot with the right linestyle
axis([10,11,10,11]) %move dummy points out of view
legend('HMM rep 1', 'HMM rep 2', 'HMM rep 3', 'HMM rep 4', 'HMM rep 5')
axis off %hide axis


%%
% Correlation values between prediction error and metafeature - heatmaps
figure
sgtitle('Correlation values between prediction error and metafeature - heatmaps')
subplot(2,2,1); imagesc(pearson_error_entropy);  
title('Pearson''s correlation - entropy'); colorbar; xlabel('HMM Repetition'); ylabel('Intelligence variable')
subplot(2,2,2); imagesc(pearson_error_likelihood); 
title('Pearson''s correlation - likelihood'); colorbar; xlabel('HMM Repetition'); ylabel('Intelligence variable')
subplot(2,2,3); imagesc(spearman_error_entropy);
title('Spearman''s correlation - entropy'); colorbar; xlabel('HMM Repetition'); ylabel('Intelligence variable')
subplot(2,2,4); imagesc(spearman_error_likelihood); 
title('Spearman''s correlation - likelihood'); colorbar; xlabel('HMM Repetition'); ylabel('Intelligence variable')

% Correlation values between prediction error and metafeature - histograms
figure
sgtitle('Correlation values between prediction error and metafeature - histograms') 
subplot(2,2,1); histogram(pearson_error_entropy);
title('Pearson''s correlation - entropy'); xlabel('Correlation value'); ylabel('Frequency')
subplot(2,2,2); histogram(pearson_error_likelihood);
title('Pearson''s correlation - likelihood'); xlabel('Correlation value'); ylabel('Frequency')
subplot(2,2,3); histogram(spearman_error_entropy);
title('Spearman''s correlation - entropy'); xlabel('Correlation value'); ylabel('Frequency')
subplot(2,2,4); histogram(spearman_error_likelihood);
title('Spearman''s correlation - likelihood'); xlabel('Correlation value'); ylabel('Frequency')

% Correlation values between prediction error and metafeature (Gaussianized) - heatmaps
figure
sgtitle('Correlation values between prediction error and metafeature (Gaussianized) - heatmaps')
subplot(2,2,1); imagesc(pearson_error_entropy_gauss);  
title('Pearson''s correlation - entropy'); colorbar; xlabel('HMM Repetition'); ylabel('Intelligence variable')
subplot(2,2,2); imagesc(pearson_error_likelihood_gauss); 
title('Pearson''s correlation - likelihood'); colorbar; xlabel('HMM Repetition'); ylabel('Intelligence variable')
subplot(2,2,3); imagesc(spearman_error_entropy_gauss);
title('Spearman''s correlation - entropy'); colorbar; xlabel('HMM Repetition'); ylabel('Intelligence variable')
subplot(2,2,4); imagesc(spearman_error_likelihood_gauss); 
title('Spearman''s correlation - likelihood'); colorbar; xlabel('HMM Repetition'); ylabel('Intelligence variable')

% Correlation values between prediction error and metafeature  (Gaussianized) - histograms
figure
sgtitle('Correlation between prediction error and metafeature (Gaussianized) for 34 features, 5 HMM repetitions') 
subplot(2,2,1); histogram(pearson_error_entropy_gauss);
title('Pearson''s correlation - entropy'); xlabel('Correlation value'); ylabel('Frequency')
subplot(2,2,2); histogram(pearson_error_likelihood_gauss);
title('Pearson''s correlation - likelihood'); xlabel('Correlation value'); ylabel('Frequency')
subplot(2,2,3); histogram(spearman_error_entropy_gauss);
title('Spearman''s correlation - entropy'); xlabel('Correlation value'); ylabel('Frequency')
subplot(2,2,4); histogram(spearman_error_likelihood_gauss);
title('Spearman''s correlation - likelihood'); xlabel('Correlation value'); ylabel('Frequency')

%%

figure()
for rep = 1:repetitions
    Entropy_plot = Entropy(:,rep);
    Predictions_plot = Predictions(:,rep);
    Likelihood_plot = Likelihood(:,rep);
    scatter(Predictions_plot,Entropy_plot);
    hold on
    %xlim([27 31])
end
xlabel('Predictions')
ylabel('Entropy')
title('Relationship between predictions and entropy')


figure()
for rep = 1:repetitions
    Entropy_plot = Entropy(:,rep);
    Predictions_accuracy_plot = Prediction_accuracy(:,rep);
    Likelihood_plot = Likelihood(:,rep);
    scatter(Predictions_accuracy_plot,Entropy_plot);
    hold on
    %xlim([27 31])
end
xlabel('Prediction accuracy')
ylabel('Entropy')
title('Relationship between prediction accuracy and entropy')
% %%
% for rep = 1:45
%     figure()
%    Entropy_rep = Entropy_subject_ln(:,rep);
%    Entropy_rep_gauss = Entropy_gaussianized(:,rep);
%   
%    subplot(2,1,1)
%    histogram(Entropy_rep)
%    title('Entropy (original)')
%    xlabel('Entropy'); ylabel('Frequency')
%    subplot(2,1,2)
%    histogram(Entropy_rep_gauss)
%    title('Entropy (Gaussianized)')
%    xlabel('Entropy (Gaussianized)'); ylabel('Frequency')
% 
%    
% end
% % % 

%% Old code

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

% I think we want to see prediction accuracy for a given subject, and the
% entropy plotted on a graph, and see if there is a relationship between the two



% % load meta data
% %load('HMMs_meta_data_subject_BEST_5_reps')
% load('HMMs_meta_data_subject')
% MF2 = Entropy_subject;
% MF3 = likelihood_subject;
% standardized_MF2 = (MF2-mean(MF2))./std(MF2) + 1;
% % Question: do we want to normalize?
% % For e.g. entropy, the higher the state, the higher the entropy. Now we
% % don't just want to emphasise the higher state predictions, so we need to
% % normalize to get them on the same scale. Likelihood is very large so we
% % scale just to help us read it a bit better
% metafeature_check = standardized_MF2;
% %normalized_MF2 = (MF2-min(MF2))./(max(MF2)-min(MF2));
% %normalized_MF2_mean_1 = normalized_MF2 - mean(normalized_MF2) + 1;
% %normalized_MF2_mean_1 = MF2-min(MF2);
% 
% load('vars_best_5_reps.mat')
% load('HMM_predictions_stack_meta_BEST_5_reps.mat')
% %load('vars_v4.mat')
% %load('HMM_predictions_stack.mat')
% for var = 1:34
%     var;
% y_new = vars(:,var);
% yhat_ST = squeeze(vars_hat_ST(:,var,:));
% 
% 
% [B,I] = sort(metafeature_check(:,1:3),2);
% % rank from 1 to 3 about which repet    ition has highest entropy
% % 1 is highest entropy, 3 is lowest entropy
% % theoretically, higher entropy should be the best guess
% r = repmat((1:3),length(y_new),1);
% 
% r_metafeature = NaN(length(y_new),3);
% for i = 1:length(y_new)
%     index_store = I(i,:);
%     rank_store = r(i,:);
%     rank_store(index_store) = rank_store;
%     r_metafeature(i,:) = rank_store;
% end
% r_metafeature;
% % 
% % % Now, we find the accuracy of our predictions and rank them
% % prediction_accuracy = abs(y_new-yhat_ST(1:3)).^2;
% % [B_yhat,I_yhat] = sort(abs(y_new-yhat_ST(1:3)).^2,2);
% % 
% % r = repmat((1:3),length(y_new),1);
% % 
% % r_yhat = NaN(length(y_new),3);
% % for i = 1:length(y_new)
% %     index_store = I_yhat(i,:);
% %     rank_store = r(i,:);
% %     rank_store(index_store) = rank_store;
% %     r_yhat(i,:) = rank_store;
% % end
% % r_yhat;
% % 
% % % How many times was the highest entropy equal to the best prediction?
% % % Second highest equal to second best and so on.
% % sum(r_metafeature == r_yhat)
% % 
% % % How many rank positions out is the metafeature ranking and then
% % % predictions ranking across all subjects? (/mean per subject)
% % % rank_diff = r_metafeature - r_yhat;
% % % sum(sum(abs(rank_diff)))/1001
% % end
% % 


% % are the metafeatures correlated with the predictions in any way?
% corr_mat = corr(standardized_MF2(:,1:3),squeeze(vars_hat_ST(:,1,1:3)))
% diag(corr_mat) % take the diagonal to get columnwise correlation
% 
% %% Plots
% % we want to plot the relationship between HMM runs' accuracy and their
% % metafeatures
% 
% % load data
% load('HMM_predictions.mat')
% load('HMMs_meta_data_subject_r_1_27_40_63')
% load('vars_target.mat')
% 
% % store data
% Entropy_plot = Entropy_subject_all(:,1:27);
% %Entropy_plot = Entropy_subject_all(:,[2 6 14 25 44]);
% vars_target = vars;
% vars_predictions = vars_hat(:,:,1:27);
% %vars_predictions = vars_hat(:,:,[2 6 14 25 44]);
% 
% % calculate prediction accuracies
% n_subjects = 1001;
% squared_error = ((vars_target-vars_predictions).^2);%/n_subjects
% mean_squared_error = squeeze(sum(squared_error,'omitnan')/nnz(~isnan(squared_error)));
% 
% % here we plot entropy vs prediction accuracy
% for var = 1
%     for rep = 1
%         X = mean(Entropy_plot);
%         Y = mean_squared_error(var,:);
%         figure()
%         scatter(X,Y,'x')
%         %plot(X,Y)
%         title('Relationship between entropy and prediction accuracy for 27 HMM runs')
%         xlabel('Mean Entropy'); ylabel('Mean Squared Error')
%         
%     end
% end
% % 
% %%
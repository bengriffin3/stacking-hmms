% % % FC-HMM
% % parfor r = 65:80
% %     search3_rep1(r)
% % end
% 
% % % Mean-FC-HMM
% % parfor r = 81:96
% %     search3_rep1(r)
% % end
% % 
% % % Mean-HMM
% % parfor r = 98:105
% %     search3_rep1(r)
% % end
% 
% % parfor r = 65:84
% %     search3_distmats(r)
% % end
n_vars = 10;
parfor r = 65:80%:7%:24
    for j = 1:n_vars
        search3_rep1_preds(r,j)
    end
end

% % % 'numIterations' is an integer with the total number of iterations in the loop.
% % % Feel free to increase this even higher and see other progress monitors fail.
% % clc
% % numIterations =8;
% % 
% % % Then construct a ParforProgressbar object:
% % ppm = ParforProgressbar(numIterations);
% % 
% % parfor r = 1:numIterations
% %     % do some parallel computation
% %     search3_rep1_preds(r)
% %     % increment counter to track progress
% %     ppm.increment();
% % end
% % 
% % % Delete the progress handle when the parfor loop is done (otherwise the timer that keeps updating the progress might not stop).
% % delete(ppm);
% 
% %%
% 
% % 
% 
% clc
% n_vars = 34;
% n_subjects = 1001;
% n_reps = 24;
% 
% 
% %DistMat_all = NaN(n_subjects,n_subjects,n_reps);
% DistMat_all = NaN(n_subjects,n_subjects,84);
% for i = 65:84%1:n_reps
%     i
%            load(['KLdistances_ICA50r' num2str(i) '.mat'])
%            DistMat_all(:,:,i) = DistMat(:,:,i);
% end
% DistMat = DistMat_all;
% %save('KLdistances_ICA50.mat','DistMat','DistStatic')
% save('KLdistances_ICA50_all.mat','DistMat')
% 
% %%
% explained_variance_all = NaN(n_vars,n_reps);
% vars_hat_all = NaN(n_subjects,n_vars,n_reps);
% for i = 1:n_reps
%     i
%     load(['staticFC_predictions_r' num2str(i) '.mat'])
%     explained_variance_all(:,i) = explained_variance;
%     vars_hat_all(:,:,i) = vars_hat;
% end
% 
% explained_variance = explained_variance_all;
% vars_hat = vars_hat_all;
% 
% save('staticFC_predictions.mat','explained_variance','vars_hat')
% %%
% 
% explained_variance_all = NaN(n_vars,n_reps);
% vars_hat_all = NaN(n_subjects,n_vars,n_reps);
% 
% for i = 1:24%n_reps
%     i
%     load(['HMM_predictions_r' num2str(i) '_KRR_bias_correct.mat'])
%     explained_variance_all(:,i) = explained_variance(:,i);
%     vars_hat_all(:,:,i) = vars_hat(:,:,i);
% 
% end
% 
% explained_variance = explained_variance_all;
% vars_hat = vars_hat_all;
% 
% save('HMM_predictions_KRR_bias_correct.mat','explained_variance','vars_hat','folds_all')
% 
% %%
% load vars_target.mat
% clc
% v = 34;
% [M,I] = max(squeeze(subject_squared_error(:,v,:)));
% [M' I']
% 
% 
% a = [vars(1:n_subjects,v) squeeze(vars_hat_krr_predict(:,v,:))]; 
% [M,I] = maxk(a,3)
% [M,I] = mink(a,3);
% 
% % sum_folds = NaN(10,1);
% % b = folds_all(:,34);
% % for i = 1:10
% %     c = b{i};
% %     sum_folds(i) = sum(vars(c,34));
% % end
% % [[folds_all{:,34}]' sum_folds]





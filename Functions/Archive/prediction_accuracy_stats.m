function [mse_ST,MSE_stack_all,ev_ST,ev_stack_all] = prediction_accuracy_stats(vars_target,pred_ST,pred_stack_all)

n_subjects = size(pred_ST,1);

pred_stack_ls = pred_stack_all(:,1);
pred_stack_rdg = pred_stack_all(:,2);
pred_FWLS_ls = pred_stack_all(:,3);
pred_FWLS_rdg = pred_stack_all(:,4);
pred_stack_RF = pred_stack_all(:,5);
pred_FWLS_RF = pred_stack_all(:,6);

% Mean squared error of predictions
mse_ST = sum((vars_target - pred_ST).^2)/n_subjects;
mse_stack_ls = sum((vars_target - pred_stack_ls).^2)/n_subjects;
mse_stack_ridge = sum((vars_target - pred_stack_rdg).^2)/n_subjects;
mse_FWLS_ls = sum((vars_target - pred_FWLS_ls).^2)/n_subjects;
mse_FWLS_ridge = sum((vars_target - pred_FWLS_rdg).^2)/n_subjects;
mse_stack_RF = sum((vars_target - pred_stack_RF).^2)/n_subjects;
mse_FWLS_RF = sum((vars_target - pred_FWLS_RF).^2)/n_subjects;

% Explained variance  
ev_ST = corr(pred_ST,vars_target).^2;
ev_stack_ls = corr(pred_stack_ls,vars_target).^2;
ev_stack_ridge = corr(pred_stack_rdg,vars_target).^2;
ev_FWLS_ls = corr(pred_FWLS_ls,vars_target).^2;
ev_FWLS_ridge = corr(pred_FWLS_rdg,vars_target).^2;
ev_stack_RF = corr(pred_stack_RF,vars_target).^2;
ev_FWLS_RF = corr(pred_FWLS_RF,vars_target).^2;

% store prediction stats
MSE_stack_all = [mse_stack_ls mse_stack_ridge mse_FWLS_ls mse_FWLS_ridge mse_stack_RF mse_FWLS_RF];
ev_stack_all = [ev_stack_ls ev_stack_ridge ev_FWLS_ls ev_FWLS_ridge ev_stack_RF ev_FWLS_RF];

end


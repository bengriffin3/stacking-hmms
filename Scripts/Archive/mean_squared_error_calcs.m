% Calculate subject-by-subject squared error
repetitions = 4;
dirichlet_test = 4;
lags = 4;
n_subjects = 1001;
load('HMM_predictions.mat')
load('vars_target.mat')
mean_squared_error =  NaN(size(vars,2),repetitions,dirichlet_test,lags);
coefficients_of_determination =  NaN(size(vars,2),repetitions,dirichlet_test,lags);
subject_squared_error = NaN(size(vars,1),size(vars,2),repetitions,dirichlet_test,lags);
for r = 1:repetitions
    for d = 1:dirichlet_test
        for l = 1:lags
            for j = 1:size(vars,2)
                disp(['Vars ' num2str(j) ])
                y = vars(:,j); % here probably you need to remove subjects with missing values
                % BG code to remove subjects with missing values
                non_nan_idx = find(~isnan(y));
                mean_squared_error(j,r,d,l) = sum((vars(non_nan_idx,j) - vars_hat(non_nan_idx,j,r,d,l)).^2)/n_subjects;
                subject_squared_error(non_nan_idx,j,r,d,l) =(vars(non_nan_idx,j) - vars_hat(non_nan_idx,j,r,d,l)).^2;

                
                y = y(non_nan_idx); % actual data
                f = vars_hat(non_nan_idx,j,r,d,l);% model data
                RSS = sum((y - f).^2);
                TSS = sum((y - mean(y)).^2);
                coefficients_of_determination(j,r,d,l) = 1 - RSS/TSS;

            end
        end
    end
end

%load('HMM_predictions.mat')
save("HMM_predictions_2.mat",'explained_variance','mean_squared_error','subject_squared_error','folds_all','vars_hat')

%% Check variables have been saved correctly
clear;
clc;
load("HMM_predictions_2.mat")
explained_variance_2 = explained_variance;
load("HMM_predictions.mat")
explained_variance_2 == explained_variance

%% Display correlation matrix between errors of each HMM prediction
% clear; clc;
% repetitions = 5;
% dirichlet_test = 4;
% n_subjects = 1001;
% n_vars = 34;
% load("HMM_predictions_2.mat")
% corr_mat = ones(n_vars,repetitions,dirichlet_test);
% for j = 1:n_vars
%     for r = 1:repetitions-1
%         for d = r+1:dirichlet_test
%             % Calculate correlations
%             corr_mat(j,r,d) = corr(subject_squared_error(:,j,r,d),subject_squared_error(:,j,d,r),'rows','complete');
%             % Corr(a,b) = Corr(b,a)
%             corr_mat(j,d,r) = corr_mat(j,r,d);
%         end
%     end
% end
% 
% corr_subject_squared_error = corr_mat; % vars x K x Dirichlet diag
% squeeze(corr_subject_squared_error(j,:,:))
% save("HMM_predictions_2.mat",'explained_variance','mean_squared_error','subject_squared_error','folds_all','vars_hat')
%% Check variables have been saved correctly
load("HMM_predictions_2.mat")
explained_variance_2 = explained_variance;
load("HMM_predictions.mat")
explained_variance_2 == explained_variance



%%
clc
% Calculate subject-by-subject squared error
repetitions = 64;
n_subjects = 1001;
load('HMM_predictions_rep_3.mat')
load('vars_target.mat')
%load('vars_126.mat')
vars = vars(1:n_subjects,:);
mean_squared_error =  NaN(size(vars,2),repetitions);
coefficients_of_determination =  NaN(size(vars,2),repetitions);
subject_squared_error = NaN(size(vars,1),size(vars,2),repetitions);
for r = 1:repetitions
    for j = 1:size(vars,2)
        disp(['Vars ' num2str(j) ])
        y = vars(:,j); % here probably you need to remove subjects with missing values
        % BG code to remove subjects with missing values
        non_nan_idx = find(~isnan(y));
        mean_squared_error(j,r) = sum((vars(non_nan_idx,j) - vars_hat(non_nan_idx,j,r)).^2)/n_subjects;
        subject_squared_error(non_nan_idx,j,r) =(vars(non_nan_idx,j) - vars_hat(non_nan_idx,j,r)).^2;
        
        %%% ADD COEFFICIENT OF DETERMINATION (THEN ADD IT TO
        %%% PROGRESS SUMMARY DOC)

        y = y(non_nan_idx); % actual data
        f = vars_hat(non_nan_idx,j,r);% predicted (model) data
        RSS = sum((y - f).^2);
        TSS = sum((y - mean(y)).^2);
        coefficients_of_determination(j,r) = 1 - RSS/TSS;
        % note that the coefficient of determination becomes less than 0 if
        % the observed value is closer to the mean than the model
        % prediction (in other words, the sum of squared errors is larger than the null model). 
        % This is obviously quite likely for the 'terrible
        % predictions' that get get sometimes

    end
end


save("HMM_predictions_rep_3_new.mat",'explained_variance','mean_squared_error','subject_squared_error','vars_hat','folds','alphas','coefficients_of_determination')




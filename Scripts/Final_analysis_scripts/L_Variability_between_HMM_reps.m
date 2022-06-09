%% Spearman's correlation between HMM repetition predictions
clear; clc;
% set directory
%DirOut = 'C:\Users\au699373\OneDrive - Aarhus Universitet\Dokumenter\MATLAB\HMMMAR_BG\HMMMAR Results\MAR_2022\Investigating_variability\Repetition_3_ICA50_TDE_WRONG_DIM_SHOULD_BE_100\';
DirOut = '/Users/bengriffin/Library/CloudStorage/OneDrive-AarhusUniversitet/Dokumenter/MATLAB/HMMMAR_BG/HMMMAR Results/MAR_2022/Investigating_variability/Repetition_3_ICA50_TDE_WRONG_DIM_SHOULD_BE_100/';

% load predictions
%HMM_predictions = load([DirOut 'HMM_predictions.mat'], 'vars_hat','subject_squared_error');
HMM_predictions = load([DirOut 'HMM_predictions_vary_alpha_param.mat'], 'vars_hat','subject_squared_error','mean_squared_error','explained_variance');
n_subjects = size(HMM_predictions.vars_hat,1);
n_vars = size(HMM_predictions.vars_hat,2);
load('vars_target.mat')


vars_hat_reshape = reshape(HMM_predictions.vars_hat,n_subjects,n_vars, []); % reshape predictions
subject_squared_error_reshape = reshape(HMM_predictions.subject_squared_error,n_subjects,n_vars, []); % reshape errors
n_reps = size(vars_hat_reshape,3);
exclude_predictions = false(n_reps,n_vars);

% find best X predicted subjects
n_subjects_asses = 100;
n_new_well_pred_subjects = repmat([n_subjects_asses; NaN(n_reps-1,1)],1,n_vars);
%n_new_well_pred_subjects = cell(n_vars,1);
well_pred_subjects = NaN(n_subjects_asses,n_reps,n_vars);

%figure;
for i = 11%:14%n_vars

    well_pred_subjects_vec = [];
    vars_hat_reshape_i = squeeze(vars_hat_reshape(:,i,:));
    subject_squared_error_reshape_i = squeeze(subject_squared_error_reshape(:,i,:));

    % remove terrible predictions
    max_vars = max(vars(:,i))*1.1; % note highest target feature value + add a buffer
    min_vars = min(vars(:,i))/1.1; % note lowest target feature value + add a buffer
    exclude_predictions(:,i) = (max(vars_hat_reshape_i)' > max_vars) | (min(vars_hat_reshape_i)' < min_vars); % note predictions with out-of-range predictions
    vars_hat_reshape_i_exc = vars_hat_reshape_i(:,~exclude_predictions(:,i)); % exclude all out of range 
    fprintf('Number of excluded predictions %i\n', nnz(exclude_predictions(:,i))) % display how many predictions we removed
    

    % spearman's rank between HMMs
    if nnz(exclude_predictions(:,i)) == n_reps; continue; end

    
    vars_hat_reshape_spear = corr(vars_hat_reshape_i_exc,'type','Spearman','rows','complete');
    vars_hat_reshape_spear(eye(size(vars_hat_reshape_spear))==1) = nan;
    %     subplot(2,2,i-10)
    figure
    subplot(2,1,1)
    imagesc(vars_hat_reshape_spear); colorbar;
    xlabel('Predictions from HMM repetitions'); ylabel('Predictions from HMM repetitions');
    title(sprintf('Spearman''s correlation between HMM predictions for variable %i',i))

    subject_squared_error_reshape_i_exc = subject_squared_error_reshape_i(:,~exclude_predictions(:,i));
    n_good_reps = size(vars_hat_reshape_i_exc,2);

    % reorder errors so best EV is first
    [~,order_EV] = sort(HMM_predictions.explained_variance(i,~exclude_predictions(:,i)));
    subject_squared_error_reshape_i_exc_reorder = subject_squared_error_reshape_i_exc(:,order_EV);

    stack_ev = NaN(n_good_reps,1);
    stack_W = NaN(n_good_reps,n_good_reps);


    for rep = 1:n_good_reps%n_reps
        % determine top/bottom x well predicted per HMM repetition
        [~,well_pred_subjects(:,rep,i)] = mink(subject_squared_error_reshape_i_exc_reorder(:,rep),n_subjects_asses);
        [~,badly_pred_subjects] = maxk(subject_squared_error_reshape_i_exc_reorder(:,rep),n_subjects_asses);
        if rep == 1; well_pred_subjects_vec = well_pred_subjects(:,rep,i); continue; end

        % find unique subjects predicted well/poorly
        n_new_well_pred_subjects(rep,i) = nnz(setdiff(well_pred_subjects(:,rep,i),well_pred_subjects_vec));
        %n_new_well_pred_subjects{i} =  nnz(setdiff(well_pred_subjects,well_pred_subjects_vec));
        well_pred_subjects_vec = unique([well_pred_subjects_vec; well_pred_subjects(:,rep,i)]);

        % Now we stack predictions
        sim_params.stack_method = 'least_squares';
        var_target = vars(:,i);
        X = vars_hat_reshape_i_exc(:,1:rep); % Store HMM predictions up to rep i
        X = rmmissing(X); % remove NaNs
        var_target = rmmissing(var_target);
        y = var_target;

        % stack predictions
        [yhat,stack_W(rep,1:rep)] = stacking_regress_ben(X,y,sim_params);
        
        % test model
        stack_sse = (yhat - y).^2; stack_mse = sum(stack_sse)/n_subjects; stack_ev(rep) = corr(yhat,var_target).^2;
    
    
    end
    
    subplot(2,1,2)
    %bar(n_new_well_pred_subjects(~exclude_predictions(:,i),i))
    yyaxis left
    bar(n_new_well_pred_subjects(1:n_good_reps,i))
    xlabel('HMM repetition'); ylabel('Number of subjects')
    title(sprintf('No. unique subjects in the best predicted %i from HMM i that is not in the HMMs 1 to i-1',n_subjects_asses))
    
    yyaxis right
    scatter(1:n_good_reps,sort(HMM_predictions.explained_variance(i,~exclude_predictions(:,i)),'descend'),'x')
    ylabel('Explained Variance of HMM repetition')
    hold on
    scatter(1:n_good_reps,stack_ev)
    legend('','Individual EV','Stacked EV')
    
    figure;
    imagesc(stack_W); colorbar;
    xlabel('HMM weight'); ylabel('No. HMMs in stack')

end
fprintf('Total predictions excluded %i out of a possible %i, i.e. %i%% \n', nnz(exclude_predictions), numel(exclude_predictions),round(nnz(exclude_predictions)/numel(exclude_predictions)*100,0)) % display how many predictions we removed




%% Overlap of 'good' and 'bad' subjects

n_common_well_subjects = NaN(n_good_reps,n_good_reps);
n_common_bad_subjects = NaN(n_good_reps,n_good_reps);
for k = 11:14%n_vars
    k
    for i = 1:n_good_reps
        for j = 1:n_good_reps
            n_common_well_subjects(i,j) = nnz(intersect(well_pred_subjects(:,i,k),well_pred_subjects(:,j,k)));
            n_common_bad_subjects(i,j) = nnz(intersect(well_pred_subjects(:,i,k),well_pred_subjects(:,j,k)));
        end
    end
    % set diagonal to NaNs so we can plot
    n_common_well_subjects(eye(size(n_common_well_subjects))==1) = nan;
    n_common_bad_subjects(eye(size(n_common_bad_subjects))==1) = nan;

    figure;
    subplot(1,2,1)
    imagesc(n_common_well_subjects); colorbar;
    xlabel('HMM repetition'); ylabel('HMM repetition');
    title(sprintf('Well predicted subjects (top %i)',n_subjects_asses))

    subplot(1,2,2)
    imagesc(n_common_bad_subjects); colorbar;
    xlabel('HMM repetition'); ylabel('HMM repetition');
    title('Poorly predicted subjects (bottom 100)')
    title(sprintf('Poorly predicted subjects (bottom %i)',n_subjects_asses))

    sgtitle('Crossover of well/poorly predicted subjects between distinct HMM repetitions')
end







% 
%% What are the reps we are looking at?

%DirOut = 'C:\Users\au699373\OneDrive - Aarhus Universitet\Dokumenter\MATLAB\HMMMAR_BG\HMMMAR Results\MAR_2022\Investigating_variability\Repetition_1_ICA50_TDE\';
DirOut = 'C:\Users\au699373\OneDrive - Aarhus Universitet\Dokumenter\MATLAB\HMMMAR_BG\HMMMAR Results\MAR_2022\Investigating_variability\Repetition_2_ICA50_TDE_WRONG_DIM_SHOULD_BE_100\';
%DirOut = 'C:\Users\au699373\OneDrive - Aarhus Universitet\Dokumenter\MATLAB\HMMMAR_BG\HMMMAR Results\MAR_2022\Repetition_2\';
%DirOut = 'C:\Users\au699373\OneDrive - Aarhus Universitet\Dokumenter\MATLAB\HMMMAR_BG\HMMMAR Results\MAR_2022\Repetition_3_ICA50\';
n_reps = 64;
Index = (1:n_reps)';
K_store = NaN(n_reps,1);
DD_store = NaN(n_reps,1);
lags_store = NaN(n_reps,1);
zeromean_store = NaN(n_reps,1);
covtype_store = cell(n_reps,1);
ndim_store = NaN(n_reps,1);
order_store = NaN(n_reps,1);

for i = 1:n_reps
    i
    load([DirOut 'HMMs_r' num2str(i) '_d' num2str(i) '_GROUP'],'hmm')
    DD_store(i) = hmm.train.DirichletDiag;
    %lags_store{i} = min(hmm.train.embeddedlags):max(hmm.train.embeddedlags);
    lags_store(i) = max(hmm.train.embeddedlags);
    K_store(i) = hmm.K;
    zeromean_store(i) = hmm.train.zeromean;
    covtype_store{i} = hmm.train.covtype;
    ndim_store(i) = hmm.train.ndim;
    order_store(i) = hmm.train.order;

end

% put in table
%T = table(Index,K_store,DD_store,lags_store,zeromean_store,string(covtype_store))
column_names = {'Index','K (states)','Dirichlet_diag','Lags (3 = -3:3)','Zeromean','No. dimensions','Order','Covtype'};
dataCell = [num2cell([Index,K_store,DD_store,lags_store,zeromean_store, ndim_store, order_store]), covtype_store];

% Output table to figure
figure;

%uitable('Data',dataCell,'ColumnName',column_names,'RowName',T.Properties.RowNames,'Units', 'Normalized', 'Position',[0, 0, 1, 1]);
uitable('Data',dataCell,'ColumnName',column_names,'Units', 'Normalized', 'Position',[0, 0, 1, 1]);





%%

















%%% OLD CODE

% %% Overlap of 'good' and 'bad' subjects
% 
% % reshape errors
% subject_squared_error_reshape = reshape(HMM_predictions.subject_squared_error,n_subjects,n_vars, []);
% 
% % find best X predicted subjects
% n_subjects_asses = 200;
% well_pred_subjects = NaN(n_subjects_asses,n_reps,n_vars); n_common_well_subjects = NaN(n_reps,n_reps,n_vars);
% badly_pred_subjects = NaN(n_subjects_asses,n_reps,n_vars); n_common_bad_subjects = NaN(n_reps,n_reps,n_vars);
% n_new_well_pred_subjects = repmat([n_subjects_asses; NaN(n_reps-1,1)],1,n_vars);
% well_pred_subjects_vec = [];
% 
% for i = 1:34
%     i
%     subject_squared_error_reshape_i = squeeze(subject_squared_error_reshape(:,i,:));
%     vars_hat_reshape_i = squeeze(vars_hat_reshape(:,i,:));
% 
%     % remove terrible predictions
%     max_vars = max(vars(:,i))*1.1; % note highest target feature value + add a buffer
%     min_vars = min(vars(:,i))/1.1; % note lowest target feature value + add a buffer
%     exclude_predictions(:,i) = (max(vars_hat_reshape_i)' > max_vars) | (min(vars_hat_reshape_i)' < min_vars); % note predictions with out-of-range predictions
%     vars_hat_reshape_i_exc = vars_hat_reshape_i(:,~exclude_predictions(:,i)); % exclude all out of range
%     fprintf('Number of excluded predictions %i\n', nnz(exclude_predictions(:,i))) % display how many predictions we removed
% 
% 
%     for rep = 1:n_reps
%         % determine top/bottom x well predicted per HMM repetition
%         [~,well_pred_subjects(:,rep,i)] = mink(subject_squared_error_reshape_i(:,rep),n_subjects_asses);
%         [~,badly_pred_subjects(:,rep,i)] = maxk(subject_squared_error_reshape_i(:,rep),n_subjects_asses);
%         if rep == 1; well_pred_subjects_vec = well_pred_subjects(:,rep,i); continue; end
% 
%         % find unique subjects predicted well/poorly
%         n_new_well_pred_subjects(rep,i) = nnz(setdiff(well_pred_subjects(:,rep,i),well_pred_subjects_vec));
%         well_pred_subjects_vec = unique([well_pred_subjects_vec; well_pred_subjects(:,rep,i)]);
%     end
% end
% n_new_well_pred_subjects


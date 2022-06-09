% Code used in Vidaurre et al. (2021) NEuroImage
%
% This script assumes that all the preprocessing has been done and 
% that we have in memory: 
%
% - The rs fMRI data, including f, the file names with the rs-fMRI data, and
% T, the length of each session 
% (T is a cell, and T{subject} = [1200 1200 1200 1200] in the
% case of the HCP, because there are 1200 time points per session.)
% - The behavioural variables in vars
% - The family structure in twins
% - The confounds in conf (eg motion,sex)
% - The structural data in cell Anatomy (three elements, one per each type of structural) 
%
load hcp1003_RESTall_LR_groupICA50.mat
f = data(grotKEEP);
n_sessions= 4;
n_timepoints = 1200;
T_subject{1} = repmat(n_timepoints,n_sessions,1)';
T = repmat(T_subject,size(vars,1),1);


% The pipeline is
%
% 1. Running the HMM with the function hmmmar and computing the
% dual-estimated HMM models with hmmdual
%
% 2. Compute the distance matrices in HMM space as well as for the static
% FC. This is done respectively with hmm_kl() on the dual estimated models.
% For the static FC matrices, this is done with the wishart_kl function. 
%
% 3. Compute the predictions on the structural data, also based on distance
% matrices. These are just Euclidean distances. The prediction is done
% through the predictPhenotype.m function. Then, save the residuals of
% these predictions as deconfounded behavioural variables. 
%
% 4. Run the predictions using the static FC distance matrices on the raw
% or structure-deconfounded behavioural variables, using predictPhenotype.m
%
% 5. Similarly, run the predictions using each of the HMM distance matrices on the raw
% or structure-deconfounded behavioural variables, using predictPhenotype.m

%mydir = '/home/diegov/MATLAB/'; % set to your directory
%addpath(genpath([ mydir 'HMM-MAR'])) % HMM repository

ICAdim = 50; % number of ICA components (25,50,100,etc)
%K = 8; % no. states
covtype = 'full'; % type of covariance matrix
zeromean = 1;


DirOut = '/Users/bengriffin/Library/CloudStorage/OneDrive-AarhusUniversitet/Dokumenter/MATLAB/HMMMAR_BG/HMMMAR Results/FEB_2022/K_prior_grid_search_4/'; % Test folder
%DirOut = 'C:\Users\au699373\OneDrive - Aarhus Universitet\Dokumenter\MATLAB\HMMMAR_BG\HMMMAR Results\FEB_2022\K_prior_grid_search_4\'; % Test folder
% We will save here the distance matrices

%K_vec = [3 3 3 6 6 6 9 9 9 12 12 12 15 15 15];
K_vec = [3 9 15];
hmm_train_dirichelt_diag_vec = [10 10000 10000000 10000000000 10000000000000 10000000000000000 10000000000000000000 10000000000000000000000];

repetitions = length(K_vec); % to run it multiple times (keeping all the results)
dirichlet_test = length(hmm_train_dirichelt_diag_vec);
TR = 0.75; N = length(f); 

%% Run the HMMs (5 repetitions, with states characterised by a covariance matrix)
tic
options = struct();
%options.K = K; % number of states
options.order = 0; % no autoregressive components
options.covtype = covtype;
options.zeromean = zeromean;
options.Fs = 1/TR;
options.standardise = 1;
options.DirichletDiag = 10;
options.dropstates = 0;
options.cyc = 50;
options.initcyc = 5;
options.initrep = 3;
options.verbose = 0;
% stochastic options
options.BIGNbatch = round(N/30);
options.BIGtol = 1e-7;
options.BIGcyc = 100;
options.BIGundertol_tostop = 5;
options.BIGforgetrate = 0.7;
options.BIGbase_weights = 0.9;

options_singlesubj = struct();
%options_singlesubj.K = K; % number of states
options_singlesubj.order = 0; % no autoregressive components
options_singlesubj.zeromean = zeromean; % don't model the mean
options_singlesubj.covtype = covtype;
options_singlesubj.Fs = 1/TR;
options_singlesubj.standardise = 1;
%options_singlesubj.DirichletDiag = 10;
options_singlesubj.dropstates = 0;
options_singlesubj.cyc = 100;
options_singlesubj.verbose = 0;

options_singlesubj_dr = options_singlesubj;
options_singlesubj_dr.updateGamma = 0;

% We run the HMM multiple times
for r = 1:repetitions
    disp(['Repetition ' num2str(r) ])
    K = K_vec(r);
    options.K = K; % number of states
    options_singlesubj.K = K; % number of states

    for d = 1:dirichlet_test
    options_singlesubj.DirichletDiag = hmm_train_dirichelt_diag_vec(d);



    % Run the HMM at the group level and get some statistics
    % (eg fractional occupancy)
    [hmm,Gamma,~,vpath,~,~,fehist] = hmmmar(f,T,options);
    FOgroup = zeros(N,K); % Fractional occupancy
    meanActivations = zeros(ICAdim,K); % maps
    for j=1:N
        ind = (1:4800) + 4800*(j-1);
        FOgroup(j,:) = mean(Gamma(ind,:));
        X = zscore(f{j}); %X = zscore(dlmread(f{j}));
        for k = 1:K
            meanActivations(:,k) = meanActivations(:,k) + ...
                sum(X .* repmat(Gamma(ind,k),1,ICAdim))';
        end
    end
    meanActivations = meanActivations ./ repmat(sum(Gamma),ICAdim,1);
    switchingRate = getSwitchingRate(Gamma,T,options);
    maxFO = getMaxFractionalOccupancy(Gamma,T,options);
    
    % Subject specific stuff (dual-estimation)
    options_singlesubj.hmm = hmm;
    FOdual = zeros(N,K);
    parfor j = 1:N
        X = f{j};%X = dlmread(f{j});
        % dual-estimation
        [HMMs_dualregr{j},Gammaj{j}] = hmmdual(X,T{j},hmm);
        for k = 1:K
            HMMs_dualregr{j}.state(k).prior = [];
        end
        FOdual(j,:) = mean(Gammaj{j});
    end
    
    save([DirOut 'HMMs_r' num2str(r) '_d' num2str(d) '_GROUP.mat'],...
       'hmm','Gamma','vpath','FOgroup','meanActivations','switchingRate','maxFO','fehist')
    save([DirOut 'HMMs_r' num2str(r) '_d' num2str(d)  '_states_' num2str(K) '.mat'],...
       'HMMs_dualregr','Gammaj','FOdual')
%     save([DirOut 'HMMs_r' num2str(r) '_GROUP.mat'],...
%        'hmm','Gamma','vpath','FOgroup','meanActivations','switchingRate','maxFO','fehist')
%     save([DirOut 'HMMs_r' num2str(r) '_states_' num2str(K) '.mat'],...
%        'HMMs_dualregr','Gammaj','FOdual')
    end
end
toc
%% Create distance matrices between models 
clc
% between HMMs
DistHMM = zeros(N,N,repetitions,repetitions); % subj x subj x repetitions
% DistHMM_n1n2 = zeros(N,N,5);
% DistHMM_n2n1 = zeros(N,N,5);

for r = 1%:repetitions
    K = K_vec(r);
    disp(['Repetition ' num2str(r) ])
    for d = 1:dirichlet_test
    %out = load([DirOut 'HMMs_r' num2str(r) '_states_' num2str(K) '.mat']);
    out = load([DirOut 'HMMs_r' num2str(r) '_d' num2str(d)  '_states_' num2str(K) '.mat']);
    for n1 = 1:N-1
        for n2 = n1+1:N
            % FO is contained in TPC; TPC is contained in HMM
%             DistHMM_n1n2(n1,n2,r) = hmm_kl(out.HMMs_dualregr{n1},out.HMMs_dualregr{n2})/2;
%             DistHMM_n2n1(n1,n2,r) = hmm_kl(out.HMMs_dualregr{n2},out.HMMs_dualregr{n1})/2;

%             DistHMM(n1,n2,r) = (hmm_kl(out.HMMs_dualregr{n1},out.HMMs_dualregr{n2}) ...
%                 + hmm_kl(out.HMMs_dualregr{n2},out.HMMs_dualregr{n1}))/2;
            DistHMM(n1,n2,r,d) = (hmm_kl_BG(out.HMMs_dualregr{n1},out.HMMs_dualregr{n2}) ...
                + hmm_kl(out.HMMs_dualregr{n2},out.HMMs_dualregr{n1}))/2;
            DistHMM(n2,n1,r,d) = DistHMM(n1,n2,r,d);

%             DistHMM_n1n2(n2,n1,r) = DistHMM_n1n2(n1,n2,r);
%             DistHMM_n2n1(n2,n1,r) = DistHMM_n2n1(n1,n2,r);

        end
    end
    disp(num2str(r))
    end
end
DistMat = DistHMM;


% Create correlation matrix for each subject
corr_mat = zeros(ICAdim,ICAdim,N);
for s = 1:N %for each subject
    sub = data{s};
    for i = 1:ICAdim
        for j = 1:ICAdim
            corr_coeff = corrcoef(sub(:,i),sub(:,j));
            corr_mat(i,j,s) = corr_coeff(2,1);
        end
    end
    disp(['Subject no. ' num2str(s) ])
end
V = corr_mat;


% between static FC matrices
DistStatic = zeros(N);
for n1 = 1:N-1
    for n2 = n1+1:N
        DistStatic(n1,n2) = ( wishart_kl(V(:,:,n1),V(:,:,n2),sum(T{n1}),sum(T{n2})) + ...
            wishart_kl(V(:,:,n2),V(:,:,n1),sum(T{n2}),sum(T{n1})) ) /2;
        DistStatic(n2,n1) = DistStatic(n1,n2);
    end
    disp(['Subject no. ' num2str(n1) ])
end

save([DirOut 'KLdistances_ICA' num2str(ICAdim) '.mat'],'DistMat','DistStatic');


%% Predictions of behaviour using structurals
% The code here is a bit complex but what matters is the calls to
% predictPhenotype.m
% 
% prediction parameters
parameters_prediction = struct();
parameters_prediction.verbose = 0;
parameters_prediction.method = 'KRR';
parameters_prediction.alpha = [0.1 0.5 1.0 5];
parameters_prediction.sigmafact = [1/2 1 2];
% 
% for ia = 1:4 % cycle through the structural variables
%     
%     A = Anatomy{ia}; 
%  
%     % computing the Euclidean distnces between the structurals 
%     D = zeros(size(A,1));
%     for n1 = 1:size(A,1)-1
%         for n2 = n1+1:size(A,1)
%             D(n1,n2) = sqrt( sum( (A(n1,:) - A(n2,:)).^2 ) );
%             D(n2,n1) = D(n1,n2);
%         end
%     end
%     
%     % performing the prediction
%     explained_variance = NaN(size(vars,2),1);
%     vars_hat = NaN(N,size(vars,2)); % the predicted variables
%     for j = 1:size(vars,2)
%         y = vars(:,j); % here probably you need to remove subjects with missing values
%         [yhat,stats] = predictPhenotype(y,D,parameters_prediction,twins,conf);
%         explained_variance(j) = corr(squeeze(yhat),y).^2;
%         vars_hat(:,j) = yhat;
%     end
% 
%     save('structural_predictions.mat','vars_hat','explained_variance')
%         
% end


%% Predictions of behaviour using the static FC (with and without structural deconfounding) 
% The static FC is used only through the distance matrices computed previously

Corrected_by_structure = 0; % set this to 0...3 

vars0 = vars; 

switch Corrected_by_structure
    case 1
        vars = vars - vars_hat;
    case 2
        vars = vars - vars_hat;
    case 3
        vars = vars - vars_hat;
end

D = DistStatic; 
explained_variance = NaN(size(vars,2),1);
vars_hat = NaN(N,size(vars,2)); % the predicted variables


for j = 1:size(vars,2)
    disp(['Variable ' num2str(j) ])
    y = vars(:,j); % here probably you need to remove subjects with missing values
    y_new = y;
    D_new = D;
    conf_new = conf;
    twins_new = twins;

    % BG code to remove subjects with missing values
    non_nan_idx = find(~isnan(y));
    which_nan = isnan(y);
    if any(which_nan)
        y_new = y(~which_nan);
        D_new = D(~which_nan,~which_nan);
        conf_new = conf(~which_nan,:);
        twins_new = twins(~which_nan,~which_nan);
        warning('NaN found on Yin, will remove...')
    end

     [yhat,stats] = predictPhenotype(y_new,D_new,parameters_prediction,twins_new,conf_new);
     explained_variance(j) = corr(squeeze(yhat),y_new).^2;
     vars_hat_store = NaN(size(vars,1),1);
     vars_hat_store(non_nan_idx) = yhat;
     vars_hat(:,j) = vars_hat_store;


end

save([DirOut 'staticFC_predictions.mat'],'vars_hat','explained_variance')

vars = vars0; 

%% Predictions of behaviour using the HMMs (with and without structural deconfounding) 
rng('default')
n_folds = 10;

Corrected_by_structure = 0; % set this to 0...3 

vars0 = vars; 

switch Corrected_by_structure
    case 1
        vars = vars - vars_hat;
    case 2
        vars = vars - vars_hat;
    case 3
        vars = vars - vars_hat;
end

DistHMM = DistMat;
explained_variance = NaN(size(vars,2),repetitions,dirichlet_test);
mean_squared_error =  NaN(size(vars,2),repetitions,dirichlet_test);
coefficient_of_determination = NaN(size(vars,2),repetitions,dirichlet_test);
vars_hat = NaN(N,size(vars,2),repetitions,dirichlet_test); % the predicted variables
folds_all = cell(n_folds,size(vars,2));

for r = 1%:repetitions
    disp(['Repetition ' num2str(r) ])
    for d = 1:dirichlet_test
        disp(['Dirichlet diag ' num2str(d) ])
        D = DistHMM(:,:,r,d);
        for j = 1:size(vars,2)
            disp(['Vars ' num2str(j) ])

            y = vars(:,j); % here probably you need to remove subjects with missing values

            y_new = y;
            D_new = D;
            conf_new = conf;
            twins_new = twins;

            % BG code to remove subjects with missing values
            non_nan_idx = find(~isnan(y));
            which_nan = isnan(y);
            if any(which_nan)
                y_new = y(~which_nan);
                D_new = D(~which_nan,~which_nan);
                conf_new = conf(~which_nan,:);
                twins_new = twins(~which_nan,~which_nan);
                warning('NaN found on Yin, will remove...')
            end

            % y_normalize = 

            %%% KERNEL RIDGE REGRESSION
            % Make predictions and save them
            parameters_prediction.method = 'KRR'; % using KRR
            [yhat,stats,~,~,beta,folds] = predictPhenotype(y_new,D_new,parameters_prediction,twins_new,conf_new);

            explained_variance(j,r,d) = corr(squeeze(yhat),y_new).^2;
            mean_squared_error(j,r,d) = sum((vars(non_nan_idx,j) - vars_hat(non_nan_idx,j,r,d)).^2)/n_subjects;
            coefficient_of_determination(j,r,d) = 1 - sum((vars(non_nan_idx,j) - vars_hat(non_nan_idx,j)).^2) / sum((vars(non_nan_idx,j) - mean(vars(non_nan_idx,j))).^2);
            vars_hat_store = NaN(size(vars,1),1);
            vars_hat_store(non_nan_idx) = yhat;
            vars_hat(:,j,r,d) = vars_hat_store;
            folds_all(:,j) = folds;

        end

    end
end


save([DirOut 'HMM_predictions.mat'],'vars_hat','explained_variance','mean_squared_error','coefficient_of_determination','folds_all')



%% Prediction Stacking
DirOut = 'C:\Users\au699373\OneDrive - Aarhus Universitet\Dokumenter\MATLAB\HMMMAR_BG\HMMMAR Results\FEB_2022\TDE_runs\zeromean_0\';
load([DirOut 'HMM_predictions.mat'],'vars_hat');%,'explained_variance');
load('folds_all');

vars_hat_stack_ls = NaN(N,size(vars,2));
vars_hat_stack_rdg = NaN(N,size(vars,2));
vars_hat_stack_rf = NaN(N,size(vars,2));

for r = 1:repetitions
    disp(['Repetition ' num2str(r) ])
    for d = 1:dirichlet_test
        disp(['Dirichlet diag ' num2str(d) ])
        for j = 1:size(vars,2) 
            disp(['Vars ' num2str(j) ])

            y = vars(:,j); % here probably you need to remove subjects with missing values
            pred_ST = vars_hat(:,j,:,:);

            y_new = y;
            C = squeeze(pred_ST);
            pred_ST_new = reshape(C,[size(C,1) size(C,2)*size(C,3)]);

            % BG code to remove subjects with missing values
            non_nan_idx = find(~isnan(y));
            which_nan = isnan(y);
            if any(which_nan)
                y_new = y(~which_nan);
                pred_ST_new = squeeze(pred_ST_new(~which_nan,:,:,:));
                warning('NaN found on Yin, will remove...')
            end
        
            [vars_hat_stack_ls(non_nan_idx,j), vars_hat_stack_rdg(non_nan_idx,j), vars_hat_stack_rf(non_nan_idx,j)] = hmm_prediction_stacking(y_new,pred_ST_new,folds_all(:,j));


        end
    end
end


%save([DirOut 'HMM_predictions_stack.mat'],'vars_hat_stack_ls', 'vars_hat_stack_rdg', 'vars_hat_stack_rf')

%% Prediction Analysis
load([DirOut 'HMM_predictions.mat'],'vars_hat','explained_variance','mean_squared_error');
load([DirOut 'HMM_predictions_stack.mat'],'vars_hat_stack_ls', 'vars_hat_stack_rdg', 'vars_hat_stack_rf')
n_vars = 34;
n_subjects = 1001;
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
    explained_variance_rf(j)= corr(vars(:,j),vars_hat_stack_rf(:,j),'rows','complete').^2;

    % Calculate mean squared error
    mean_squared_error_ls(j) = sum((vars(:,j) - vars_hat_stack_ls(:,j)).^2,'omitnan')/n_subjects;
    mean_squared_error_rdg(j) = sum((vars(:,j) - vars_hat_stack_rdg(:,j)).^2,'omitnan')/n_subjects;
    mean_squared_error_rf(j) = sum((vars(:,j) - vars_hat_stack_rf(:,j)).^2,'omitnan')/n_subjects;

    % Calculate coefficient of determination
    coefficient_of_determination_ls(j) = 1 - sum((vars(:,j) - vars_hat_stack_ls(:,j)).^2,'omitnan') / sum((vars(:,j) - mean(vars(:,j),'omitnan')).^2,'omitnan');
    coefficient_of_determination_rdg(j) = 1 - sum((vars(:,j) - vars_hat_stack_rdg(:,j)).^2,'omitnan') / sum((vars(:,j) - mean(vars(:,j),'omitnan')).^2,'omitnan');
    coefficient_of_determination_rf(j) = 1 - sum((vars(:,j) - vars_hat_stack_rf(:,j)).^2,'omitnan') / sum((vars(:,j) - mean(vars(:,j),'omitnan')).^2,'omitnan');
    

end

%%% Normalize errors

%mean_squared_error_ls_norm = mean_squared_error_plot./repmat((max(mean_squared_error_plot,[],2)-min(mean_squared_error_plot,[],2)),1,24);
%mean_squared_error_rdg_norm = mean_squared_error_plot./repmat((max(mean_squared_error_plot,[],2)-min(mean_squared_error_plot,[],2)),1,24);
%mean_squared_error_rf_norm = mean_squared_error_plot./repmat((max(mean_squared_error_plot,[],2)-min(mean_squared_error_plot,[],2)),1,24);

save('HMM_predictions_analysis.mat','explained_variance_ls', 'explained_variance_rdg', 'explained_variance_rf','mean_squared_error_ls',...
'mean_squared_error_rdg','mean_squared_error_rf','coefficient_of_determination_ls','coefficient_of_determination_rdg','coefficient_of_determination_rf')

%% Plot analysis
% Plot explained variance
figure; X = 1:n_vars;
scatter(X,explained_variance_ls,'r'); hold on
scatter(X,explained_variance_rdg,'k');
scatter(X,explained_variance_rf,'g');

% here we plot the original predictions
c = linspace(min(hmm_train_dirichelt_diag_vec), max(hmm_train_dirichelt_diag_vec),24);
explained_variance_plot = reshape(explained_variance,[size(explained_variance,1) size(explained_variance,2)*size(explained_variance,3)]);
for j = 1:34; scatter((j*ones(24,1))',explained_variance_plot(j,:),[],c,'x'); hold on; end; colormap winter;
h = colorbar; ylabel(h, 'Dirichlet_diag')
legend('Least Squares','Ridge Regression','Random Forests','Original HMMs');
xlabel('Intelligence Feature'); ylabel('Explained Variance, r^2')

% Plot mean squared error
figure; X = 1:n_vars;
scatter(X,mean_squared_error_ls,'r'); hold on
scatter(X,mean_squared_error_rdg,'k')
scatter(X,mean_squared_error_rf,'g')

% here we plot the original predictions
c = linspace(min(hmm_train_dirichelt_diag_vec), max(hmm_train_dirichelt_diag_vec),24);
mean_squared_error_plot = reshape(mean_squared_error,[size(mean_squared_error,1) size(mean_squared_error,2)*size(mean_squared_error,3)]);
for j = 1:34; scatter((j*ones(24,1))',mean_squared_error_plot(j,:),[],c,'x'); hold on; end; colormap winter;
h = colorbar; ylabel(h, 'Dirichlet_diag')
legend('Least Squares','Ridge Regression','Random Forests','Original HMMs');
xlabel('Intelligence Feature'); ylabel('Mean squared error')

%% Plot coefficient of determination
figure; X = 1:n_vars;
scatter(X,coefficient_of_determination_ls,'r'); hold on
scatter(X,coefficient_of_determination_rdg,'k')
scatter(X,coefficient_of_determination_rf,'g')

% here we plot the original predictions
c = linspace(min(hmm_train_dirichelt_diag_vec), max(hmm_train_dirichelt_diag_vec),24);
coefficient_of_determination_plot = reshape(coefficient_of_determination,[size(coefficient_of_determination,1) size(coefficient_of_determination,2)*size(coefficient_of_determination,3)]);
for j = 1:34; scatter((j*ones(24,1))',coefficient_of_determination_plot(j,:),[],c,'x'); hold on; end; colormap winter;
h = colorbar; ylabel(h, 'Dirichlet_diag')
legend('Least Squares','Ridge Regression','Random Forests','Original HMMs');
xlabel('Intelligence Feature'); ylabel('Coefficient of determination, r^2')



%%



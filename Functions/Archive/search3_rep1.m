function search3_rep1(r)
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
build_regression_data_V3;
load hcp1003_RESTall_LR_groupICA50.mat
f = data(grotKEEP);
n_sessions= 4;
n_timepoints = 1200;
T_subject{1} = repmat(n_timepoints,n_sessions,1)';
T = repmat(T_subject,size(f,1),1);
%r = str2num(r);

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

%mydir = '/home/ben/Documents/MATLAB/git_repos/'; % set to your directory
mydir = 'C:\Users\au699373\OneDrive - Aarhus Universitet\Dokumenter\MATLAB\'; % set to your directory
addpath(genpath([ mydir 'HMM-MAR_repos'])) % HMM repository
addpath(genpath([ mydir 'HMM-MAR_BG'])) % HMM repository
addpath(genpath([ mydir 'NetsPredict'])) % HMM repository


%mydir = 'C:\Users\au699373\OneDrive - Aarhus Universitet\Dokumenter\MATLAB\'; % set to your directory
%addpath(genpath([ mydir 'HMM-MAR_repos'])) % HMM repository
%addpath(genpath([ mydir 'HMMMAR_BG'])) % HMM repository

ICAdim = 50; % number of ICA components (25,50,100,etc)
%K = 8; % no. states
covtype = 'uniquefull'; % type of covariance matrix
%zeromean = 1;
zeromean = 0



%DirOut = '/Users/bengriffin/Library/CloudStorage/OneDrive-AarhusUniversitet/Dokumenter/MATLAB/HMMMAR_BG/HMMMAR Results/Test/'; % Test folder
%DirOut = 'C:\Users\au699373\OneDrive - Aarhus Universitet\Dokumenter\MATLAB\HMMMAR_BG\HMMMAR Results\FEB_2022\K_prior_grid_search_5\'; % Test folder
%DirOut = '/home/ben/Documents/MATLAB/git_repos/HMM-MAR-master/BG_neuroimage/All/Repetition_1/';
%DirOut = '/media/share/16.2/BG_data/Repetition_1/';
% 
% % We will save here the distance matrices
DirOut = 'C:\Users\au699373\OneDrive - Aarhus Universitet\Dokumenter\MATLAB\HMMMAR_BG\HMMMAR Results\MAR_2022\Investigating_variability\Repetition7_non_TDE\';

% 1-64
%K_vec = repmat(repelem([3 7 11 15],1,4),1,4)';
%hmm_train_dirichlet_diag_vec = repelem([10 10000000 10000000000000 10000000000000000000],1,16)';
%lags = repmat({-1:1, -3:3, -5:5, -15:15},1,16)';

% 65-80
%K_vec = repmat(repelem([3 7 11 15],1,4),1,5)';
%hmm_train_dirichlet_diag_vec = [repelem([10 10000000 10000000000000 10000000000000000000],1,16)'; repmat([10 10000000 10000000000000 10000000000000000000],1,4)'];
% 0 lags

% % 81-96
% K_vec = repmat(repelem([3 7 11 15],1,4),1,6)';
% hmm_train_dirichlet_diag_vec = [repelem([10 10000000 10000000000000 10000000000000000000],1,16)'; repmat([10 10000000 10000000000000 10000000000000000000],1,4)'; repmat([10 10000000 10000000000000 10000000000000000000],1,4)'];
% % change zeromean to 0

% 97-112
K_vec = repmat(repelem([3 7 11 15],1,4),1,7)';
hmm_train_dirichlet_diag_vec = [repelem([10 10000000 10000000000000 10000000000000000000],1,16)'; repmat([10 10000000 10000000000000 10000000000000000000],1,4)'; repmat([10 10000000 10000000000000 10000000000000000000],1,4)'; repmat([10 10000000 10000000000000 10000000000000000000],1,4)'];
% change zeromean to 0
% change covtype to 'uniquefull'

% Currently I have varied:
% - states - 3, 7, 11, 15
% - dirichlet_diag - 10, 10e6, 10e12, 10e18
% and I have done this for the TDE model with lags - 1, 3, 9, 15

% Now I want to repeat the variation of the states and dirichlet_diag (16
% models), but for different HMMs
% To do:
% 1) FC-HMM
% 2) Mean-FC HMM
% 3) Var-HMM
% 4) MAr-HMM


TR = 0.75; N = length(f);
% 
% Run the HMMs (5 repetitions, with states characterised by a covariance matrix)
tic
options = struct(); 
%options.K = K; % number of states
options.order = 0;
% no autoregressive components 
options.covtype = covtype;
options.zeromean = zeromean; 
options.Fs = 1/TR; 
options.standardise = 1;
%options.DirichletDiag = 10;
options.dropstates = 0;
options.cyc = 50;
options.initcyc = 5; 
options.initrep = 3; 
options.verbose = 0; 

%%%% TDE ADDED OPTIONS
%options.embeddedlags =  % THIS IS WHAT WE WILL VARY
% Also for HMM-TDE we need to make sure (defined above)
options.order = 0;
%options.covtype='full';
%options.zeromean=1;


% stochastic options 
options.BIGNbatch = round(N/30); 
options.BIGtol = 1e-7; 
options.BIGcyc = 100; 
options.BIGundertol_tostop = 5;
options.BIGforgetrate = 0.7; 
options.BIGbase_weights = 0.9;

options_singlesubj = struct(); %options_singlesubj.K = K; % number of states 
options_singlesubj.order = 0; % no autoregressive components
options_singlesubj.zeromean = zeromean; % don't model the mean
options_singlesubj.covtype = covtype;
options_singlesubj.Fs = 1/TR;
options_singlesubj.standardise = 1; 
%options_singlesubj.DirichletDiag = 10;
options_singlesubj.dropstates = 0; options_singlesubj.cyc = 100;
options_singlesubj.verbose = 0;

options_singlesubj_dr = options_singlesubj;
options_singlesubj_dr.updateGamma = 0;

% We run the HMM multiple times

disp(['Repetition ' num2str(r)])
K = K_vec(r);
options.K = K; %  number of states
options_singlesubj.K = K; % number of states
options.DirichletDiag = hmm_train_dirichlet_diag_vec(r);
options_singlesubj.DirichletDiag =  hmm_train_dirichlet_diag_vec(r);

% to run a TDE model
%options.embeddedlags = lags{r};
%options.pca = ICAdim*2;

% Run the HMM at the group level and get some statistics % (eg
%  fractional occupancy)
[hmm,Gamma,~,vpath,~,~,~] = hmmmar(f,T,options); FOgroup = zeros(N,K); % Fractional occupancy

% save stats
%     meanActivations = zeros(ICAdim,K); % maps
%     for j=1:N
%         ind = (1:4800) + 4800*(j-1);
%         FOgroup(j,:) = mean(Gamma(ind,:));
%         X= zscore(f{j}); %X = zscore(dlmread(f{j}));
%         for k = 1:K
%             meanActivations(:,k) = meanActivations(:,k) + ...
%                 sum(X .* repmat(Gamma(ind,k),1,ICAdim))';
%         end
%     end
%     meanActivations = meanActivations ./ repmat(sum(Gamma),ICAdim,1);
%     switchingRate = getSwitchingRate(Gamma,T,options);
%     maxFO = getMaxFractionalOccupancy(Gamma,T,options);

% Subject specific stuff (dual-estimation)
options_singlesubj.hmm = hmm;
FOdual = zeros(N,K);

parfor j = 1:N
    fprintf('Subject Number %i\n',j)
    X = f{j};%X = dlmread(f{j}); % dual-estimation
    [HMMs_dualregr{j},Gammaj{j}] = hmmdual(X,T{j},hmm);
    for k = 1:K
        HMMs_dualregr{j}.state(k).prior = [];
    end
    FOdual(j,:) = mean(Gammaj{j});
end


save([DirOut 'HMMs_r' num2str(r) '_d' num2str(r)...
    '_GROUP.mat'], 'hmm','Gamma','vpath','FOgroup')%,'meanActivations','switchingRate','maxFO')
save([DirOut 'HMMs_r' num2str(r) '_d' num2str(r)...
    '_states_' num2str(K) '.mat'],'HMMs_dualregr','Gammaj','FOdual', '-v7.3')


%
% % Create distance matrices between models
% clc
% % between HMMs
% DistHMM = zeros(N,N,repetitions); % subj x subj x repetitions
% 
% %for r = 1:repetitions
%     K = K_vec(r);
%     disp(['Repetition ' num2str(r) ])
%     for d = r%1:dirichlet_test
%         %out = load([DirOut 'HMMs_r' num2str(r) '_states_' num2str(K) '.mat']);
%         out = load([DirOut 'HMMs_r' num2str(r) '_d' num2str(d)  '_states_' num2str(K) '.mat']);
%         for n1 = 1:N-1
%             for n2 = n1+1:N
%                 % FO is contained in TPC; TPC is contained in HMM
%                 %             DistHMM_n1n2(n1,n2,r) = hmm_kl(out.HMMs_dualregr{n1},out.HMMs_dualregr{n2})/2;
%                 %             DistHMM_n2n1(n1,n2,r) = hmm_kl(out.HMMs_dualregr{n2},out.HMMs_dualregr{n1})/2;
% 
%                 %             DistHMM(n1,n2,r) = (hmm_kl(out.HMMs_dualregr{n1},out.HMMs_dualregr{n2}) ...
%                 %                 + hmm_kl(out.HMMs_dualregr{n2},out.HMMs_dualregr{n1}))/2;
%                 DistHMM(n1,n2,r) = (hmm_kl(out.HMMs_dualregr{n1},out.HMMs_dualregr{n2}) ...
%                     + hmm_kl(out.HMMs_dualregr{n2},out.HMMs_dualregr{n1}))/2;
%                 DistHMM(n2,n1,r) = DistHMM(n1,n2,r);
% 
%                 %             DistHMM_n1n2(n2,n1,r) = DistHMM_n1n2(n1,n2,r);
%                 %             DistHMM_n2n1(n2,n1,r) = DistHMM_n2n1(n1,n2,r);
% 
%             end
%         end
%         disp(num2str(r))
%     end
% %end
% DistMat = DistHMM;
% 
% % Create correlation matrix for each subject
% corr_mat = zeros(ICAdim,ICAdim,N);
% for s = 1:N % for each subject
%     sub = data{s};
%     for i = 1:ICAdim
%         for j = 1:ICAdim
%             corr_coeff = corrcoef(sub(:,i),sub(:,j));
%             corr_mat(i,j,s) = corr_coeff(2,1);
%         end
%     end
%     disp(['Subject no. ' num2str(s) ])
% end
% V = corr_mat;
% 
% % between static FC matrices
% DistStatic = zeros(N);
% for n1 = 1:N-1
%     sprintf('Static - vars %i', n1)
%     for n2 = n1+1:N
%         DistStatic(n1,n2) = ( wishart_kl(V(:,:,n1),V(:,:,n2),sum(T{n1}),sum(T{n2})) + ...
%             wishart_kl(V(:,:,n2),V(:,:,n1),sum(T{n2}),sum(T{n1})) ) /2;
%         DistStatic(n2,n1) = DistStatic(n1,n2);
%     end
%     disp(['Subject no. ' num2str(n1) ])
% end
% 
% save([DirOut 'KLdistances_ICA' num2str(ICAdim) '_r' num2str(r) '.mat'],'DistMat','DistStatic', '-v7.3');
% %save([DirOut 'KLdistances_ICA' num2str(ICAdim) 'r' num2str(r)  '.mat'],'DistMat', '-v7.3');
% 
% 
% %% Predictions of behaviour using structurals
% % The code here is a bit complex but what matters is the calls to
% % predictPhenotype.m
% %
% % prediction parameters
% parameters_prediction.verbose = 0;
% parameters_prediction = struct();
% parameters_prediction.method = 'KRR';
% %parameters_prediction.alpha = [0.1 0.5 1.0 5];
% parameters_prediction.alpha = [1.0 5];
% parameters_prediction.sigmafact = [1/2 1 2];
% %
% 
% %% Predictions of behaviour using the static FC (with and without structural deconfounding) 
% % The static FC is used only through the distance matrices computed previously
% 
% D = DistStatic; 
% explained_variance = NaN(size(vars,2),1);
% vars_hat = NaN(N,size(vars,2)); % the predicted variables
% 
% for j = 1:size(vars,2)
%     sprintf('Static - vars %i', j)
%     y = vars(:,j); % here probably you need to remove subjects with missing values
% 
%     y_new = y;
%     D_new = D;
%     conf_new = conf;
%     twins_new = twins;
% 
%     % BG code to remove subjects with missing values
%     non_nan_idx = find(~isnan(y));
%     which_nan = isnan(y);
%     if any(which_nan)
%         y_new = y(~which_nan);
%         D_new = D(~which_nan,~which_nan);
%         conf_new = conf(~which_nan,:);
%         twins_new = twins(~which_nan,~which_nan);
%         warning('NaN found on Yin, will remove...')
%     end
% 
% 
%     [yhat,~] = predictPhenotype(y_new,D_new,parameters_prediction,twins_new,conf_new);
%     explained_variance(j) = corr(squeeze(yhat),y_new).^2;
%     vars_hat(non_nan_idx,j) = yhat;
% end
% 
% save([DirOut 'staticFC_predictions_r' num2str(r) '.mat'],'vars_hat','explained_variance')
% 
% 
% 
% % Predictions of behaviour using the HMMs (with and without structural deconfounding)
% %rng('default')
% %n_folds = 430;
% 
% %load([DirOut 'KLdistances_ICA' num2str(ICAdim) 'r' num2str(r)  '.mat'],'DistMat')
% % load([DirOut 'KLdistances_ICA50_NOISE.mat'],'DistMat_flat_NOISE');
% 
% %DistHMM = DistMat_flat_NOISE;
% DistHMM = DistMat;
% explained_variance = NaN(size(vars,2),repetitions);
% vars_hat = NaN(N,size(vars,2),repetitions);
% explained_variance_krr_predict = NaN(size(vars,2),repetitions);
% vars_hat_krr_predict = NaN(N,size(vars,2),repetitions);
% folds_all = cell(10,34);
% 
% 
% %for r = 1:repetitions
% disp(['Repetition ' num2str(r) ])
% for d = r%1:dirichlet_test
%     %r_track = r_vec(r);
%     %d_track = d_vec(d);
%     %l_track = l_vec(l);
%     D = DistHMM(:,:,r);
%     for j = 1:size(vars,2)
%         disp(['Vars ' num2str(j) ])
% 
%         y = vars(:,j); % here probably you need to remove subjects with missing values
% 
%         y_new = y;
%         D_new = D;
%         conf_new = conf;
%         twins_new = twins;
% 
%         % BG code to remove subjects with missing values
%         non_nan_idx = find(~isnan(y));
%         which_nan = isnan(y);
%         if any(which_nan)
%             y_new = y(~which_nan);
%             D_new = D(~which_nan,~which_nan);
%             conf_new = conf(~which_nan,:);
%             twins_new = twins(~which_nan,~which_nan);
%             warning('NaN found on Yin, will remove...')
%         end
% 
%         % Make predictions and save them
%         parameters_prediction.method = 'KRR'; % using KRR
%         %[predictedY,predictedYD,YD,stats,beta_all,folds] = predictPhenotype(y_new,D_new,parameters_prediction,twins_new,conf_new);
%         [yhat,~,~,~,~,folds] = predictPhenotype(y_new,D_new,parameters_prediction,twins_new,conf_new);
%         folds_all(:,j) = folds;
% 
%         explained_variance(j,r) = corr(squeeze(yhat),y_new).^2;
%         vars_hat_store = NaN(size(vars,1),1);
%         vars_hat_store(non_nan_idx) = yhat;
%         vars_hat(:,j,r) = vars_hat_store;
% 
%         [yhat_krr_predict] = krr_predict(y_new,D_new,parameters_prediction,twins_new,[],conf_new);
%                             
%         explained_variance_krr_predict(j,r) = corr(squeeze(yhat_krr_predict),y_new).^2;
%         vars_hat_store = NaN(size(vars,1),1);
%         vars_hat_store(non_nan_idx) = yhat;
%         vars_hat_krr_predict(:,j,r) = vars_hat_store;
% 
% 
%     end
% end
% 
% %save([DirOut 'HMM_predictions_r' num2str(r) '.mat'],'vars_hat','explained_variance','folds_all')
% save([DirOut 'HMM_predictions_r' num2str(r) '.mat'],'vars_hat','explained_variance','vars_hat_krr_predict','explained_variance_krr_predict','folds_all')
toc
end
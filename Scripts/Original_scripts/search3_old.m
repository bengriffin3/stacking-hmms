function search3(r)
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
T = repmat(T_subject,size(vars,1),1);
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

mydir = 'C:\Users\au699373\OneDrive - Aarhus Universitet\Dokumenter\MATLAB\'; % set to your directory
addpath(genpath([ mydir 'HMM-MAR_repos'])) % HMM repository
addpath(genpath([ mydir 'HMMMAR_BG'])) % HMM repository

ICAdim = 50; % number of ICA components (25,50,100,etc)
%K = 8; % no. states
covtype = 'full'; % type of covariance matrix
zeromean = 1;


%DirData = [mydir 'data/HCP/TimeSeries/group1200/3T_HCP1200_MSMAll_d' num2str(ICAdim) '_ts2/'];
%DirOut = '/Users/bengriffin/Library/CloudStorage/OneDrive-AarhusUniversitet/Dokumenter/MATLAB/HMMMAR_BG/HMMMAR Results/Test/'; % Test folder
DirOut = 'C:\Users\au699373\OneDrive - Aarhus Universitet\Dokumenter\MATLAB\HMMMAR_BG\HMMMAR Results\FEB_2022\K_prior_grid_search_3\'; % Test folder
% We will save here the distance matrices


K_vec = repelem([3 5 7 9 11 13 15 17 19 21],3)';
%hmm_train_dirichlet_diag_vec = [10 10000 10000000 10000000000 10000000000000 10000000000000000 10000000000000000000 10000000000000000000000 10000000000000000000000000];
hmm_train_dirichlet_diag_vec = [10];

repetitions = length(K_vec); % to run it multiple times (keeping all the results)
dirichlet_test = length(hmm_train_dirichlet_diag_vec);
TR = 0.75; N = length(f); 

% Run the HMMs (5 repetitions, with states characterised by a covariance matrix)
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
    disp(['Repetition ' num2str(r) ])
    K = K_vec(r);
    options.K = K; % number of states
    options_singlesubj.K = K; % number of states

    for d = 1:dirichlet_test
    options_singlesubj.DirichletDiag = hmm_train_dirichlet_diag_vec(d);

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

    end
toc
% 
% Create distance matrices between models 
%clc
% between HMMs
DistHMM = zeros(N,N,repetitions,dirichlet_test); % subj x subj x repetitions
% % DistHMM_n1n2 = zeros(N,N,5);
% % DistHMM_n2n1 = zeros(N,N,5);
% 
% 
% 

%for r = 1:repetitions
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
                DistHMM(n1,n2,r,d) = (hmm_kl(out.HMMs_dualregr{n1},out.HMMs_dualregr{n2}) ...
                    + hmm_kl(out.HMMs_dualregr{n2},out.HMMs_dualregr{n1}))/2;
                DistHMM(n2,n1,r,d) = DistHMM(n1,n2,r,d);

                %             DistHMM_n1n2(n2,n1,r) = DistHMM_n1n2(n1,n2,r);
                %             DistHMM_n2n1(n2,n1,r) = DistHMM_n2n1(n1,n2,r);

            end
        end
        disp(num2str(r))
    end
%end
DistMat = DistHMM;



% Create correlation matrix for each subject
corr_mat = zeros(ICAdim,ICAdim,N);

% between static FC matrices
DistStatic = zeros(N);


save([DirOut 'KLdistances_ICA' num2str(ICAdim) 'r' num2str(r)  '.mat'],'DistMat','DistStatic');
% 
% 
% Predictions of behaviour using structurals
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

% Predictions of behaviour using the HMMs (with and without structural deconfounding) 
rng('default')
%n_folds = 430;

DistHMM = DistMat;
explained_variance = NaN(size(vars,2),repetitions,dirichlet_test);
vars_hat = NaN(N,size(vars,2),repetitions,dirichlet_test); % the predicted variables

%for r = 1:repetitions
disp(['Repetition ' num2str(r) ])
for d = 1:dirichlet_test
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

        % Make predictions and save them
        parameters_prediction.method = 'KRR'; % using KRR
        %[predictedY,predictedYD,YD,stats,beta_all,folds] = predictPhenotype(y_new,D_new,parameters_prediction,twins_new,conf_new);
        [yhat,~,~,~,~,~] = predictPhenotype(y_new,D_new,parameters_prediction,twins_new,conf_new);
        %[yhat,stats,~,~,beta] = predictPhenotype_NN(y_new,D_new,parameters_prediction,twins_new,conf_new);
        %         explained_variance_KRR(j,r) = corr(squeeze(yhat),y_new).^2;
        explained_variance(j,r,d) = corr(squeeze(yhat),y_new).^2;
        vars_hat_store = NaN(size(vars,1),1);
        vars_hat_store(non_nan_idx) = yhat;
        vars_hat(:,j,r,d) = vars_hat_store;
        %         vars_hat_KRR(:,j,r) = vars_hat_store;
        %beta_all(:,j,r,:) = beta; % need to sort this out because it
        %doesn't work when subjects are removed e.g. variable 1

        %         % Make predictions using k-nearest neighbours
        %         parameters_prediction.method = 'NN';
        %         [yhat,stats,~,~,beta] = predictPhenotype_NN(y_new,D_new,parameters_prediction,twins_new,conf_new);
        %         explained_variance_KNN(j,r) = corr(squeeze(yhat),y_new).^2;
        %         vars_hat_store = NaN(size(vars,1),1);
        %         vars_hat_store(non_nan_idx) = yhat;
        %         vars_hat_KNN(:,j,r) = vars_hat_store;
    end
end


save([DirOut 'HMM_predictions_r' num2str(r) '.mat'],'vars_hat','explained_variance')

end
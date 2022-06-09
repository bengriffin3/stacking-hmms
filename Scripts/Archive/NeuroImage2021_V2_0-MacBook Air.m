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
tic
build_regression_data_V2_0;
load hcp1003_RESTall_LR_groupICA50.mat
f = data(grotKEEP);
n_sessions= 4;
n_timepoints = 1200;
T_subject{1} = repmat(n_timepoints,n_sessions,1)';
T = repmat(T_subject,size(vars,1),1);
% 
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

%mydir = 'C:\Users\au699373\OneDrive - Aarhus Universitet\Dokumenter\MATLAB\'; % set to your directory
%addpath(genpath([ mydir 'HMM-MAR_repos'])) % HMM repository

ICAdim = 50; % number of ICA components (25,50,100,etc)
K = 8; % no. states
covtype = 'full'; % type of covariance matrix
zeromean = 1;
%repetitions = 63; % to run it multiple times (keeping all the results)

%DirData = [mydir 'data/HCP/TimeSeries/group1200/3T_HCP1200_MSMAll_d' num2str(ICAdim) '_ts2/'];
%DirOut = ['C:\Users\au699373\OneDrive - Aarhus Universitet\Dokumenter\MATLAB\HMMMAR_BG\HMMMAR Results\Test\']; % Test folder
DirOut = ['C:\Users\au699373\OneDrive - Aarhus Universitet\Dokumenter\MATLAB\HMMMAR_BG\HMMMAR Results\FC_HMM_zeromean_1_covtype_full_stack_vs_63_reps\']; % Test folder
% We will save here the distance matrices

TR = 0.75; N = length(f);

K_vec = [3 3 3 4 4 4 5 5 5 6 6 6 7 7 7 8 8 8 9 9 9 10 10 10 11 11 11 12 12 12 13 13 13 14 14 14 15 15 15 16 16 16 17 17 17 18 18 18 19 19 19 20 20 20 21 21 21 22 22 22 23 23 23];

repetitions = length(K_vec);
%% Run the HMMs (5 repetitions, with states characterised by a covariance matrix)

options = struct();
options.K = K; % number of states
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
options_singlesubj.K = K; % number of states
options_singlesubj.order = 0; % no autoregressive components
options_singlesubj.zeromean = zeromean; % don't model the mean
options_singlesubj.covtype = covtype;
options_singlesubj.Fs = 1/TR;
options_singlesubj.standardise = 1;
options_singlesubj.DirichletDiag = 10;
options_singlesubj.dropstates = 0;
options_singlesubj.cyc = 100;
options_singlesubj.verbose = 0;

options_singlesubj_dr = options_singlesubj;
options_singlesubj_dr.updateGamma = 0;

% We run the HMM multiple times
for r = 40:repetitions%1:repetitions
    disp(['Repetition ' num2str(r) ])
    K = K_vec(r)
    options.K = K; % number of states
    options_singlesubj.K = K; % number of states

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
    
    save([DirOut 'HMMs_r' num2str(r) '_GROUP.mat'],...
        'hmm','Gamma','vpath','FOgroup','meanActivations','switchingRate','maxFO','fehist')
    save([DirOut 'HMMs_r' num2str(r) '_states_' num2str(K) '.mat'],...
        'HMMs_dualregr','Gammaj','FOdual')
        %'HMMs','HMMs_dualregr','FO','FOdual')
 
end

%% Create distance matrices between models 

% between HMMs
DistHMM = zeros(N,N,repetitions); % subj x subj x repetitions
for r = 40:48%1:repetitions
    K = K_vec(r)
    disp(['Repetition ' num2str(r) ])
    %out = load([DirOut 'HMMs_r' num2str(r)  '_states_' num2str(K) '.mat']);
    out = load(['HMMs_r' num2str(r)  '_states_' num2str(K) '.mat']);
    for n1 = 1:N-1
        for n2 = n1+1:N
            % FO is contained in TPC; TPC is contained in HMM
            DistHMM(n1,n2,r) = (hmm_kl(out.HMMs_dualregr{n1},out.HMMs_dualregr{n2}) ...
                + hmm_kl(out.HMMs_dualregr{n2},out.HMMs_dualregr{n1}))/2;
            DistHMM(n2,n1,r) = DistHMM(n1,n2,r);
        end
    end
    disp(num2str(r))
end
DistMat = DistHMM;

%%
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
        DistStatic(n2,n1) = DistStatic(n1,n2); % distance is symmetrical (i.e. distance between subject 1 and 2 is same as subject 2 and 1)
    end
    disp(['Subject no. ' num2str(n1) ])
end

save(['KLdistances_ICA' num2str(ICAdim) '.mat'],'DistMat','DistStatic');


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


% %% Predictions of behaviour using the static FC (with and without structural deconfounding) 
% % The static FC is used only through the distance matrices computed previously
% 
% Corrected_by_structure = 0; % set this to 0...3 
% 
% vars0 = vars; 
% 
% switch Corrected_by_structure
%     case 1
%         vars = vars - vars_hat;
%     case 2
%         vars = vars - vars_hat;
%     case 3
%         vars = vars - vars_hat;
% end
% 
% D = DistStatic;
% explained_variance = NaN(size(vars,2),1);
% vars_hat = NaN(N,size(vars,2)); % the predicted variables
% 
% 
% for j = 1:size(vars,2)
%     disp(['Variable ' num2str(j) ])
%     y = vars(:,j); % here probably you need to remove subjects with missing values
%     y_new = y;
%     D_new_static = D;
%     conf_new = conf;
%     twins_new = twins;
% 
%     % BG code to remove subjects with missing values
%     non_nan_idx = find(~isnan(y));
%     which_nan = isnan(y);
%     if any(which_nan)
%         y_new = y(~which_nan);
%         D_new_static = D(~which_nan,~which_nan);
%         conf_new = conf(~which_nan,:);
%         twins_new = twins(~which_nan,~which_nan);
%         warning('NaN found on Yin, will remove...')
%     end
% 
%      [yhat,stats] = predictPhenotype(y_new,D_new_static,parameters_prediction,twins_new,conf_new);
%      explained_variance(j) = corr(squeeze(yhat),y_new).^2;
%      vars_hat_store = NaN(size(vars,1),1);
%      vars_hat_store(non_nan_idx) = yhat;
%      vars_hat(:,j) = vars_hat_store;
% 
% 
% end
% 
% save('staticFC_predictions.mat','vars_hat','explained_variance')
% 
% vars = vars0; 

%% Predictions of behaviour using the HMMs (with and without structural deconfounding) 



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

DistHMM = DistMatKEEP;
explained_variance = NaN(size(vars,2),repetitions);
vars_hat = NaN(N,size(vars,2),repetitions); % the predicted variables


for r = 1%%:45%repetitions%1:repetitions
    disp(['Repetition ' num2str(r) ])
    D = DistHMM(:,:,r);
    for j = 1:size(vars,2)
        disp(['Vars ' num2str(j) ])

        y = vars(:,j); % here probably you need to remove subjects with missing values

        y_new = y;
        D_new_HMM = D;
        conf_new = conf;
        twins_new = twins;
        
        % BG code to remove subjects with missing values
        non_nan_idx = find(~isnan(y));
        which_nan = isnan(y);
        if any(which_nan)
            y_new = y(~which_nan);
            D_new_HMM = D(~which_nan,~which_nan);
            conf_new = conf(~which_nan,:);
            twins_new = twins(~which_nan,~which_nan);
            warning('NaN found on Yin, will remove...')
        end

        % Make predictions and save them
        [yhat,stats] = predictPhenotype(y_new,D_new_HMM,parameters_prediction,twins_new,conf_new);
        explained_variance(j,r) = corr(squeeze(yhat),y_new).^2;
        vars_hat_store = NaN(size(vars,1),1);
        vars_hat_store(non_nan_idx) = yhat;
        vars_hat(:,j,r) = vars_hat_store;
    end
end

% save('HMM_predictions.mat','vars_hat','explained_variance')

vars = vars0;

%% METADATA AT GROUP LEVEL
maxFO_all_reps = zeros(N,repetitions);
% fehist_all_reps = zeros(100,repetitions);
K_all_reps = zeros(N,repetitions);
switchingRate_all_reps = zeros(N,repetitions);
%vpath_all_reps = cell(N,repetitions);
%Gamma_all_reps = cell(N,repetitions);
for i = 1:repetitions
    i
    load (['HMMs_r' num2str(i) '_GROUP.mat'])

%     if length(fehist) == 99
%         fehist = [ fehist fehist(end)];
%     end
    for j = 1:N
        K_all_reps(j,i) = length(unique(vpath(((4800*j)-4799):4800*j)));
        %vpath_all_reps{j,i} = vpath(((4800*j)-4799):4800*j);
        %Gamma_all_reps{j,i} = Gamma(((4800*j)-4799):4800*j,:);

    end
    maxFO_all_reps(:,i) = maxFO;
%     fehist_all_reps(:,i) = fehist;
    %K_all_reps(i) = hmm.K;
    switchingRate_all_reps(:,i) = switchingRate;
end
save(['HMMs_meta_data.mat'],...
    'maxFO_all_reps','K_all_reps','switchingRate_all_reps','vpath_all_reps','Gamma_all_reps')

%% METADATA AT SUBJECT LEVEL

% Gammaj_subject = cell(1001,5);
% FOdual_subject = cell(5,1);
Entropy_subject = NaN(1001,45);
likelihood_subject = NaN(1001,45);
K_vec = [3 3 3 4 4 4 5 5 5 6 6 6 7 7 7 8 8 8 9 9 9 10 10 10 11 11 11 12 12 12 13 13 13 14 14 14 15 15 15 16 16 16 17 17 17 18 18 18 19 19 19 20 20 20 21 21 21 22 22 22 23 23 23];
for i = 41:45
    i
    K = K_vec(i);
    load(['HMMs_r' num2str(i)  '_states_' num2str(K) '.mat']);
%     Gammaj_subject(:,i) = Gammaj;
%     FOdual_subject{i} = FOdual;

    for j = 1:1001
        j
        FOdual_sub = FOdual(j,:);
        Entropy_subject(j,i) = -sum(FOdual_sub.*log2(FOdual_sub));
        [fe,ll] = hmmfe_single_subject(f{j},T{j},HMMs_dualregr{j},Gammaj{j});
        likelihood_subject(j,i) = -sum(ll);
    end

end

save(['HMMs_meta_data_subject.mat'],...
    'Entropy_subject','likelihood_subject')

%% Predictions of behaviour using the HMMs (STACKED + METAFEATURES)
clc
tic
%rng('default') % set seed for reproducibility
build_regression_data_V2_0;
vars0 = vars;
%%% LOAD FOR PREDICTION
parameters_prediction = struct();
parameters_prediction.verbose = 0;
parameters_prediction.method = 'KRR';
parameters_prediction.alpha = [0.1 0.5 1.0 5];
parameters_prediction.sigmafact = [1/2 1 2];
repetitions = 5;


%%% LOAD DISTANCE MATRICES, METADATA, AND STATIC PREDICTIONS
%DirOut = ['C:\Users\au699373\OneDrive - Aarhus Universitet\Dokumenter\MATLAB\HMMMAR_BG\HMMMAR Results\FC_HMM_zeromean_1_covtype_full_stack_vs_63_reps\']; % Test folder
DirOut = ['Users/bengriffin/OneDrive - Aarhus Universitet/Dokumenter/MATLAB/HMMMAR_BG/HMMMAR Results/FC_HMM_zeromean_1_covtype_full_stack_vs_63_reps/']; % Test folder
load([ DirOut 'KLdistances_ICA50_BEST_5_reps.mat'])
DistMat = DistMatKEEP;
load([ DirOut 'HMMs_meta_data_subject_r_1_27_40_63.mat'])
%load([ DirOut 'HMMs_meta_data_GROUP.mat'])
load([ DirOut 'staticFC_predictions.mat'])
explained_variance_static = explained_variance;

%%%%%%%%%%%%%%%% METAFEATURES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n_subjects = size(vars,1); % number of subjects
MF1 = ones(size(vars,1),repetitions); % first metafeature is the constant one
% MF2 = K_all_reps; % Max possible K for repetitions = (3,5,8,11,14)
% MF3 = maxFO_all_reps;
% MF4 = switchingRate_all_reps;
MF5 = Entropy_subject_all(:,[2,6,14,25,44]);
MF6 = likelihood_subject_all(:,[2,6,14,25,44]);

%metafeatures = [MF1 log(MF2) MF3 exp(MF4)];
%metafeatures_all = {MF1, MF2, MF3, MF4, MF5, MF6};
%metafeatures_all = {MF1, log10(MF2), MF3, MF4, log(MF5), log10(log10(MF6))};
metafeatures_all = {MF1,log(MF5), log10(log10(MF6))};

%metafeatures = [MF1 MF3];
% n_metafeatures = size(metafeatures,2)/repetitions;
%%
for mf = 2%:3%:6
mf = 2;
    mf;

     %metafeatures = [metafeatures_all{1} normalize(metafeatures_all{mf})]; %%% NORMALIZE THE DATA
     metafeatures = [metafeatures_all{1} metafeatures_all{mf}];
     n_metafeatures = size(metafeatures,2)/repetitions;
%     metafeatures(:,6:end) = ((metafeatures(:,6:end) - mean(metafeatures(:,6:end)))./std(metafeatures(:,6:end)))+1

    DistHMM = DistMat;
    % Initialise variables (stacking)
    W_stack = NaN(repetitions,10,1,size(vars,2)); % a weight for every repetition of the HMM, for every fold, for every intelligence feature
    W_stack_static = NaN(repetitions+1,10,1,size(vars,2));
    W_FWLS = NaN(repetitions*n_metafeatures,10,size(vars,2)); % a weight for every repetition of the HMM, for every fold, for every intelligence feature
    W_FWLS_norm = NaN(repetitions*n_metafeatures,10,size(vars,2)); % a weight for every repetition of the HMM, for every fold, for every intelligence feature
    W_ridge = NaN(repetitions,10,size(vars,2)); % a weight for every repetition of the HMM, for every fold, for every intelligence feature

    explained_variance_ST = NaN(size(vars,2),repetitions);
    explained_variance_stack = NaN(size(vars,2),1);
    explained_variance_stack_static = NaN(size(vars,2),1);
    explained_variance_FWLS = NaN(size(vars,2),1);
    explained_variance_FWLS_norm = NaN(size(vars,2),1);
    explained_variance_ridge = NaN(size(vars,2),1);

    vars_hat_ST = NaN(size(vars,1),size(vars,2),repetitions); % each repetition of HMM prediction
    vars_hat_stack = NaN(size(vars,1),size(vars,2),1); % stacked prediction
    vars_hat_FWLS = NaN(size(vars,1),size(vars,2),repetitions); % metafeatures prediction

    sigma_fold = NaN(size(vars,2),10,repetitions);
    alpha_fold = NaN(size(vars,2),10,repetitions);

    for j = 1:size(vars,2)
        disp(['Vars ' num2str(j) ])

        y = vars(:,j); % here probably you need to remove subjects with missing values

        y_new = y;
        D_new_HMM = DistHMM;
        D_new_static = DistStatic;
        conf_new = conf;
        twins_new = twins;
        metafeatures_new = metafeatures;

        % BG code to remove subjects with missing values
        non_nan_idx = find(~isnan(y));
        which_nan = isnan(y);
        if any(which_nan)
            y_new = y(~which_nan);
            D_new_HMM = DistHMM(~which_nan,~which_nan,:);
            D_new_static = DistStatic(~which_nan,~which_nan);
            conf_new = conf(~which_nan,:);
            twins_new = twins(~which_nan,~which_nan);
            metafeatures_new = metafeatures(~which_nan,:);
            warning('NaN found on Yin, will remove...')
        end
        D_new_rep_HMM = reshape(mat2cell(D_new_HMM, size(D_new_HMM,1), size(D_new_HMM,2), ones(1,size(D_new_HMM,3))),[repetitions 1 1]);

        % Stacked predictions
        [yhat_ST,yhat_stack,yhat_stack_static,yhat_FWLS,yhat_FWLS_norm,yhat_ridge,w_stack,w_stack_static,w_FWLS,w_FWLS_norm,w_ridge,sigma,alpha] = predictPhenotype_stack_meta_V2(y_new,D_new_rep_HMM,D_new_static,metafeatures_new,parameters_prediction,twins_new,conf_new);

        % weights to combine predictions
        W_stack(:,:,j) = w_stack;
        W_stack_static(:,:,j) = w_stack_static;
        W_FWLS(:,:,j) = w_FWLS;
        W_FWLS_norm(:,:,j) = w_FWLS_norm;
        W_ridge(:,:,j) = w_ridge;

        % Calculate explained variance
        explained_variance_ST(j,:) = corr(squeeze(yhat_ST),y_new).^2;
        explained_variance_stack(j) = corr(squeeze(yhat_stack),y_new).^2;
        explained_variance_stack_static(j) = corr(squeeze(yhat_stack_static),y_new).^2;
        explained_variance_FWLS(j) = corr(squeeze(yhat_FWLS),y_new).^2;
        explained_variance_FWLS_norm(j) = corr(squeeze(yhat_FWLS_norm),normalize(y_new)).^2;
        explained_variance_ridge(j) = corr(squeeze(yhat_ridge),y_new).^2;

 

        %         vars_hat_store_ST = NaN(size(vars,1),repetitions);
        %         vars_hat_store_ST(non_nan_idx,:) = yhat_ST;
        %         vars_hat_ST(:,j,:) = vars_hat_store_ST;
        %
        %         vars_hat_stack_store = NaN(size(vars,1),1);
        %         vars_hat_stack_store(non_nan_idx) = yhat_stack;
        %         vars_hat_stack(:,j) = vars_hat_stack_store;
        %
        %         vars_hat_FWLS_store = NaN(size(vars,1),1);
        %         vars_hat_FWLS_store(non_nan_idx) = yhat_FWLS;
        %         vars_hat_FWLS(:,j) = vars_hat_FWLS_store;




    end


    %save('HMM_predictions_stack.mat','explained_variance_ST','vars_hat_ST','explained_variance_ST_reps','vars_hat_ST_reps','W_stack','W_FWLS')

    vars = vars0;

    toc

    % Plot the original 5 repetitions and the new stacked + metafeature/stacked combined predictions
    X = 1:size(vars,2);
    figure
    scatter(X,explained_variance_FWLS,'o','r') % plot my stacked value with metafeature
    hold on
    scatter(X,explained_variance_FWLS_norm,'o','c') % plot my stacked value with metafeature
    %scatter(X,explained_variance_ridge,'o','m') % plot my stacked value with metafeature
    scatter(X,explained_variance_stack_static,'o','k') % plot Diego's stacked value  including static prediction
    scatter(X,explained_variance_static,'o','b') % plot Diego's stacked value  including static prediction
    scatter(X,explained_variance_stack,'o','g') % plot Diego's stacked value
    %colour_repetitions = ['b','b','b','b','b']
    colour_repetitions = {[30, 144, 255],[24, 125, 233],[18, 106, 210],[6, 67, 165],[0, 48, 143]};
    for m = 1:5
        %scatter(X,explained_variance_ST(:,m),'x','MarkerEdgeColor',colour_repetitions{m}/255,'MarkerFaceColor',colour_repetitions{m}/255);%colour_repetitions{m}) % plot the original 5 repetitions of HMM
        scatter(X,explained_variance_ST(:,m),'x')
    end
    % Format chart
    xlabel('Intelligence features'); ylabel('r^2')
    legend('FWLS predictor','FWLS normalized predictor','Stacked with static predictor','Static predictor','Stacked predictor','3 States','5 States','8 States','11 States','14 States')
    title(sprintf('Metafeature %i',mf))
end


%% Stack using ridge regression
rng('default') % set seed for reproducibility
build_regression_data_V2_0;
vars0 = vars;
%%% LOAD FOR PREDICTION
parameters_prediction = struct();
parameters_prediction.verbose = 0;
parameters_prediction.method = 'KRR';
parameters_prediction.alpha = [0.1 0.5 1.0 5];
parameters_prediction.sigmafact = [1/2 1 2];
repetitions = 5;


%%% LOAD DISTANCE MATRICES, METADATA, AND STATIC PREDICTIONS
%DirOut = ['C:\Users\au699373\OneDrive - Aarhus Universitet\Dokumenter\MATLAB\HMMMAR_BG\HMMMAR Results\FC_HMM_zeromean_1_covtype_full_stack_vs_63_reps\']; % Test folder
DirOut = ['Users/bengriffin/OneDrive - Aarhus Universitet/Dokumenter/MATLAB/HMMMAR_BG/HMMMAR Results/FC_HMM_zeromean_1_covtype_full_stack_vs_63_reps/']; % Test folder
load([ DirOut 'KLdistances_ICA50_BEST_5_reps.mat'])
DistMat = DistMatKEEP;
% load([ DirOut 'HMMs_meta_data_subject_r_1_27_40_63.mat'])
%load([ DirOut 'HMMs_meta_data_GROUP.mat'])
% load([ DirOut 'staticFC_predictions.mat'])
% explained_variance_static = explained_variance;
%load([ DirOut 'KLdistances_ICA50.mat'])


% n_subjects = size(vars,1); % number of subjects

DistHMM = DistMat;
% Initialise variables (stacking)
W_stack_ridge = NaN(repetitions,10,size(vars,2));
explained_variance_ST = NaN(size(vars,2),repetitions);
explained_variance_stack = NaN(size(vars,2),1);
explained_variance_stack_ridge = NaN(size(vars,2),1);

yhat_stack_store = cell(size(vars,2),1);
yhat_ST_store = cell(size(vars,2),1);

for j = 12%:size(vars,2)
    disp(['Vars ' num2str(j) ])
    
    y = vars(:,j); % here probably you need to remove subjects with missing values
    
    y_new = y;
    D_new_HMM = DistHMM;
    D_new_static = DistStatic;
    conf_new = conf;
    twins_new = twins;
    
    
    % BG code to remove subjects with missing values
    non_nan_idx = find(~isnan(y));
    which_nan = isnan(y);
    if any(which_nan)
        y_new = y(~which_nan);
        D_new_HMM = DistHMM(~which_nan,~which_nan,:);
        D_new_static = DistStatic(~which_nan,~which_nan);
        conf_new = conf(~which_nan,:);
        twins_new = twins(~which_nan,~which_nan);
        %metafeatures_new = metafeatures(~which_nan,:);
        warning('NaN found on Yin, will remove...')
    end
    D_new_rep_HMM = reshape(mat2cell(D_new_HMM, size(D_new_HMM,1), size(D_new_HMM,2), ones(1,size(D_new_HMM,3))),[repetitions 1 1]);
    
    % Stacked (ridge) predictions
    [yhat_stack_ridge,W_ridge,yhat_ST] = predictPhenotype_stack_ridge_V2(y_new,D_new_rep_HMM,parameters_prediction,twins_new,conf_new);
    %[yhat_stack_ridge,~,~,~,~,~,~,W_stack,~,yhat_ST] =predictPhenotype_stack_RIDGE_NEW(y_new,D_new_rep_HMM,parameters_prediction,twins_new,conf_new);
    %[yhat_stack,~,~,~,~,~,~,W,~,yhat_ST] = predictPhenotype_stack(y_new,D_new_rep_HMM,parameters_prediction,twins_new,conf_new);
    
    yhat_ST_store{j} = yhat_ST;
    yhat_stack_store{j} = yhat_stack;
    %yhat_stack_store{j} = yhat_stack_ridge;
    
    
    % weights to combine predictions
    %W_stack(:,:,j) = w_stack;
    %W_stack_ridge(:,:,j) = w_ridge;
    
    % Calculate explained variance
    explained_variance_ST(j,:) = corr(squeeze(yhat_ST),y_new).^2;
    explained_variance_stack(j) = corr(squeeze(yhat_stack),y_new).^2;
    %explained_variance_stack_ridge(j) = corr(squeeze(yhat_stack_ridge),y_new).^2;
    
    
end
%yhat_stack;
 
    % Plot the original 5 repetitions and the new stacked + metafeature/stacked combined predictions
    X = 1:size(vars,2);
    figure 
    scatter(X,explained_variance_stack,'o','b') % plot Diego's stacked value
    hold on
    for m = 1:5
        scatter(X,explained_variance_ST(:,m),'x')
    end
    % Format chart
    xlabel('Intelligence features'); ylabel('r^2')


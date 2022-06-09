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
n_sessions= 4;% Code used in Vidaurre et al. (2021) NEuroImage
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


%DirData = [mydir 'data/HCP/TimeSeries/group1200/3T_HCP1200_MSMAll_d' num2str(ICAdim) '_ts2/'];
%DirOut = '/Users/bengriffin/Library/CloudStorage/OneDrive-AarhusUniversitet/Dokumenter/MATLAB/HMMMAR_BG/HMMMAR Results/Test/'; % Test folder
DirOut = 'C:\Users\au699373\OneDrive - Aarhus Universitet\Dokumenter\MATLAB\HMMMAR_BG\HMMMAR Results\FEB_2022\Test\'; % Test folder
%DirOut = 'C:\Users\au699373\OneDrive - Aarhus Universitet\Dokumenter\MATLAB\HMMMAR_BG\HMMMAR Results\FEB_2022\K_prior_grid_search_4\'; % Test folder
% We will save here the distance matrices

%K_vec = [3 3 3 5 5 5 7 7 7 9 9 9 11 11 11 13 13 13 15 15 15];
K_vec = [9]
%K_vec = repelem([3 5 7 9 11 13 15],10)'


hmm_train_dirichlet_diag_vec = [10 10000 10000000 10000000000 10000000000000];% 10000000000000000 10000000000000000000 10000000000000000000000 10000000000000000000000000];
%hmm_train_dirichlet_diag_vec = repelem([10 100 1000 10000 100000 1000000 10000000 100000000 1000000000 10000000000],7)'

repetitions = length(K_vec); % to run it multiple times (keeping all the results)
dirichlet_test = length(hmm_train_dirichlet_diag_vec);
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
        disp(['Dir ' num2str(d) ])
        options.DirichletDiag = hmm_train_dirichlet_diag_vec(d);
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
%     save([DirOut 'HMMs_r' num2str(r) '_GROUP.mat'],...
%        'hmm','Gamma','vpath','FOgroup','meanActivations','switchingRate','maxFO','fehist')
%     save([DirOut 'HMMs_r' num2str(r) '_states_' num2str(K) '.mat'],...
%        'HMMs_dualregr','Gammaj','FOdual')
    end
end
toc
%% Create distance matrices between models 
% %clc
% % between HMMs
% DistHMM = zeros(N,N,repetitions,dirichlet_test); % subj x subj x repetitions
% % DistHMM_n1n2 = zeros(N,N,5);
% % DistHMM_n2n1 = zeros(N,N,5);
% 
% 
% 
% parfor r = 5:repetitions
%     K = K_vec(r);
%     disp(['Repetition ' num2str(r) ])
%     for d = 7:dirichlet_test
%     %out = load([DirOut 'HMMs_r' num2str(r) '_states_' num2str(K) '.mat']);
%     out = load([DirOut 'HMMs_r' num2str(r) '_d' num2str(d)  '_states_' num2str(K) '.mat']);
%     for n1 = 1:1000%N-1
%         t = n1
%         for n2 = t+1:N
%             % FO is contained in TPC; TPC is contained in HMM
% %             DistHMM_n1n2(n1,n2,r) = hmm_kl(out.HMMs_dualregr{n1},out.HMMs_dualregr{n2})/2;
% %             DistHMM_n2n1(n1,n2,r) = hmm_kl(out.HMMs_dualregr{n2},out.HMMs_dualregr{n1})/2;
% 
% %             DistHMM(n1,n2,r) = (hmm_kl(out.HMMs_dualregr{n1},out.HMMs_dualregr{n2}) ...
% %                 + hmm_kl(out.HMMs_dualregr{n2},out.HMMs_dualregr{n1}))/2;
%             DistHMM(n1,n2,r,d) = (hmm_kl_BG(out.HMMs_dualregr{n1},out.HMMs_dualregr{n2}) ...
%                 + hmm_kl(out.HMMs_dualregr{n2},out.HMMs_dualregr{n1}))/2;
%             %DistHMM(n2,n1,r,d) = DistHMM(n1,n2,r,d);
% 
% %             DistHMM_n1n2(n2,n1,r) = DistHMM_n1n2(n1,n2,r);
% %             DistHMM_n2n1(n2,n1,r) = DistHMM_n2n1(n1,n2,r);
% 
%         end
%     end
%     disp(num2str(r))
%     end
% end
% DistMat = DistHMM;

clc
DistHMM = cell(repetitions,1); % subj x subj x repetitions

parfor r = 1:repetitions
    K = K_vec(r);
    DistHMM{r} = zeros(N,N,dirichlet_test);
    disp(['Repetition ' num2str(r) ])
    for d = 1:dirichlet_test
        disp(['Dir ' num2str(d) ])
        out = load([DirOut 'HMMs_r' num2str(r) '_d' num2str(d)  '_states_' num2str(K) '.mat']);

        for n1 = 1:N
            for n2 = n1+1:N
                    DistHMM{r}(n1,n2,d) = (hmm_kl_BG(out.HMMs_dualregr{n1},out.HMMs_dualregr{n2}) ...
                    + hmm_kl(out.HMMs_dualregr{n2},out.HMMs_dualregr{n1}))/2;

               % 
                %DistHMM(n2,n1,r,d) = DistHMM(n1,n2,r,d);

            end
        end
    end
end

% Distances between HMM distributions are symmetric (can maybe do this
% without for loop?)
for r = 1:repetitions
    for d = 1:dirichlet_test
        Dist_refl = DistHMM{r}(:,:,d);
        Dist_refl = Dist_refl + (triu(Dist_refl))';
        %y_new(:,:,i) = y_new(:,:,i) + (triu(x_new(:,:,i)))';

        DistHMM{r}(:,:,d) = Dist_refl;
    end
end

DistMat = DistHMM;

%%
% Create correlation matrix for each subject
corr_mat = zeros(ICAdim,ICAdim,N);
% for s = 1:N %for each subject
%     sub = data{s};
%     for i = 1:ICAdim
%         for j = 1:ICAdim
%             corr_coeff = corrcoef(sub(:,i),sub(:,j));
%             corr_mat(i,j,s) = corr_coeff(2,1);
%         end
%     end
%     disp(['Subject no. ' num2str(s) ])
% end
V = corr_mat;


% between static FC matrices
DistStatic = zeros(N);
% for n1 = 1:N-1
%     for n2 = n1+1:N
%         DistStatic(n1,n2) = ( wishart_kl(V(:,:,n1),V(:,:,n2),sum(T{n1}),sum(T{n2})) + ...
%             wishart_kl(V(:,:,n2),V(:,:,n1),sum(T{n2}),sum(T{n1})) ) /2;
%         DistStatic(n2,n1) = DistStatic(n1,n2);
%     end
%     disp(['Subject no. ' num2str(n1) ])
% end

save([DirOut 'KLdistances_ICA' num2str(ICAdim) '.mat'],'DistMat','DistStatic');


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
%      [yhat,stats] = predictPhenotype(y_new,D_new,parameters_prediction,twins_new,conf_new);
%      explained_variance(j) = corr(squeeze(yhat),y_new).^2;
%      vars_hat_store = NaN(size(vars,1),1);
%      vars_hat_store(non_nan_idx) = yhat;
%      vars_hat(:,j) = vars_hat_store;
% 
% 
% end
% 
% save([DirOut 'staticFC_predictions.mat'],'vars_hat','explained_variance')
% 
% vars = vars0; 

%% Predictions of behaviour using the HMMs (with and without structural deconfounding) 
rng('default')
%n_folds = 430;

DistHMM = DistMat;
explained_variance = NaN(size(vars,2),repetitions,dirichlet_test);
vars_hat = NaN(N,size(vars,2),repetitions,dirichlet_test); % the predicted variables

% explained_variance_KRR = NaN(size(vars,2),repetitions);
% vars_hat_KRR = NaN(N,size(vars,2),repetitions); % the predicted variables
%beta_all = NaN(N,size(vars,2),repetitions,n_folds);

% let's try K-Nearest Neighbours



% explained_variance_KNN = NaN(size(vars,2),repetitions);
% vars_hat_KNN = NaN(N,size(vars,2),repetitions); % the predicted variables


for r = 1:repetitions
    disp(['Repetition ' num2str(r) ])
    for d = 7:dirichlet_test
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
        [yhat,stats,~,~,beta,folds] = predictPhenotype(y_new,D_new,parameters_prediction,twins_new,conf_new);
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
end

%save([DirOut 'HMM_predictions_KRR.mat'],'vars_hat_KRR','explained_variance_KRR')
%save([DirOut 'HMM_predictions_KNN.mat'],'vars_hat_KNN','explained_variance_KNN')

%vars = vars0; 

%explained_variance = [explained_variance_KRR explained_variance_KNN];
%vars_hat = cat(3,vars_hat_KRR, vars_hat_KNN);
save([DirOut 'HMM_predictions.mat'],'vars_hat','explained_variance')

%%
% which fold has 761 in
% for ifold = 1:430
%     a = folds{ifold};
%     if any(a(:) == 761)
%         ifold
%     end
% end
% it's in fold 404

%% METADATA AT GROUP LEVEL
%N = 1001; 
%DirOut = 'C:\Users\au699373\OneDrive - Aarhus Universitet\Dokumenter\MATLAB\HMMMAR_BG\HMMMAR Results\FEB_2022\Varying_states_12_30\';
%DirOut = '/Users/bengriffin/Library/CloudStorage/OneDrive-AarhusUniversitet/Dokumenter/MATLAB/HMMMAR_BG/HMMMAR Results/FEB_2022/K_prior_grid_search_2/';
%K_vec = repelem(3:2:17,3);
%K_vec = 12:30;
%repetitions = length(K_vec);
%hmm_train_dirichlet_diag_vec = [10 100 1000 10000 100000];
%hmm_train_dirichlet_diag_vec = 10;
%dirichlet_test = length(hmm_train_dirichlet_diag_vec);


maxFO_all_reps = zeros(N,repetitions,dirichlet_test);
K_all_reps = zeros(N,repetitions,dirichlet_test);
switchingRate_all_reps = zeros(N,repetitions,dirichlet_test);
%vpath_all_reps = cell(n_subjects,repetitions);
%Gamma_all_reps = cell(n_subjects,repetitions);
FO_metastate1 = NaN(N,repetitions,dirichlet_test);
FO_metastate2 = NaN(N,repetitions,dirichlet_test);
FO_PCs = NaN(N,max(K_vec)-1,repetitions,dirichlet_test);

% what about states that aren't part of either metastate (e.g. state 5 in
% Diego's paper?
% Well, cluster's were found using Ward's algorithm so I have done that - I
% assume Ward's would just exclude a state if it didn't fit with the two
% metastates?


for i = 1:repetitions
    i
    for d = 1:dirichlet_test
    
    % load fractional occupancy data
    K = K_vec(i);
    load([DirOut 'HMMs_r' num2str(i) '_d' num2str(d)  '_GROUP.mat']);
    %load([DirOut 'HMMs_r' num2str(i) '_GROUP.mat']);
    % Form FO correlation matrix
    R = FOgroup;
    C = corr(R);

    R_m = R-mean(R);
    covarianceMatrix = cov(R_m);
    [V,D] = eig(covarianceMatrix);

    %%%%%%%%%%%%%%%%%%%%%%%%% PCA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    [coeff,score,latent,~,explained] = pca(R');%,'NumComponents',2); % we only take the first PC
    [~,ord] = sort(score(:,1));
    
    %[coeff,score,latent,~,explained] = pca(R); [~,ord] = sort(score(1,:)); % we only take the first PC
   
    
    coeff; % these are the principle component vectors (the evectors of the covariance matrix)
    score; % these are the representations of R in the principal component space i.e. the coefficients of the linear combination of original variables that make up each new component
    latent; % these are the corresponding evalues
    explained; % see how much variance is explained by each PC

    % Display FO matrix (states reordered)
    %figure(); imagesc(C(ord,ord)); colorbar; % figure 2 (b)

    % Display transition probability matrix (states reordered)
    P = hmm.P;
    P(eye(size(P))==1) = nan;
    %figure(); imagesc(P(ord,ord)); colorbar; % figure 2(a)

    FO_PCs(:,1:K-1,i,d) = coeff; % need to sort out signs of these? + make sure saving the rg

    %%%%%%%%%%%%%%%%%%%%%%%%% PCA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%%%%%%%%%%%%%%%%%%%%% METASTATE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Detect heirarchical cluster (two clusters)
    Z = linkage(C,'ward');
    clusters_two = cluster(Z,'Maxclust',2);

    % Check to see if there should be a state excluded from the metastates
    clusters_three = cluster(Z,'Maxclust',3);
    [GC, GR] = groupcounts(clusters_three);
    states = (1:K_vec(i))';

    if length(states) == 3 % if 3 states set MS1 = state 1 and MS2 = state 3 because state 2 seems most excluded from MS (CHECK THIS)
        clusters = [1 2];
        states = [1 3];
    elseif min(GC) == 1
        sprintf('Warning, one state has been excluded from the metastates')
        single_cluster = GR(GC ==1); % identify cluster with one element
        statesKEEP = ~(clusters_three == single_cluster); % specify which states to keep (i.e. all except single_cluster)
        clusters = clusters_three(statesKEEP); % remove cluster with one element
        states = states(statesKEEP);
    else
        clusters = clusters_two;
    end

    %figure(); [H,T,outperm] = dendrogram(Z); % Display heirarchical structure and FO matrix (states reordered to group up metastates i.e. in order of dendogram)

    % Divide states into clusters (is this the correct way to find clusters?
    metastate1 = states(clusters==min(clusters));
    metastate2 = states(clusters==max(clusters));

    % Find FO of metastates
    FO_metastate1(:,i,d) = sum(R(:,metastate1),2);
    FO_metastate2(:,i,d) = sum(R(:,metastate2),2);
    %corr(FO_metastate1(:,i),vars, 'rows','complete')'
    %corr(FO_metastate2(:,i),vars, 'rows','complete')'
    

    %%%%%%%%%%%%%%%%%%%%%% METASTATE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

     for j = 1:N
         K_all_reps(j,i,d) = length(unique(vpath(((4800*j)-4799):4800*j)));
%         vpath_all_reps{j,i} = vpath(((4800*j)-4799):4800*j);
%         %Gamma_all_reps{j,i} = Gamma(((4800*j)-4799):4800*j,:);
%         %K_v_path(:,i) = max(vpath_all_reps{j,i});
% 
     end
    maxFO_all_reps(:,i,d) = maxFO;
    switchingRate_all_reps(:,i,d) = switchingRate;
    end
end


% % Check metatstate component signs
% corr_F01 = corr(FO_metastate1,'rows','complete');
% 
% % create new FO metastates with aligned metastates
% FO_metastate1(:,corr_F01(:,1)<0) = 0;
% FO_metastate2(:,corr_F01(:,1)>0) = 0;
% FO_metastate_new_1 = FO_metastate1 + FO_metastate2;
% FO_metastate_new_2 = 1-FO_metastate_new_1;
% 
% %metastate profile is defined as the FO of the cognitive metastate minus the FO of the sensorimotor metastate
% metastate_profile = FO_metastate_new_1 - FO_metastate_new_2;

%corr(metastate_profile,vars, 'rows','complete')'

% align PCs (note we only get K-1 number of PCs, so for 3 states we only get 2 PCs)
FO_PCs_aligned = NaN(size(FO_PCs));
for j = 1:(length(repetitions)-1) % for each PC
    for d = 1:length(dirichlet_test)
    FO_PCi = squeeze(FO_PCs(:,j,:,d));
    corr_FO_PC = corr(FO_PCi);
    idx = corr_FO_PC(:,end)<0.1; % use the final row to check alignment
    FO_PCi(:,idx) = FO_PCi(:,idx)*-1;
    FO_PCs_aligned(:,j,:,d) = FO_PCi;
    end
end

 save([DirOut 'HMMs_meta_data_GROUP_FINAL.mat'],...
    'maxFO_all_reps','K_all_reps','switchingRate_all_reps','FO_PCs_aligned')



 %%
 vpath_subject = NaN(N,repetitions,1200*4); % no. sessions = 4, no. time points = 1200
 for i = 1:repetitions
    i
    % load fractional occupancy data
    K = K_vec(i);
    load ([DirOut 'HMMs_r' num2str(i) '_GROUP.mat'])

     for j = 1:N
         vpath_subject(j,i,:) = vpath(((4800*j)-4799):4800*j);
     end

 end

 %% METADATA AT SUBJECT LEVEL
%Gammaj_subject = cell(1001,5); % computationally expensive - only store if going to use (maybe easier way to store? 
load([DirOut 'KLdistances_ICA50.mat'])

Entropy_subject_log2 = NaN(N,repetitions);
Entropy_subject_log10 = NaN(N,repetitions);
Entropy_subject_ln = NaN(N,repetitions);
likelihood_subject = NaN(N,repetitions);
Lifetimes_subject_state = NaN(N,repetitions,20);


FOdual_subject = NaN(N,max(K_vec));
for i = 1:repetitions
    i
    K = K_vec(i)
    load([DirOut 'HMMs_r' num2str(i)  '_states_' num2str(K) '.mat']);
    load ([DirOut 'HMMs_r' num2str(i) '_GROUP.mat']) % load group data (for vpath)
%     Gammaj_subject(:,i) = Gammaj;
    FOdual_subject(:,1:K) = FOdual;

    for j = 1:N
        j
        FOdual_sub = FOdual(j,:);
        Entropy_subject_log2(j,i) = -sum(FOdual_sub.*log2(FOdual_sub));
        Entropy_subject_log10(j,i) = -sum(FOdual_sub.*log10(FOdual_sub));
        Entropy_subject_ln(j,i) = -sum(FOdual_sub.*log(FOdual_sub));
        [fe,ll] = hmmfe_single_subject(f{j},T{j},HMMs_dualregr{j},Gammaj{j});
        likelihood_subject(j,i) = -sum(ll);
        
        % get state lifetimes (first note vpaths)
        vpath_subject(j,i,:) = vpath(((4800*j)-4799):4800*j);
        LifeTimes = getStateLifeTimes_BG(squeeze(vpath_subject(j,i,:)),T{j},K);         %LifeTimes = getStateLifeTimes(Gammaj{j},T{j});
        % !!! issue is if subject didn't enter state then this isn't noted !!!
        for k = 1:length(LifeTimes)
            Lifetimes_subject_state(j,i,k) = sum(LifeTimes{k});
        end
    end

end

subject_distance_sum = squeeze(sum(DistMat));
subject_distance_mean = squeeze(mean(DistMat));


save([DirOut 'HMMs_meta_data_subject.mat'],...
    'Entropy_subject_log2','Entropy_subject_log10','Entropy_subject_ln','FOdual_subject','likelihood_subject','subject_distance_sum','subject_distance_mean','Lifetimes_subject_state')


%%
n_bins = 20
for rep = 1:15
    figure(3)
    sgtitle(sprintf('Distribution of metafeature values'))
    subplot(15,1,rep)
    histogram(metastate_profile(:,rep),n_bins)
    xlabel('Metafeature'); ylabel('Frequency');
end


%% Reduce memory space by removing Gamma / Gammaj's

for i = 1:24%repetitions
    i
    K = K_vec(i)
    for d = 1:dirichlet_test
    load(['HMMs_r' num2str(i) '_d' num2str(d) '_states_' num2str(K) '.mat']);
    load (['HMMs_r' num2str(i) '_d' num2str(d)  '_GROUP.mat'])


    save(['HMMs_r' num2str(i) '_d' num2str(d) '_states_' num2str(K) '.mat'],'FOdual','HMMs_dualregr');
    save(['HMMs_r' num2str(i) '_d' num2str(d)  '_GROUP.mat'],'FOgroup','fehist','hmm','maxFO','meanActivations','switchingRate','vpath')


    end

end



%load(['HMMs_r' num2str(i) '_states_' num2str(K) '.mat']);
%load (['HMMs_r' num2str(i) '_GROUP.mat'])
%save(['HMMs_r' num2str(i) '_states_' num2str(K) '.mat'],'FOdual','HMMs_dualregr');
%save(['HMMs_r' num2str(i) '_GROUP.mat'],'FOgroup','fehist','hmm','maxFO','meanActivations','switchingRate','vpath')


%load([DirOut 'HMMs_r' num2str(i) '_d_' num2str(d) '_states_' num2str(K) '.mat']);
%load([DirOut 'HMMs_r' num2str(i) '_d_' num2str(d) '_GROUP.mat'])
%save([DirOut 'HMMs_r' num2str(i) '_d_' num2str(d) '_states_' num2str(K) '.mat'],'FOdual','HMMs_dualregr');
%save([DirOut 'HMMs_r' num2str(i) '_d_' num2str(d) '_GROUP.mat'],'FOgroup','fehist','hmm','maxFO','meanActivations','switchingRate','vpath')
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


%DirData = [mydir 'data/HCP/TimeSeries/group1200/3T_HCP1200_MSMAll_d' num2str(ICAdim) '_ts2/'];
%DirOut = '/Users/bengriffin/Library/CloudStorage/OneDrive-AarhusUniversitet/Dokumenter/MATLAB/HMMMAR_BG/HMMMAR Results/Test/'; % Test folder
DirOut = 'C:\Users\au699373\OneDrive - Aarhus Universitet\Dokumenter\MATLAB\HMMMAR_BG\HMMMAR Results\FEB_2022\K_prior_grid_search\'; % Test folder
% We will save here the distance matrices

K_vec = [3 5 7 9 11 13 15];
%hmm_train_dirichlet_diag_vec = [10]; 

%K_vec = [3 3 3 5 5 5 7 7 7 9 9 9 11 11 11 13 13 13 15 15 15 17 17 17 19 19 19 21 21 21];
hmm_train_dirichlet_diag_vec = [10 100 1000 10000];

repetitions = length(K_vec); % to run it multiple times (keeping all the results)
dirichlet_test = length(hmm_train_dirichlet_diag_vec);
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
%clc
% between HMMs
%DistHMM = zeros(N,N,repetitions,repetitions); % subj x subj x repetitions
% DistHMM_n1n2 = zeros(N,N,5);
% DistHMM_n2n1 = zeros(N,N,5);


for r = 1:repetitions
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
%      [yhat,stats] = predictPhenotype(y_new,D_new,parameters_prediction,twins_new,conf_new);
%      explained_variance(j) = corr(squeeze(yhat),y_new).^2;
%      vars_hat_store = NaN(size(vars,1),1);
%      vars_hat_store(non_nan_idx) = yhat;
%      vars_hat(:,j) = vars_hat_store;
% 
% 
% end
% 
% save([DirOut 'staticFC_predictions.mat'],'vars_hat','explained_variance')
% 
% vars = vars0; 

% Predictions of behaviour using the HMMs (with and without structural deconfounding) 
rng('default')
%n_folds = 430;

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
vars_hat = NaN(N,size(vars,2),repetitions,dirichlet_test); % the predicted variables

% explained_variance_KRR = NaN(size(vars,2),repetitions);
% vars_hat_KRR = NaN(N,size(vars,2),repetitions); % the predicted variables
%beta_all = NaN(N,size(vars,2),repetitions,n_folds);

% let's try K-Nearest Neighbours



% explained_variance_KNN = NaN(size(vars,2),repetitions);
% vars_hat_KNN = NaN(N,size(vars,2),repetitions); % the predicted variables


for r = 1:repetitions
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
        [yhat,stats,~,~,beta,folds] = predictPhenotype(y_new,D_new,parameters_prediction,twins_new,conf_new);
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
end

%save([DirOut 'HMM_predictions_KRR.mat'],'vars_hat_KRR','explained_variance_KRR')
%save([DirOut 'HMM_predictions_KNN.mat'],'vars_hat_KNN','explained_variance_KNN')

vars = vars0; 

%explained_variance = [explained_variance_KRR explained_variance_KNN];
%vars_hat = cat(3,vars_hat_KRR, vars_hat_KNN);
%save([DirOut 'HMM_predictions.mat'],'vars_hat','explained_variance')

%%
% which fold has 761 in
% for ifold = 1:430
%     a = folds{ifold};
%     if any(a(:) == 761)
%         ifold
%     end
% end
% it's in fold 404

%% METADATA AT GROUP LEVEL
N = 1001; 
DirOut = 'C:\Users\au699373\OneDrive - Aarhus Universitet\Dokumenter\MATLAB\HMMMAR_BG\HMMMAR Results\FEB_2022\K_prior_grid_search\';
%DirOut = '/Users/bengriffin/Library/CloudStorage/OneDrive-AarhusUniversitet/Dokumenter/MATLAB/HMMMAR_BG/HMMMAR Results/FEB_2022/K_prior_grid_search/';
%K_vec = repelem(3:2:17,3);
K_vec = [3 5 7 9 11 13 15];
repetitions = length(K_vec);
hmm_train_dirichlet_diag_vec = [10 100 1000 10000];
dirichlet_test = length(hmm_train_dirichlet_diag_vec);


maxFO_all_reps = zeros(N,repetitions,dirichlet_test);
K_all_reps = zeros(N,repetitions,dirichlet_test);
switchingRate_all_reps = zeros(N,repetitions,dirichlet_test);
%vpath_all_reps = cell(n_subjects,repetitions);
%Gamma_all_reps = cell(n_subjects,repetitions);
FO_metastate1 = NaN(N,repetitions,dirichlet_test);
FO_metastate2 = NaN(N,repetitions,dirichlet_test);
FO_PCs = NaN(N,max(K_vec)-1,repetitions,dirichlet_test);

% what about states that aren't part of either metastate (e.g. state 5 in
% Diego's paper?
% Well, cluster's were found using Ward's algorithm so I have done that - I
% assume Ward's would just exclude a state if it didn't fit with the two
% metastates?


for i = 1:repetitions
    i
    for d = 1:dirichlet_test
    
    % load fractional occupancy data
    K = K_vec(i);
    load([DirOut 'HMMs_r' num2str(i) '_d' num2str(d)  '_GROUP.mat']);
    % Form FO correlation matrix
    R = FOgroup;
    C = corr(R);

    R_m = R-mean(R);
    covarianceMatrix = cov(R_m);
    [V,D] = eig(covarianceMatrix);

    %%%%%%%%%%%%%%%%%%%%%%%%% PCA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    [coeff,score,latent,~,explained] = pca(R');%,'NumComponents',2); % we only take the first PC
    [~,ord] = sort(score(:,1));
    
    %[coeff,score,latent,~,explained] = pca(R); [~,ord] = sort(score(1,:)); % we only take the first PC
   
    
    coeff; % these are the principle component vectors (the evectors of the covariance matrix)
    score; % these are the representations of R in the principal component space i.e. the coefficients of the linear combination of original variables that make up each new component
    latent; % these are the corresponding evalues
    explained; % see how much variance is explained by each PC

    % Display FO matrix (states reordered)
    %figure(); imagesc(C(ord,ord)); colorbar; % figure 2 (b)

    % Display transition probability matrix (states reordered)
    P = hmm.P;
    P(eye(size(P))==1) = nan;
    %figure(); imagesc(P(ord,ord)); colorbar; % figure 2(a)

    FO_PCs(:,1:K-1,i,d) = coeff; % need to sort out signs of these? + make sure saving the rg

    %%%%%%%%%%%%%%%%%%%%%%%%% PCA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%%%%%%%%%%%%%%%%%%%%% METASTATE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Detect heirarchical cluster (two clusters)
    Z = linkage(C,'ward');
    clusters_two = cluster(Z,'Maxclust',2);

    % Check to see if there should be a state excluded from the metastates
    clusters_three = cluster(Z,'Maxclust',3);
    [GC, GR] = groupcounts(clusters_three);
    states = (1:K_vec(i))';

    if length(states) == 3 % if 3 states set MS1 = state 1 and MS2 = state 3 because state 2 seems most excluded from MS (CHECK THIS)
        clusters = [1 2];
        states = [1 3];
    elseif min(GC) == 1
        sprintf('Warning, one state has been excluded from the metastates')
        single_cluster = GR(GC ==1); % identify cluster with one element
        statesKEEP = ~(clusters_three == single_cluster); % specify which states to keep (i.e. all except single_cluster)
        clusters = clusters_three(statesKEEP); % remove cluster with one element
        states = states(statesKEEP);
    else
        clusters = clusters_two;
    end

    %figure(); [H,T,outperm] = dendrogram(Z); % Display heirarchical structure and FO matrix (states reordered to group up metastates i.e. in order of dendogram)

    % Divide states into clusters (is this the correct way to find clusters?
    metastate1 = states(clusters==min(clusters));
    metastate2 = states(clusters==max(clusters));

    % Find FO of metastates
    FO_metastate1(:,i,d) = sum(R(:,metastate1),2);
    FO_metastate2(:,i,d) = sum(R(:,metastate2),2);
    %corr(FO_metastate1(:,i),vars, 'rows','complete')'
    %corr(FO_metastate2(:,i),vars, 'rows','complete')'
    

    %%%%%%%%%%%%%%%%%%%%%% METASTATE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

     for j = 1:N
         K_all_reps(j,i,d) = length(unique(vpath(((4800*j)-4799):4800*j)));
%         vpath_all_reps{j,i} = vpath(((4800*j)-4799):4800*j);
%         %Gamma_all_reps{j,i} = Gamma(((4800*j)-4799):4800*j,:);
%         %K_v_path(:,i) = max(vpath_all_reps{j,i});
% 
     end
    maxFO_all_reps(:,i,d) = maxFO;
    switchingRate_all_reps(:,i,d) = switchingRate;
    end
end


% % Check metatstate component signs
% corr_F01 = corr(FO_metastate1,'rows','complete');
% 
% % create new FO metastates with aligned metastates
% FO_metastate1(:,corr_F01(:,1)<0) = 0;
% FO_metastate2(:,corr_F01(:,1)>0) = 0;
% FO_metastate_new_1 = FO_metastate1 + FO_metastate2;
% FO_metastate_new_2 = 1-FO_metastate_new_1;
% 
% %metastate profile is defined as the FO of the cognitive metastate minus the FO of the sensorimotor metastate
% metastate_profile = FO_metastate_new_1 - FO_metastate_new_2;

% align PCs (note we only get K-1 number of PCs, so for 3 states we only get 2 PCs)
FO_PCs_aligned = NaN(size(FO_PCs));
for j = 1:(repetitions-1) % for each PC
    j
    for d = 1:length(dirichlet_test)
    FO_PCi = squeeze(FO_PCs(:,j,:,d));
    corr_FO_PC = corr(FO_PCi);
    idx = corr_FO_PC(:,end)<0.1; % use the final row to check alignment
    FO_PCi(:,idx) = FO_PCi(:,idx)*-1;
    FO_PCs_aligned(:,j,:,d) = FO_PCi;
    end
end
j = 1; d = 1;
corr(squeeze(FO_PCs_aligned(:,j,:,d)))

%corr(metastate_profile,vars, 'rows','complete')'
 save([DirOut 'HMMs_meta_data_GROUP_FINAL.mat'],...
    'maxFO_all_reps','K_all_reps','switchingRate_all_reps','FO_PCs_aligned')



 %%
 vpath_subject = NaN(N,repetitions,1200*4); % no. sessions = 4, no. time points = 1200
 for i = 1:repetitions
    i
    % load fractional occupancy data
    K = K_vec(i);
    load ([DirOut 'HMMs_r' num2str(i) '_GROUP.mat'])

     for j = 1:N
         vpath_subject(j,i,:) = vpath(((4800*j)-4799):4800*j);
     end

 end

 %% METADATA AT SUBJECT LEVEL
%Gammaj_subject = cell(1001,5); % computationally expensive - only store if going to use (maybe easier way to store? 
load([DirOut 'KLdistances_ICA50.mat'])

Entropy_subject_log2 = NaN(N,repetitions);
Entropy_subject_log10 = NaN(N,repetitions);
Entropy_subject_ln = NaN(N,repetitions);
likelihood_subject = NaN(N,repetitions);
Lifetimes_subject_state = NaN(N,repetitions,20);


FOdual_subject = NaN(N,max(K_vec));
for i = 1:repetitions
    i
    K = K_vec(i)
    %load([DirOut 'HMMs_r' num2str(i)  '_states_' num2str(K) '.mat']);
    %load ([DirOut 'HMMs_r' num2str(i) '_GROUP.mat']) % load group data (for vpath)
    
    load([DirOut 'HMMs_r' num2str(i) '_d' num2str(d)   '_states_' num2str(K) '.mat']);
    load([DirOut 'HMMs_r' num2str(i) '_d' num2str(d)  '_GROUP.mat']);
%     Gammaj_subject(:,i) = Gammaj;
    FOdual_subject(:,1:K) = FOdual;

    for j = 1:N
        j
        FOdual_sub = FOdual(j,:);
        Entropy_subject_log2(j,i) = -sum(FOdual_sub.*log2(FOdual_sub));
        Entropy_subject_log10(j,i) = -sum(FOdual_sub.*log10(FOdual_sub));
        Entropy_subject_ln(j,i) = -sum(FOdual_sub.*log(FOdual_sub));
        [fe,ll] = hmmfe_single_subject(f{j},T{j},HMMs_dualregr{j},Gammaj{j});
        likelihood_subject(j,i) = -sum(ll);
        
        % get state lifetimes (first note vpaths)
        vpath_subject(j,i,:) = vpath(((4800*j)-4799):4800*j);
        LifeTimes = getStateLifeTimes_BG(squeeze(vpath_subject(j,i,:)),T{j},K);         %LifeTimes = getStateLifeTimes(Gammaj{j},T{j});
        % !!! issue is if subject didn't enter state then this isn't noted !!!
        for k = 1:length(LifeTimes)
            Lifetimes_subject_state(j,i,k) = sum(LifeTimes{k});
        end
    end

end

subject_distance_sum = squeeze(sum(DistMat));
subject_distance_mean = squeeze(mean(DistMat));


save([DirOut 'HMMs_meta_data_subject.mat'],...
    'Entropy_subject_log2','Entropy_subject_log10','Entropy_subject_ln','FOdual_subject','likelihood_subject','subject_distance_sum','subject_distance_mean','Lifetimes_subject_state')


%%
n_bins = 20
for rep = 1:15
    figure(3)
    sgtitle(sprintf('Distribution of metafeature values'))
    subplot(15,1,rep)
    histogram(metastate_profile(:,rep),n_bins)
    xlabel('Metafeature'); ylabel('Frequency');
end


%% Reduce memory space by removing Gamma / Gammaj's

for i = 1:24%repetitions
    i
    K = K_vec(i)
    for d = 1:dirichlet_test
    load(['HMMs_r' num2str(i) '_d' num2str(d) '_states_' num2str(K) '.mat']);
    load (['HMMs_r' num2str(i) '_d' num2str(d)  '_GROUP.mat'])


    save(['HMMs_r' num2str(i) '_d' num2str(d) '_states_' num2str(K) '.mat'],'FOdual','HMMs_dualregr');
    save(['HMMs_r' num2str(i) '_d' num2str(d)  '_GROUP.mat'],'FOgroup','fehist','hmm','maxFO','meanActivations','switchingRate','vpath')


    end

end



%load(['HMMs_r' num2str(i) '_states_' num2str(K) '.mat']);
%load (['HMMs_r' num2str(i) '_GROUP.mat'])
%save(['HMMs_r' num2str(i) '_states_' num2str(K) '.mat'],'FOdual','HMMs_dualregr');
%save(['HMMs_r' num2str(i) '_GROUP.mat'],'FOgroup','fehist','hmm','maxFO','meanActivations','switchingRate','vpath')


%load([DirOut 'HMMs_r' num2str(i) '_d_' num2str(d) '_states_' num2str(K) '.mat']);
%load([DirOut 'HMMs_r' num2str(i) '_d_' num2str(d) '_GROUP.mat'])
%save([DirOut 'HMMs_r' num2str(i) '_d_' num2str(d) '_states_' num2str(K) '.mat'],'FOdual','HMMs_dualregr');
%save([DirOut 'HMMs_r' num2str(i) '_d_' num2str(d) '_GROUP.mat'],'FOgroup','fehist','hmm','maxFO','meanActivations','switchingRate','vpath')



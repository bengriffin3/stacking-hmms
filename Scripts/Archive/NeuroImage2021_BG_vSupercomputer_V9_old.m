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
tic
load hcp1003_RESTall_LR_groupICA50.mat
f = data(grotKEEP);
n_sessions= 4;
n_timepoints = 1200;
T_subject{1} = repmat(n_timepoints,n_sessions,1)';
T = repmat(T_subject,size(vars,1),1);
% datadir = '/home/share/Data/HCP/';
% vars = dlmread([ datadir 'Behavioural/vars.txt'] ,' ');
% twins = dlmread([ datadir 'Behavioural/twins.txt'],' ');
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

mydir = '/home/ben/Documents/MATLAB/'; % set to your directory
addpath(genpath([ mydir 'HMM-MAR-master'])) % HMM repository

ICAdim = 50; % number of ICA components (25,50,100,etc)
K = 8; % no. states
%covtype = 'uniquediag'; % type of covariance matrix
repetitions = 5;%30; % to run it multiple times (keeping all the results)

%DirData = [mydir 'data/HCP/TimeSeries/group1200/3T_HCP1200_MSMAll_d' num2str(ICAdim) '_ts2/'];
%DirOut = [mydir 'experiments_HCP1200/hmms/ICA' num2str(ICAdim) '/K' num2str(K)];

%DirData = ['home/share/Data/HCP/rest/'];
%DirOut = [mydir 'git_repos/HMM-MAR-master/BG_neuroimage/Changing_models/25_10_2021/zeromean_1_cov_full_stacked/Archive/'];
DirOut = ['C:\Users\au699373\OneDrive - Aarhus Universitet\Dokumenter\MATLAB\HMMMAR_BG\HMMMAR Results\FC_HMM_zeromean_1_covtype_full_stack_ALL_vs/']; % Test folder
% We will save here the distance matrices

TR = 0.75; N = length(f); 

%% Run the HMMs (5 repetitions, with states characterised by a covariance matrix)

options = struct();
options.K = K; % number of states
options.order = 0; % no autoregressive components
options.covtype = 'full';%covtype;
options.zeromean = 1; %zeromean;
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
options_singlesubj.zeromean = 1;%zeromean; % don't model the mean
options_singlesubj.covtype = 'full';%covtype;
options_singlesubj.Fs = 1/TR;
options_singlesubj.standardise = 1;
options_singlesubj.DirichletDiag = 10;
options_singlesubj.dropstates = 0;
options_singlesubj.cyc = 100;
options_singlesubj.verbose = 0;

options_singlesubj_dr = options_singlesubj;
options_singlesubj_dr.updateGamma = 0;

fehist_all_reps = cell(repetitions,1);
 K_vec = [3 5 8 13 15];

% We run the HMM multiple times
for r = 1:repetitions
        disp(['Repetition ' num2str(r) ])
    
    K = K_vec(r);
    options.K = K;
    options_singlesubj.K = K;
    % Run the HMM at the group level and get some statistics
    % (eg fractional occupancy)
    [hmm,Gamma,~,~,~,~,fehist] = hmmmar(f,T,options);
    fehist_all_reps{r} = fehist; % free energy history
    FOgroup = zeros(N,K); % Fractional occupancy
    meanActivations = zeros(ICAdim,K); % maps
    for j=1:N
        ind = (1:4800) + 4800*(j-1);
        FOgroup(j,:) = mean(Gamma(ind,:));
        cat_reorder = zscore(f{j});%dlmread(f{j}));
        for k = 1:K
            meanActivations(:,k) = meanActivations(:,k) + ...
                sum(cat_reorder .* repmat(Gamma(ind,k),1,ICAdim))';
        end
    end
    meanActivations = meanActivations ./ repmat(sum(Gamma),ICAdim,1);
    switchingRate = getSwitchingRate(Gamma,T,options);
    maxFO = getMaxFractionalOccupancy(Gamma,T,options);
    
    disp(['Group level complete'])
    
    %Subject specific stuff (dual-estimation)
    options_singlesubj.hmm = hmm;
    FOdual = zeros(N,K);
    parfor j = 1:N
        cat_reorder = f{j};%dlmread(f{j});
        % dual-estimation
        [HMMs_dualregr{j},Gammaj] = hmmdual(cat_reorder,T{j},hmm);
        for k = 1:K
            HMMs_dualregr{j}.state(k).prior = [];
        end
        FOdual(j,:) = mean(Gammaj);
    end
    
    save([DirOut 'HMMs_r' num2str(r) '_GROUP_states_' num2str(K) '.mat'],...
        'hmm','FOgroup','meanActivations','switchingRate','maxFO', 'fehist')
    save([DirOut 'HMMs_r' num2str(r)  '_states_' num2str(K) '.mat'],...
        'HMMs_dualregr','FOdual')
    

end


%% Create distance matrices between models (i)

% between HMMs
DistHMM = zeros(N,N,repetitions); % subj x subj x repetitions
for r = 1:repetitions
    disp(num2str(r))
    K = K_vec(r)
    out = load([DirOut 'HMMs_r' num2str(r)  '_states_' num2str(K) '.mat']);
    for n1 = 1:N-1
        for n2 = n1+1:N
            % FO is contained in TPC; TPC is contained in HMM
            DistHMM(n1,n2,r) = (hmm_kl(out.HMMs_dualregr{n1},out.HMMs_dualregr{n2}) ...
                + hmm_kl(out.HMMs_dualregr{n2},out.HMMs_dualregr{n1}))/2;
            DistHMM(n2,n1,r) = DistHMM(n1,n2,r);
            % Here we computed the symmetric KL divergence between each
            % pair of (dual estiamte) subject HMMs
        end
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
        DistStatic(n2,n1) = DistStatic(n1,n2); % distance is symmetrical (i.e. distance between subject 1 and 2 is same as subject 2 and 1)
    end
    disp(['Subject no. ' num2str(n1) ])
end

save(['KLdistances_ICA' num2str(ICAdim) '.mat'],'DistMat','DistStatic');


%% Predictions of behaviour using structurals
% The code here is a bit complex but what matters is the calls to
% predictPhenotype.m

% prediction parameters
parameters_prediction = struct();
parameters_prediction.verbose = 0;
parameters_prediction.method = 'KRR';
parameters_prediction.alpha = [0.1 0.5 1.0 5];
parameters_prediction.sigmafact = [1/2 1 2];
%%
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

% Predictions of behaviour using the static FC (with and without structural deconfounding) 
% The static FC is used only through the distance matrices computed previously
clc
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
vars_hat = NaN(size(vars,1),size(vars,2)); % the predicted variables

% let's remove variables where we have more than 100 subjects with NaNs
% recorded
nan_variables = find(sum(isnan(vars))' > 100);
vars(:,nan_variables) = []; % these variables are so dodgy let's just ignore it

for k = 1:size(vars,2) %:length(intelligence_variable_vec) % only intelligence variables
        disp(['Var no. ' num2str(k) ])
    y = vars(:,k); % here probably you need to remove subjects with missing values
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
     explained_variance(k) = corr(squeeze(yhat),y_new).^2;
     vars_hat_store = NaN(size(vars,1),1);
     vars_hat_store(non_nan_idx) = yhat;
     vars_hat(:,k) = vars_hat_store;

end

save('staticFC_predictions.mat','vars_hat','explained_variance')

vars = vars0; 

%% Predictions of behaviour using the HMMs (with and without structural deconfounding) 
clc
%%% LOAD STUFF FOR PREDICTION
parameters_prediction = struct();
parameters_prediction.verbose = 0;
parameters_prediction.method = 'KRR';
parameters_prediction.alpha = [0.1 0.5 1.0 5];
parameters_prediction.sigmafact = [1/2 1 2];
repetitions = 5;

vars0 = vars;

Corrected_by_structure = 0; % set this to 0...3 

switch Corrected_by_structure
    case 1
        vars = vars - vars_hat;
    case 2
        vars = vars - vars_hat;
    case 3
        vars = vars - vars_hat;
end



DistHMM = DistMat;
% Initialise variables
explained_variance = NaN(size(vars,2),repetitions);
vars_hat = NaN(size(vars,1),size(vars,2),repetitions); % the predicted variables


% let's remove variables where we have more than 100 subjects with NaNs
% recorded
%nan_variables = find(sum(isnan(vars))' > 100);
%vars(:,nan_variables) = []; % this variable is so dodgy let's just ignore it

for r = 1:repetitions
    disp(['Repetition ' num2str(r) ])
    D = DistHMM(:,:,r);
    for k = 1:size(vars,2) %:length(intelligence_variable_vec) % only intelligence variables
        disp(['Vars ' num2str(k) ])
        y = vars(:,k); % here probably you need to remove subjects with missing values
        
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
        [yhat,stats] = predictPhenotype(y_new,D_new,parameters_prediction,twins_new,conf_new);
        explained_variance(k,r) = corr(squeeze(yhat),y_new).^2;
        vars_hat_store = NaN(size(vars,1),1);
        vars_hat_store(non_nan_idx) = yhat;
        vars_hat(:,k,r) = vars_hat_store;
    
    end

end

save('HMM_predictions.mat','vars_hat','explained_variance')

vars = vars0;


%% Predictions of behaviour using the HMMs (STACKED)
clc
vars0 = vars;
%%% LOAD STUFF FOR PREDICTION
parameters_prediction = struct();
parameters_prediction.verbose = 0;
parameters_prediction.method = 'KRR';
parameters_prediction.alpha = [0.1 0.5 1.0 5];
parameters_prediction.sigmafact = [1/2 1 2];
repetitions = 5;
load('KLdistances_ICA50')

DistHMM = DistMat;
% Initialise variables (stacking)
explained_variance_ST = NaN(size(vars,2),1);
vars_hat_ST = NaN(size(vars,1),size(vars,2),1); % the predicted variables
explained_variance_ST_reps = NaN(size(vars,2),repetitions);
vars_hat_ST_reps = NaN(size(vars,1),size(vars,2),repetitions);

W = NaN(repetitions,10,1);%size(vars,2)); % a weight for every repetition of the HMM, for every fold, for every intelligence feature

for k = 1:size(vars,2)
        disp(['Vars ' num2str(k) ])
        
        y = vars(:,k); % here probably you need to remove subjects with missing values
        
        y_new = y;
        D_new = DistHMM;
        conf_new = conf;
        twins_new = twins;

        % BG code to remove subjects with missing values
        non_nan_idx = find(~isnan(y));
        which_nan = isnan(y);
        if any(which_nan)
            y_new = y(~which_nan);
            D_new = DistHMM(~which_nan,~which_nan,:);
            conf_new = conf(~which_nan,:);
            twins_new = twins(~which_nan,~which_nan);
            warning('NaN found on Yin, will remove...')
        end

        D_new_rep= reshape(mat2cell(D_new, size(D_new,1), size(D_new,2), ones(1,size(D_new,3))),[repetitions 1 1]);
        [yhat_ST_BG,yhat_ST_reps_BG,w_BG] = predictPhenotype_stack_BG(y_new,D_new_rep,parameters_prediction,twins_new,conf_new);
      
        
        explained_variance_ST_BG(k) = corr(squeeze(yhat_ST_BG),y_new).^2;
        explained_variance_ST_reps_BG(k,:) = corr(squeeze(yhat_ST_reps_BG),y_new).^2;
        vars_hat_ST_store_BG = NaN(size(vars,1),1);
        vars_hat_ST_store_BG(non_nan_idx) = yhat_ST_BG;
        vars_hat_ST_BG(:,k) = vars_hat_ST_store_BG;

        vars_hat_store_ST_reps_BG = NaN(size(vars,1),repetitions);
        vars_hat_store_ST_reps_BG(non_nan_idx,:) = yhat_ST_reps_BG;
        vars_hat_ST_reps_BG(:,k,:) = vars_hat_store_ST_reps_BG;

        W_BG(:,:,k) = w_BG;

        [yhat_ST,~,~,~,~,~,~,w,~,yhat_ST_reps] = predictPhenotype_stack(y_new,D_new_rep,parameters_prediction,twins,conf_new);
        
        explained_variance_ST(k) = corr(squeeze(yhat_ST),y_new).^2;
        explained_variance_ST_reps(k,:) = corr(squeeze(yhat_ST_reps),y_new).^2;
        vars_hat_ST_store = NaN(size(vars,1),1);
        vars_hat_ST_store(non_nan_idx) = yhat_ST;
        vars_hat_ST(:,k) = vars_hat_ST_store;

        vars_hat_store_ST_reps = NaN(size(vars,1),repetitions);
        vars_hat_store_ST_reps(non_nan_idx,:) = yhat_ST_reps;
        vars_hat_ST_reps(:,k,:) = vars_hat_store_ST_reps;

        W(:,:,k) = w;
        

end
explained_variance_ST_BG = explained_variance_ST_BG';

save('HMM_predictions_stack_BG.mat','explained_variance_ST_BG','vars_hat_ST_BG','explained_variance_ST_reps_BG','vars_hat_ST_reps_BG','W_BG')
save('HMM_predictions_stack.mat','explained_variance_ST','vars_hat_ST','explained_variance_ST_reps','vars_hat_ST_reps','W')

vars = vars0;

toc

w
w_BG

%%
X = 1:size(vars,2);
blue_shades = {[2, 59, 112], [13, 60, 97],[0, 157, 255],[38, 171, 255],[140, 211, 255],};
figure
scatter(X,explained_variance_ST,'o','r'); hold on;
%scatter(X,explained_variance_ST_BG,'o','g')

for i = 1:5
    scatter(X,explained_variance(:,i),'x','MarkerEdgeColor',blue_shades{i}/255); hold on
end
legend('Stacked','HMM run 1','HMM run 2','HMM run 3','HMM run 4','HMM run 5')


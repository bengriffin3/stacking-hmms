% Bad prediction search
r = 2

mydir = 'C:\Users\au699373\OneDrive - Aarhus Universitet\Dokumenter\MATLAB\'; % set to your directory
addpath(genpath([ mydir 'HMM-MAR_repos'])) % HMM repository
addpath(genpath([ mydir 'HMMMAR_BG'])) % HMM repository
addpath(genpath([ mydir 'NetsPredict'])) % HMM repository


% Load options / data etc.
build_regression_data_V3;
load hcp1003_RESTall_LR_groupICA50.mat
f = data(grotKEEP);
n_sessions= 4;
n_timepoints = 1200;
T_subject{1} = repmat(n_timepoints,n_sessions,1)';
T = repmat(T_subject,size(f,1),1);
ICAdim = 50; % number of ICA components (25,50,100,etc)
covtype = 'full'; % type of covariance matrix
zeromean = 1;

DirOut = 'C:\Users\au699373\OneDrive - Aarhus Universitet\Dokumenter\MATLAB\HMMMAR_BG\HMMMAR Results\MAR_2022\All\Repetition_6\';
K_vec = [7 7 7 7 7 7 7 7];% 7 7 7 7];
hmm_train_dirichlet_diag_vec = [10 10 10 10 10 10 10 10];% 10 10 10 10];
lags = {-3:3, -3:3, -3:3, -3:3, -3:3, -3:3, -3:3, -3:3, -3:3};%, -3:3, -3:3, -3:3};

load vars_target.mat
vars=vars(grotKEEP,:);

repetitions = length(K_vec); % to run it multiple times (keeping all the results)
TR = 0.75; N = length(f);

%%
% Prediction Parameters
parameters_prediction.verbose = 0;
parparameters_prediction = struct();
paameters_prediction.method = 'KRR';
parameters_prediction.alpha = [0.1  0.5 1.0 5];%[0.1];%[0.1 0.5 1.0 5];
parameters_prediction.sigmafact = [1/2 1 2];%[1];%[1/2 1 2];





explained_variance = NaN(size(vars,2),repetitions);
vars_hat = NaN(N,size(vars,2),repetitions);
explained_variance_krr_predict = NaN(size(vars,2),repetitions);
vars_hat_krr_predict = NaN(N,size(vars,2),repetitions);
folds_all = cell(10,34);


%for r = 1:repetitions

disp(['Repetition ' num2str(r) ])
load([DirOut 'KLdistances_ICA' num2str(ICAdim) '_r' num2str(r)  '.mat'],'DistMat')
DistHMM = DistMat;
    D = DistHMM(:,:,r);
    for j = 34%1:size(vars,2)
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
        %[yhat,~,~,~,~,folds] = predictPhenotype(y_new,D_new,parameters_prediction,twins_new,conf_new);
        [yhat,~,~,~,~,folds] = predictPhenotype_no_suppress(y_new,D_new,parameters_prediction,twins_new,conf_new);
        folds_all(:,j) = folds;

        explained_variance(j,r) = corr(squeeze(yhat),y_new).^2;
        vars_hat_store = NaN(size(vars,1),1);
        vars_hat_store(non_nan_idx) = yhat;
        vars_hat(:,j,r) = vars_hat_store;



    end
%end
%end
%%

max(vars_hat(:,34,r))

[vars(1:20,34) vars_hat(1:20,34,r) ]
format short g
mse = (sum((vars(1:20,34) - vars_hat(1:20,34,r)).^2))/20












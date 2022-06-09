%% How does reshape work?
% "Reshaped array, returned as a vector, matrix, multidimensional array, or
% cell array. The data type and number of elements in B are the same as the data 
% type and number of elements in A. The elements in B preserve their columnwise ordering from A."
% e.g. say we have an array which is 1001x34x4x4x4
% and we flatten this to 1001x34x64
% which columns are preserved where?
clc
load vars_target
DirOut = 'C:\Users\au699373\OneDrive - Aarhus Universitet\Dokumenter\MATLAB\HMMMAR_BG\HMMMAR Results\MAR_2022\All\Repetition_3\';
load([DirOut 'HMM_predictions.mat'])
size(vars_hat)
vars_hat_reshape = reshape(vars_hat,1001,34,[]);
size(vars_hat_reshape)
%v = 2;
%vars_hat(1:5,v,4,3,4)
%vars_hat_reshape(1:5,v,60)
% Order of HMMs after reshape
% 1 = [1 1 1]
% 2 = [2 1 1]
% 3 = [3 1 1]
% 4 = [4 1 1]
% 5 = [1 2 1]
% 6 = [2 2 1]
% 7 = [3 2 1]
% 8 = [4 2 1]
% 9 = [1 3 1]
% 10 = [2 3 1]
% 11 = [3 3 1]
% 12 = [4 3 1]
% 13 = [1 4 1]
% 14 = [2 4 1]
% 15 = [3 4 1]
% 16 = [4 4 1]
% 17 = [1 1 2]
% 18 = [2 1 2]
% 19 = [3 1 2]
% 20 = [4 1 2]
% 21 = [1 2 2]
% 22 = [2 2 2]
% 23 = [3 2 2]
% 24 = [4 2 2]
% 25 = [1 3 2]
% 26 = [2 3 2]
% 27 = [3 3 2]
% 28 = [4 3 2]
% 29 = [1 4 2]
% 30 = [2 4 2]
% 31 = [3 4 2]
% 32 = [4 4 2]
%...
% 60 = [4 3 4]
% 61 = [1 4 4]
% 62 = [2 4 4]
% 63 = [3 4 4]
% 64 = [4 4 4]



%% So what are the reps we are looking at? And how bad are they
n_vars = 34;
n_reps = 64;
Index = (1:n_reps)';
K_vec = (repmat(repelem([3 7 11 15],1,4),1,4))';
DD_store = zeros(n_reps,1);
lags_store = cell(n_reps,1);
lags = zeros(n_reps,1);
for i = 1:n_reps
    i
    load(['HMMs_r' num2str(i) '_d' num2str(i) '_GROUP'],'hmm')
    DD_store(i) = hmm.train.DirichletDiag;
    lags_store{i} = min(hmm.train.embeddedlags):max(hmm.train.embeddedlags);
    lags(i) = max(hmm.train.embeddedlags);
end
%K_vec = [7 7 7 7 7 7 7 7 7 7 7 7]';
%DD_store = [10 10 10 10 10 10 10 10 10 10 10 10]';
%lags =  [3 3 3 3 3 3 3 3 3 3 3 3]';%{-3:3, -3:3, -3:3, -3:3, -3:3, -3:3, -3:3, -3:3, -3:3, -3:3, -3:3, -3:3};


sse_max = NaN(n_vars,n_reps); sse_idx = NaN(n_vars,n_reps);
for i = 1:n_reps
    for v = 1:n_vars
        subject_squared_error = (squeeze(vars_hat_reshape(:,v,:)) - vars(:,v)).^2;
        [sse_max(v,i), sse_idx(v,i)] = max(subject_squared_error(:,i));
    end
end
% Create new matrix of combined column
a1 = sse_max'; a2 = sse_idx';
a3 = [zeros(size(a1)) zeros(size(a2))];
a3(:,1:2:end) = a1; a3(:,2:2:end) = a2;

vars_select = 1:34;

% Create table
T2 = table(Index,K_vec,DD_store,lags,round(a3(:,1:2*numel(vars_select)),3,"significant"));%sse_max');

% Create columns titles for table
column_names = cell(1, 72); column_names(1:4) = {'Index','K_vec','DD_store','lags'}; i = 0;
for k = 5:2:numel(column_names); i = i + 1; column_names{k} = sprintf('var %i sbj mx err',k-(3+i)); column_names{k+1} = sprintf('idx %i sbj mx err',k-(3+i)); end

%% Output table to figure
uitable('Data',T2{:,:},'ColumnName',column_names,...
 'RowName',T2.Properties.RowNames,'Units', 'Normalized', 'Position',[0, 0, 1, 1]);

% Clearly, sometimes there is a subject which is a massive outlier which is
% getting a bad prediction. This is to be expected. See for example
% how subject 407 in vars 1 is 23 and the next smallest is 26.
[M,I] = mink(vars(:,1),3)
[M,I] = maxk(vars(:,1),3)
% Same for subject 555 in variable 34
%[M,I] = mink(vars(:,34),3)
% Have a look in the table to see the for the majority of the repetitions,
% subject 407 is the worst for vars 1 and subject 555 is the worst for vars
% 34.

% Solution for this would be to remove outliers as they throw the MSE quite
% a bit u

% However, what about for vars 1 when 407 is not the worst, and instead we
% are getting terrible predictions for several of the repetitions? Why is
% this happening?

% %% Let's check out the frequency spectra
% % for rep 1
% %load('HMMs_r1_d1_states_3.mat')
% a = HMMs_dualregr{1} % subject 1
% a.train.Fs
% a.train.
% 


%% Let's start by plotting the explained variance to show that I am getting
% similar stuff to before (e.g. c. 0.2 EV for intelligence features 11:14)
vars_hat_test = reshape(vars_hat,1001,34,[]);
exp_var_test = reshape(explained_variance,34,[]);
figure;
scatter(1:34,exp_var_test,'x','b'); xlabel('Intelligence Feature'); ylabel('Explained Variance');
title(sprintf('Explained Variance of %i HMM repetition predictions', size(vars_hat_test,3)))
legend('HMM repetitions')

%% Examine predictions for bad ones
%vars_hat_test = squeeze(vars_hat(:,:,1,:,:));
%vars_hat_test = reshape(vars_hat_test, 1001, 34, []);


v = 1;
max_no = 3;
reps_select =  1:size(vars_hat_test,3); %[5 6 7];
[maxk(vars(:,v),max_no) maxk(squeeze(vars_hat_test(:,v,reps_select)),max_no) ]
[mink(vars(:,v),max_no) mink(squeeze(vars_hat_test(:,v,reps_select)),max_no) ]
%[NaN explained_variance(v,reps_select) ]

[NaN max(vars(:,v)) mean(vars(:,v),'omitnan') min(vars(:,v)) ]
%% Now let's plot for each variable, the maximum value of the actual data, and then the maximum value of the predicted data
figure;
max_vars = max(vars);
max_vars_hat = max(vars_hat_reshape,[],[1 3]);
% We scale by the actual maximum so plots are comparable across variables
max_vars_norm = max_vars./max_vars;
max_vars_hat_norm = max_vars_hat./max_vars;
scatter(1:34,max_vars_norm)
hold on
scatter(1:34,max_vars_hat_norm)
title('Order of magnitude that the largest single subject prediction is larger than the actual largest subject data point') % Note this isn't necessarily the same subject
xlabel('Intelligence Feature');
ylabel('Order of magnitude')
%% Heatmap of intelligence features x HMM types of terrible predictions
max_vars_hat_heatmap = squeeze(max(vars_hat_reshape,[],1));
%max_vars_hat_heatmap_norm = max_vars_hat_heatmap./(max_vars)';
%max_vars_hat_heatmap_norm = max_vars_hat_heatmap./a';
a = repmat(max_vars',1,n_reps);
max_vars_hat_heatmap_norm = max_vars_hat_heatmap./a;


figure;
imagesc(max_vars_hat_heatmap_norm'); colorbar
title('Order of magnitude that the largest single subject prediction is larger than the actual largest subject data point')
xlabel('Intelligence Feature'); ylabel('HMM type')

figure;
imagesc(max_vars_hat_heatmap_norm); colorbar
caxis([0, 1.3]);
title('Order of magnitude that the largest single subject prediction is larger than the actual largest subject data point')
xlabel('Intelligence Feature'); ylabel('HMM type')

%% Calculate FOs
% When we add lags to the HMM, the Gammas produced are not of the same
% length as otherwise, so let's find out how long each subject Gamma is
K_vec = (repmat(repelem([3 7 11 15],1,4),1,4))';
N = 1001; ICAdim = 50;
Gamma_length = zeros(n_reps,1);
for r = 1:n_reps
    r
    load(['HMMs_r' num2str(r) '_d' num2str(r) '_GROUP.mat'],'Gamma')
    Gamma_length(r) = size(Gamma,1);
end
n_gamma_reduction = [(4800*N - Gamma_length)/N];% lags]
% so for lags = 1, gamma should have 8 time points less per subject (4 sessions and since lags are -1, 0, 1, there is 2 for each session)
%        lags = 3, gamma should have 24 time points less per subject
%        lags = 9, gamma should have 72 time points less per subject
%        lags = 15, gamma should have 120 time points less per subject

% %% Now let's calculate FOs
% for r = 1:5%:64
%     r
%     load(['HMMs_r' num2str(r) '_d' num2str(r) '_GROUP.mat'],'hmm','Gamma','vpath','FOgroup')
%     K = K_vec(r)
%     FOgroup = zeros(N,K); % Fractional occupancy
%     meanActivations = zeros(ICAdim,K); % maps
%     time_points = 4800 - n_gamma_reduction(r);
%     for j=1:N
%         sprintf('Group - sub %i', j)
%         ind = (1:time_points) + time_points*(j-1);
%         FOgroup(j,:) = mean(Gamma(ind,:));
% 
%     end
%     %save(['HMMs_r' num2str(r) '_d' num2str(r) '_GROUP.mat'],'hmm','Gamma','vpath','FOgroup')
% end

%% Are the repetitions with the highest explained variance the ones with the lowest mean squared error?
exp_var_test = reshape(explained_variance,34,[]);
mean_sq_test = reshape(mean_squared_error,34,[]);

ev_rank = NaN(34,n_reps);
ms_rank = NaN(34,n_reps);
figure;
for i = 1:34
    mev = max(exp_var_test(i,:));
    [ev_M,ev_rank(i,:)]=sort(exp_var_test(i,:),'Descend'); %[ev_I' ev_M'] % order by largest EV first
    [ms_M,ms_rank(i,:)]=sort(mean_sq_test(i,:),'Ascend'); %[flip(ms_I') flip(ms_M')] % order by largest MS first
    Y1 = ev_rank(i,:)'; Y2 = flip(ms_rank(i,:))';
    subplot(6,6,i)
    plot([1 2],[Y1 Y2],'x-');
    xlim([0 3]); xlabel('Explained Variance Mean Squared Error')
    ylabel('Ranking of HMM prediction')
    format short g
    title(sprintf('Max EV for intelligence feature = %0.3f',mev))
end

%% Are the 'best' repetitions the same across all intelligence features?
figure; imagesc(ev_rank); colorbar; xlabel('HMM repetition prediction');
ylabel('Intelligence Feature'); title('Explained Variance')
figure; imagesc(ms_rank); colorbar;  xlabel('HMM repetition prediction');
ylabel('Intelligence Feature'); title('Mean Squared Error')

%% Spearman's correlation
figure;
a = corr(ev_rank');
a(eye(size(a))==1) = nan;
subplot(1,2,1); imagesc(a); colorbar; xlabel('Intelligence Feature'); ylabel('Intelligence Feature'); title('Explained Variance')
b = corr(ms_rank');
b(eye(size(b))==1) = nan;
subplot(1,2,2); imagesc(b); colorbar; xlabel('Intelligence Feature'); ylabel('Intelligence Feature'); title('Mean Squared Error')

sgtitle('Spearman''s rank correlation across intelligence features')


%%
% Let's get the IDs of the worst predicted subjects
% we start with variable 1
v = 1;
subject_squared_error = (squeeze(vars_hat_reshape(:,v,:)) - vars(:,v)).^2;
% Let's look at one HMM
hmm = 8;
K = 3;
[M,I] = sort(subject_squared_error(:,hmm));
I

% Now let's look at the FOs for that HMM and compare the subjects accuracy
% with their FO
%load(['HMMs_r' num2str(hmm) '_d' num2str(hmm) '_states_' num2str(K) '.mat'],'FOdual')

%%
v = 34;
max_no = 3;
vars_hat_test = reshape(vars_hat,1001,34,[]);
exp_var_test = reshape(explained_variance,34,[]);
reps_select =  1:size(vars_hat_test,3);
[maxk(vars(:,v),max_no) maxk(squeeze(vars_hat_test(:,v,reps_select)),max_no) ]
[mink(vars(:,v),max_no) mink(squeeze(vars_hat_test(:,v,reps_select)),max_no) ]
[NaN explained_variance(v,reps_select) ]

[NaN max(vars(:,v)) mean(vars(:,v),'omitnan') min(vars(:,v)) ]





[NaN max(vars(:,v)) mean(vars(:,v),'omitnan') min(vars(:,v)) ]

%% Can the FOs tell us which subjects will be badly predicted?
N = 1001; n_vars = 34; n_reps = 12;
predictedY = NaN(N,n_vars,n_reps);
corr_preds = NaN(n_vars,n_reps);
subject_squared_error_preds = NaN(N,n_vars,n_reps);
mean_squared_error_preds = NaN(n_vars,n_reps);

load('HMM_predictions.mat')
for i = 1:n_reps
        K = K_vec(i);
        load(['HMMs_r' num2str(i) '_d' num2str(i) '_states_' num2str(K) '.mat'],'FOdual');
        X = [FOdual max(FOdual,[],2)];
        
        for j = 1:34
            disp(['Vars ' num2str(j) ])
            % remove subjects
            y = subject_squared_error(:,j,i);
            non_nan_idx = find(~isnan(y));
            which_nan = isnan(y);
            y_new = y;
            X_new = X;

            if any(which_nan)
                y_new = y(~which_nan,:,:);
                X_new = X(~which_nan,:);
                warning('NaN found on Yin, will remove...')
            end

            % set folds (load from saved files)
            folds = folds_all(:,j);
            N = length(y_new);

            % for each fold split into test and train
            for ifold = 1:length(folds)
                
                J = folds{ifold};
                ji = setdiff(1:N,J);

                % Note: nested CV not needed since no hyperparameters in standard regression

                % Do I need to centre variables? I think no because regression
                % (I do need to in ridge regression)
                % let's test by centering and seeing if any difference

                % train model on training data
                b = regress(y_new(ji),X_new(ji,:));

                % make predictions on test data
                predictedY(J,j,i) = X(J,:)*b;

            end

                % test predictions
                corr_preds(j,i) = corr(predictedY(:,j,i),subject_squared_error(:,j,i),'rows','complete');
                subject_squared_error_preds(:,j,i) = (predictedY(:,j,i) - subject_squared_error(:,j,i)).^2;
                mean_squared_error_preds(j,i) = sum((predictedY(:,j,i) - subject_squared_error(:,j,i)).^2,'omitnan')/N;
        end


end 
%%
%figure; imagesc(mean_squared_error_preds); colorbar;
figure; imagesc(mean_squared_error_preds); colorbar;
figure; imagesc(corr_preds'); colorbar;
xlabel('Intelligence Feature'); ylabel('HMM repetition');
title('Correlation between subject errors as predicted by FOs and the subject errors predicted using KRR')

%%
v = 1; rep_select = 1; [subject_squared_error(:,v,rep_select) squeeze(predictedY(:,v,rep_select))]

maxk([subject_squared_error(:,v,rep_select) squeeze(predictedY(:,v,rep_select))],10)
mink([subject_squared_error(:,v,rep_select) squeeze(predictedY(:,v,rep_select))],3)


% %%
% K_vec = repmat(repelem([3 7 11 15],1,4),1,4);
% FOdual_mean = cell(64,1);
% for i = 1:64
%     i
%     K = K_vec(i);
%     load(['HMMs_r' num2str(i) '_d' num2str(i) '_states_' num2str(K) '.mat'],'FOdual')
%     FOdual_mean{i} = mean(FOdual)
%     
% end
% 
%% Which alpha parameters are giving us the worst predictions?

load('HMM_predictions_vary_param.mat')
load('vars_target.mat')
n_vars = 34;
n_folds = 10;
n_reps = 64;
alph = [0.4 0.7 1 10 100];

mean_error = NaN(n_folds,n_vars,n_reps);
exclude_predictions = false(n_reps,n_vars);
mean_error_by_alph = NaN(size(alph));

figure;
for i = 1:n_vars

%     % Optional: keep only terrible predictions
%     vars_hat_i = squeeze(vars_hat(:,i,:));
%     max_vars = max(vars(:,i))*1.1; % note highest target feature value + add a buffer
%     min_vars = min(vars(:,i))/1.1; % note lowest target feature value + add a buffer
%     exclude_predictions(:,i) = (max(vars_hat_i)' > max_vars) | (min(vars_hat_i)' < min_vars); % note predictions with out-of-range predictions
%     vars_hat_i_inc = vars_hat_i(:,exclude_predictions(:,i)); % exclude all out of range
%     fprintf('Number of excluded predictions %i\n', nnz(exclude_predictions(:,i))) % display how many predictions we removed

    %n_good_reps = size(vars_hat_i_inc,2);
    %subject_squared_errorKEEP = subject_squared_error(:,:,exclude_predictions(:,i));
    n_good_reps = 64;
    subject_squared_errorKEEP = subject_squared_error;
    
    for rep = 1:n_good_reps
        for ifold = 1:n_folds
            fold = folds_all{ifold,i};
            %alph = alpha_all_2(ifold,i,rep);
            mean_error(ifold,i,rep) = mean(subject_squared_errorKEEP(fold,i,rep),'omitnan');
        end
    end
    %alpha_all_vars = squeeze(alpha_all_2(:,i,exclude_predictions(:,i)));
    alpha_all_vars = squeeze(alpha_all_2(:,i,:));
    mean_error_vars = squeeze(mean_error(:,i,1:n_good_reps));

    for a = 1:length(alph)
        alph_search = alph(a);
        
        mean_error_by_alph(a) = sum(sum(mean_error_vars .* (alpha_all_vars == alph_search)));

    end
    subplot(6,6,i)
    bar(1:5, mean_error_by_alph)
    xticklabels({alph}); xlabel('Alpha Parameter');
    ylabel('Mean squared error')
    title(sprintf('Intelligence Feature %i',i))
    sgtitle('Which alpha parameters are causing the bad predictions (non-standardized errors)?')
end



%% Distribution of KL-divergences

%load('KLdistances_ICA50.mat')
a = DistMat_flat(:,:,1);
b = triu(a); % remove (j,i) entries as they are equal to (i,j) entries
c = b(:); % transform matrix to vector
d = c;
d(d==0) = []; % remove 0 entrues ((j,i) entries and main diagonal)


figure;
subplot(2,2,1); histogram(d);
xlabel('KL-divergence'); ylabel('Frequency');
title('Original KL divergences')
subplot(2,2,2); boxplot(d,'Orientation','horizontal')
xlabel('KL-divergence');
title('Original KL divergences')

% Now we load the Gaussinized
load('KLdistances_ICA50_gaussianized.mat')
a = KL_div_gausss(:,:,1);
b = triu(a); % remove (j,i) entries as they are equal to (i,j) entries
c = b(:); % transform matrix to vector
d = c;
d(d==0) = []; % remove 0 entrues ((j,i) entries and main diagonal)

subplot(2,2,3); histogram(d);
xlabel('KL-divergence'); ylabel('Frequency');
title('Gaussianized KL divergences')
subplot(2,2,4); boxplot(d,'Orientation','horizontal')
xlabel('KL-divergence');
title('Gaussianized KL divergences')
sgtitle('Distribution of KL-divergences between 1001 HCP subjects')




%% let's add noise to certain subjects in distance matrices and see if predictions are bad
% %load('KLdistances_ICA50.mat','DistMat_flat')
% clc
% 
% DistMat_flat_NOISE = NaN(1001,1001,64);
% for i = 1:64
%     % let's let 1 subject have really small distances to the rest
%     DMi = DistMat_flat(:,:,i);
%     [DMi(2:end,1) DMi(2:end,1)-25000];% + mean(DM1) 
%     DMi(2:end,1) = DMi(2:end,1)-(min(DMi(2:end,1)))+100;
%     DMi(1,2:end) = DMi(2:end,1);
%     DistMat_flat_NOISE(:,:,i) = DMi;
%     
%     % and 1 subject have relatively large distances to the rest
%     DMi2 = DistMat_flat_NOISE(:,:,i);
%     [DMi2(3:end,2) DMi2(3:end,2)+50000];% + mean(DM1) 
%     DMi2(3:end,2) = DMi2(3:end,2)+100000;
%     DMi2(2,3:end) = DMi2(3:end,2);
%     DistMat_flat_NOISE(:,:,i) = DMi2;
% end
% 
% 
% DistMat_flat_NOISE(1:10,1:10,3)
% 
% save('KLdistances_ICA50_NOISE.mat','DistMat_flat_NOISE')










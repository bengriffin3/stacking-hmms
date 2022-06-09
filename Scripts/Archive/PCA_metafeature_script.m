%%
clear; clc;
% Question: do we use FOgroup or FOdual?
% I think they used FOgroup in Diego's paper, but they are similar anyway
% so we could just test which one is better?
% ans: they are both basically the same anyway, group is used here
load('vars_target.mat')
DirOut = 'C:\Users\au699373\OneDrive - Aarhus Universitet\Dokumenter\MATLAB\HMMMAR_BG\HMMMAR Results\04_02_2022\Varying_states\';
K_vec = [3 3 3 6 6 6 9 9 9 12 12 12 15 15 15];
N = 1001;
rep = 15;
FO_metastate1 = NaN(N,rep);
FO_metastate2 = NaN(N,rep);
metastate_profile = NaN(N,rep);
metastate_PC1 = NaN(N,rep);
metastate_PC2 = NaN(N,rep);
FOgroup_all_reps = NaN(N,24,rep);
GammaSubMean = NaN(N,24,rep);

for i = 1:rep
    K = K_vec(i)
    % load fractional occupancy data
    load ([DirOut 'HMMs_r' num2str(i) '_GROUP.mat'],'Gamma','hmm','FOgroup')
    FOgroup_all_reps(:,1:K,i) = FOgroup; % checked - same as PNAS code

    % Form FO correlation matrix
    R = FOgroup;
    C = corr(R);

    %%%%%%%%%%%%%%%%%%% PCA %%%%%%%%%%%%%%%%%%%%%%%%%
    
    [coeff,score,latent,~,explained] = pca(R','NumComponents',2); % we only take the first PC
    [~,ord] = sort(score(:,1));


    coeff; % these are the principle component vectors (the evectors of the covariance matrix)
    latent; % these are the corresponding evalues
    explained; % see how much variance is explained by each PC

    score; % these are the representations of R in the principal component space i.e. the coefficient of the linear combination of original variables that make up the new component
    % look at these scores - whether they are positive or negative (i.e. whether the transformed data is one side side of the principal axis or another) corresponds to which metatstate they are in

    % Display FO matrix (states reordered)
    figure(); imagesc(C(ord,ord)); % figure 2 (b)
    title('FO Correlation Matrix'); xlabel('States');
    ylabel(colorbar, 'Correlation')
    

    % Display transition probability matrix (states reordered)
    P = hmm.P;
    P(eye(size(P))==1) = nan;
    %figure(); imagesc(P(ord,ord)); colorbar; % figure 2(a)

    metastate_PC1(:,i) = coeff(:,1);
    metastate_PC2(:,i) = coeff(:,2); 

    %%%%%%%%%%%%%%%%%%% PCA %%%%%%%%%%%%%%%%%%%%%%%%%

    % Detect heirarchical cluster (two clusters)
    Z = linkage(C,'ward');
    clusters_two = cluster(Z,'Maxclust',2);
    %figure(); [H,T,outperm] = dendrogram(Z); % Display heirarchical structure and FO matrix (states reordered to group up metastates i.e. in order of dendogram)

    % Divide states into clusters (is this the correct way to find clusters?
    states = (1:K_vec(i))';
    metastate1 = states(clusters_two==1);
    metastate2 = states(clusters_two==2);

    % Find FO of metastates
    FO_metastate1(:,i) = sum(R(:,metastate1),2);
    FO_metastate2(:,i) = sum(R(:,metastate2),2);
    %corr(FO_metastate1(:,i),vars, 'rows','complete')'
    %corr(FO_metastate2(:,i),vars, 'rows','complete')'
    
    %metastate profile is defined as the FO of the cognitive metastate minus the FO of the sensorimotor metastate
    metastate_profile(:,i) = FO_metastate1(:,i) - FO_metastate2(:,i);
    %corr(metastate_profile(:,i),vars, 'rows','complete')'


end

%% Sort out signs of metatstate

corr_F01 = corr(FO_metastate1);

FO_metastate1(:,corr_F01(:,1)<0) = 0;
FO_metastate2(:,corr_F01(:,1)>0) = 0;

FO_metastate_new_1 = FO_metastate1 + FO_metastate2;
FO_metastate_new_2 = 1-FO_metastate_new_1;

% check signs have been sorted (these correlations should be close to 1)
corr(FO_metastate_new_1)
corr(FO_metastate_new_2)

% 
% size(FO_metastate1)
% 
% a = corr(FO_metastate1)
% b = a(:,1)
% c = b>0
% d = b<0
% %d = b.*-c
% 
% %FO_metastate1_new = FO_metastate1(c,c)







%%

% Sort the +- signs out for the metastates
FO_metastate1_new = [ FO_metastate1(:,1:2) FO_metastate2(:,[3 4 5]), FO_metastate1(:,6:end)];
FO_metastate2_new = [ -FO_metastate2(:,1) FO_metastate2(:,[2 3 4]), -FO_metastate2(:,5)];
corr(FO_metastate1_new); % Check signs have been sorted correctly (all elements of this matrix should be close to 1)


% Check to see if we have positive correlation between metastates and
% variables / prediction accuracy
figure(); imagesc(corr(FO_metastate1_new,vars, 'rows','complete')'); colorbar % some variables are really good, others aren't
%corr(FO_metastate1_new,vars, 'rows','complete')'

% (note that this doesn't change the correlation between this and vars so I
% think unneccesary)
metastate_profile = FO_metastate1_new - FO_metastate2_new;

%save([DirOut 'HMMs_meta_data_GROUP_metastate.mat'],'FO_metastate1_new','FO_metastate2_new','metastate_profile')

% % load HMM predictions
% load([ DirOut  'HMM_predictions.mat']) % load predictions
% 
% % determine accuracy of predictions (we want our metafeature to be correlated with this)
% squared_error = (vars - vars_hat).^2;
% corr(FO_metastate1_new,squeeze(squared_error(:,1,:)), 'rows','complete')'


%% Notes
% Need to take care of signs, as in Diego's slack message
% I think we essentially mean what are we calling the 'first' and 'second'
% metastate? In the paper they assign a positive and negative value so I
% think we need to ensure consistency across repetitions



% score; % this is the data in the PC space (equal to C*coeff) i.e. we have recasted the data along the principal component axes. Note that the columns of this are orthogonal to each other
% corrcoef(score); % this shows us that the columns of score are orthogonal since they have 0 correlation

% By ranking your eigenvectors in order of their eigenvalues, highest to lowest, you get the principal components in order of significance. 
% So let's use eig to check ive done PCA correctly

%C_centre = C - mean(C); %centre FO correlation matrix % pca() in MATLAB centres data for us


% covarianceMatrix = cov(C);
% [V,D] = eig(covarianceMatrix);
% V
%figure(); imagesc(C); colorbar;
%c = cluster(Z,'Maxclust',3); % find a maximum of two clusters in the data
    %metastate2 = states(clusters_two==2);
    %FO_metastate2 = sum(R(:,metastate2),2);

%     for j=1:N
%         ind = (1:4800) + 4800*(j-1);
%         FOgroup_new(j,1:K,i) = mean(Gamma(ind,:));
%     end


%     coeff; % these are the principle component vectors (the evectors of the covariance matrix)
%     score; % these are the representations of R in the principal component space i.e. the coefficient of the linear combination of original variables that make up the new component
%     latent; % these are the corresponding evalues
%     explained; % see how much variance is explained by each PC

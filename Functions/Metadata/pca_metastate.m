function [metastate_PC1,metastate_PC2,FO_metastate1,FO_metastate2] = pca_metastate(hmm,FOgroup)
% Question: do we use FOgroup or FOdual?
% I think they used FOgroup in Diego's paper, but they are similar anyway
% so we could just test which one is better?
% ans: they are both basically the same anyway, group is used here

% N = size(vars,1);
% FO_metastate1 = NaN(N,rep);
% FO_metastate2 = NaN(N,rep);
% metastate_profile = NaN(N,rep);
% metastate_PC1 = NaN(N,rep);
% metastate_PC2 = NaN(N,rep);
% FOgroup_all_reps = NaN(N,24,rep);

K = hmm.K;

% % load fractional occupancy data
% FOgroup_all_reps(:,1:K,i) = FOgroup;

% Form FO correlation matrix
R = FOgroup;
C = corr(R);

%%%%%%%%%%%%%%%%%%% PCA %%%%%%%%%%%%%%%%%%%%%%%%%

[coeff,score,latent,~,explained] = pca(R','NumComponents',2); % we only take the first PC
[~,ord] = sort(score(:,1));


% coeff; % these are the principle component vectors (the evectors of the covariance matrix)
% score; % these are the representations of R in the principal component space i.e. the coefficient of the linear combination of original variables that make up the new component
% latent; % these are the corresponding evalues
% explained; % see how much variance is explained by each PC

% Display FO matrix (states reordered)
%figure(); imagesc(C(ord,ord)); % figure 2 (b)
title('FO Correlation Matrix'); xlabel('States');
ylabel(colorbar, 'Correlation')


% Display transition probability matrix (states reordered)
P = hmm.P;
P(eye(size(P))==1) = nan;
%figure(); imagesc(P(ord,ord)); colorbar; % figure 2(a)

metastate_PC1 = coeff(:,1);
metastate_PC2 = coeff(:,2);

%%%%%%%%%%%%%%%%%%% PCA %%%%%%%%%%%%%%%%%%%%%%%%%

% Detect heirarchical cluster (two clusters)
Z = linkage(C,'ward');
clusters_two = cluster(Z,'Maxclust',2);
% add a line that says if 'plots' selected then show dendogram?
%figure(); [H,T,outperm] = dendrogram(Z); % Display heirarchical structure and FO matrix (states reordered to group up metastates i.e. in order of dendogram)

% Divide states into clusters (is this the correct way to find clusters?
states = (1:K)';
metastate1 = states(clusters_two==1);
metastate2 = states(clusters_two==2);

% Find FO of metastates
FO_metastate1 = sum(R(:,metastate1),2);
FO_metastate2 = sum(R(:,metastate2),2);
%corr(FO_metastate1(:,i),vars, 'rows','complete')'
%corr(FO_metastate2(:,i),vars, 'rows','complete')'

% %metastate profile is defined as the FO of the cognitive metastate minus the FO of the sensorimotor metastate
% metastate_profile = FO_metastate1 - FO_metastate2;
% %corr(metastate_profile(:,i),vars, 'rows','complete')'



% 
% % Sort out signs of metatstate
% 
% corr_F01 = corr(FO_metastate1);
% 
% FO_metastate1(:,corr_F01(:,1)<0) = 0;
% FO_metastate2(:,corr_F01(:,1)>0) = 0;
% 
% FO_metastate_new_1 = FO_metastate1 + FO_metastate2;
% FO_metastate_new_2 = 1-FO_metastate_new_1;
% 
% % check signs have been sorted (these correlations should be close to 1)
% corr(FO_metastate_new_1)
% corr(FO_metastate_new_2)
% 
% % what about states that aren't part of either metastate (e.g. state 5 in
% % Diego's paper?
% 
% % 
% % size(FO_metastate1)
% % 
% % a = corr(FO_metastate1)
% % b = a(:,1)
% % c = b>0
% % d = b<0
% % %d = b.*-c
% % 
% % %FO_metastate1_new = FO_metastate1(c,c)
% 
% 
% 
% 
% 
% 
% 
% %
% 
% % Sort the +- signs out for the metastates
% FO_metastate1_new = [ FO_metastate1(:,1:2) FO_metastate2(:,[3 4 5]), FO_metastate1(:,6:end)];
% FO_metastate2_new = [ -FO_metastate2(:,1) FO_metastate2(:,[2 3 4]), -FO_metastate2(:,5)];
% corr(FO_metastate1_new); % Check signs have been sorted correctly (all elements of this matrix should be close to 1)
% 
% 
% % Check to see if we have positive correlation between metastates and
% % variables / prediction accuracy
% figure(); imagesc(corr(FO_metastate1_new,vars, 'rows','complete')'); colorbar % some variables are really good, others aren't
% %corr(FO_metastate1_new,vars, 'rows','complete')'
% 
% % (note that this doesn't change the correlation between this and vars so I
% % think unneccesary)
% metastate_profile = FO_metastate1_new - FO_metastate2_new;
% 
% %save([DirOut 'HMMs_meta_data_GROUP_metastate.mat'],'FO_metastate1_new','FO_metastate2_new','metastate_profile')
% 
% % % load HMM predictions
% % load([ DirOut  'HMM_predictions.mat']) % load predictions
% % 
% % % determine accuracy of predictions (we want our metafeature to be correlated with this)
% % squared_error = (vars - vars_hat).^2;
% % corr(FO_metastate1_new,squeeze(squared_error(:,1,:)), 'rows','complete')'
end
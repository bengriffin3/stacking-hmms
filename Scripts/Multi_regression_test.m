%% Multivariate regression
X = metafeature;
Y = vars(:,2);
Y = (Y-min(Y))/(max(Y) - min(Y));
% [beta,Sigma,E,CovB,logL] = mvregress(X,Y)
mdl = fitlm(X,Y);
% figure
% plot(mdl)
anova(mdl,'summary')

anova(mdl)
%% Condition number?
sing_vals_X = sqrt(eig(X'*X));
cond_num = max(sing_vals_X)/min(sing_vals_X);
cond_num
% condition number > 30 suggests there is multicolinearity

% vif(X)? Variance inflaction factor can tell us about multicolinearity too

%%
% Next goal:
% So I have generated a metafeature for each repetition that has a
% specified correlation with (for example) the prediction accuracy of the
% repetitions (e.g. 0.2). However, these simulated metafeatures have a
% correlation of 0 with each other, whereas the genuine metafeatures have
% acorrelation > 0.9. Therefore, let's try to simulate the metafeatures so
% that they have a correlation of 0.5 with each other, or something


% First, let's look at our simulated metafeatures and real metafeatures and
% see the correlation between them (note that this is done for a random
% simulated metafeature but it is the same for all of them
% Correlation between metafeatures of different repetitions
figure(1);
subplot(1,2,1); imagesc(corr(metafeature,'rows','complete')); colorbar; xlabel('Metafeature of HMM repetition'); ylabel('Metafeature of HMM repetition'); title('Real metafeature')
subplot(1,2,2); imagesc(corr(metafeature_simu(:,:,22),'rows','complete')); colorbar; xlabel('Metafeature of HMM repetition'); ylabel('Metafeature of HMM repetition'); title('Simulated metafeature')
sgtitle(sprintf('Matrices of correlations between metafeatures (real and simulated) of distinct HMM repetitions'))

%% Plot heatmaps of correlations between errors of predictions and metafeatures
pearson_corr_error_metafeature = NaN(n_var,n_repetitions);
pearson_corr_error_metafeature_simulated = NaN(n_var,n_repetitions);

for v = 1%:n_var
    [~,~,Metafeature,non_nan_idx] = nan_subject_remove(v,vars_norm,vars_hat_norm,metafeature_norm); % remove NaNs (this needs to be done for every var because different subjects are missing for each var)
    for j = 1:n_repetitions;[pearson_corr_error_metafeature(v,j),~] = corr(squared_error(non_nan_idx,j,v),Metafeature(:,j),'type','Pearson'); end
    for j = 1:n_repetitions; [pearson_corr_error_metafeature_simulated(v,j),~] = corr(squared_error(non_nan_idx,j,v),metafeature_simu(non_nan_idx,j,v),'type','Pearson'); end
end

figure(2)
sgtitle(sprintf('%s Correlation of prediction error vs metafeature',corr_measures{1}))
subplot(1,2,1); imagesc(pearson_corr_error_metafeature); colorbar;
xlabel('HMM repetition (increasing K->)'); ylabel('Variable'); title(sprintf('True metafeature'))
subplot(1,2,2); imagesc(pearson_corr_error_metafeature_simulated); colorbar;
xlabel('HMM repetition (increasing K->)'); ylabel('Variable'); title(sprintf('Simulated metafeature'))


% %% Set diagonal elements to 0 to see if any pattern
% figure;
% for i = 1%:34
%     corr_meta = corr(metafeature_simu(:,:,i),'rows','complete');
%     M = corr_meta - diag(diag(corr_meta));
% subplot(6,6,i); imagesc(M); colorbar; xlabel('Metafeature of HMM repetition'); ylabel('Metafeature of HMM repetition'); title('Simulated metafeature')
% end

%%
% dummy example for creating a random vector with pre-defined correlations
% between its columns
x=[  1  0.5 0.3; 0.5  1  0.2; 0.3 0.2  1 ;]; %Correlation matrix
U=chol(x); %Cholesky decomposition 
R=randn(1001,length(x)); %Random data in three columns each for X,Y and Z
Rc=R*U; %Correlated matrix Rc=[X Y Z]
corr(Rc)

%%
% Goal: re-simulate metafeatures but with a correlation to each other (e.g.
% of 0.5)

% I need to basically create a metafeature for each repetition of the HMM,
% but they have to be
% (i) correlated with the prediction accuracies of the corresponding
% repetition (i.e. the true metafeature correlation with the prediction
% accuracies)
% (ii) correlated to each other


% we start by doing it for 1 variable
% Currently, I just do (i), i.e create a metafeature which is correlated
% with one other vector (the accuracy of the corresponding HMM repetition)

size(squared_error(:,1:15,1)) % (ii) prediction accuracies

corrcoef(squared_error(:,1:15,1),metafeature)


% I now want to make sure the vector is correlation with that vector, but
% also all the other simulated metafeatures
% So what if I had a 30x30 correlation matrix, which is (n_reptitions*2) x (n_reptitions*2)
% n_repetitions*2 = n_repetitions representing the accuracy of the HMM
% repetitions and n_reptitions representing the metafeatures
% we can create this matrix for the true metafeatures easily
% we then alter this matrix so that the correlations between the
% metafeatures is different, and then re-engineer the metafeatures from
% that correlation matrix

figure(3)
imagesc(corr([squared_error(:,1:15,1)])); colorbar


%%

% first let's create it for the actual metafeatures
size([squared_error(:,1:15,1) metafeature]);
corr_errors_metafeatures = corr([squared_error(:,1:15,1) metafeature]);
figure(4); imagesc(corr_errors_metafeatures); colorbar

% and now our current simulated metafeature
corr_errors_metafeatures = corr([squared_error(:,1:15,1) squeeze(metafeature_simu(:,:,1))]);
figure(5); imagesc(corr_errors_metafeatures); colorbar

%%
%pearson_corr_error_metafeature =
%
%    0.2380    0.2223    0.1836    0.0799    0.2392    0.1455   -0.0240    0.1020    0.0369    0.0943    0.2807    0.1705    0.2312   -0.0450    0.2127


X = squared_error(:,1:15,1);
Y = Metafeature;
diag(corr(X,Y)); % pairwise correlation values we want to replicate

% so we simulate a metafeature with the same correlation
Z_1 = metafeature_simu(:,:,1);
diag(corr(X,Z_1))
figure;
% BUT correlation between real metafeature is very high
subplot(1,2,1); imagesc(corr(Y));

% and correlation between real metafeature is very low
subplot(1,2,2); imagesc(corr(Z_1));

%% For variable 1? Actually just a simulated variable essentially
simulation_cov_matrix = NaN(46);

% first prescribe correlations between predictions (to be realistic? not overly important)
predictions = squeeze(vars_hat(:,1,:));
%figure; imagesc(corr(predictions));
simulation_cov_matrix(1:15,1:15) = corr(predictions);

% second prescribe correlations between metafeatures
simulation_cov_matrix(16:30,16:30) = corr(Metafeature);

% Third prescribe correlations between simulated metafeatures - THIS IS KEY - WHAT WE WANT TO CHANGE
simulation_cov_matrix(31:45,31:45) = corr(metafeature_simu(:,:,1));


A = predictions;
A = A - mean(A);
try chol(A)
    disp('Matrix is symmetric positive definite.')
catch ME
    disp('Matrix is not symmetric positive definite')
end





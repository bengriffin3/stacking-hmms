clc
Entropy = Metafeature_all;
entropy_simu_norm = Metafeature_simu(:,:,1);
n_subjects = size(Entropy,1);
% What's the difference between our simulated metafeature and the real one?
% same means
mean(entropy_simu_norm);
mean(Entropy);
% similar standard deviations
std(entropy_simu_norm);
std(Entropy);
% let's plot them
figure(); sgtitle('Metafeature (Simulated)')
subplot(3,1,1); scatter(1:n_subjects,entropy_simu_norm(:,1));
subplot(3,1,2); scatter(1:n_subjects,sort(entropy_simu_norm(:,1)))
subplot(3,1,3); scatter(entropy_simu_norm(:,1),squared_error(:,1))
figure(); sgtitle('Entropy')
subplot(3,1,1); scatter(1:n_subjects,Entropy(:,1))
subplot(3,1,2); scatter(1:n_subjects,sort(Entropy(:,1)))
subplot(3,1,3); scatter(Entropy(:,1),squared_error(:,1))
figure(); sgtitle('Entropy Gaussianized')
subplot(3,1,1); scatter(1:n_subjects,Entropy_gauss(:,1))
subplot(3,1,2); scatter(1:n_subjects,sort(Entropy_gauss(:,1)))
subplot(3,1,3); scatter(Entropy_gauss(:,1),squared_error(:,1))
figure(); sgtitle('Likelihood')
subplot(3,1,1); scatter(1:n_subjects,Likelihood(:,1))
subplot(3,1,2); scatter(1:n_subjects,sort(Likelihood(:,1)))
subplot(3,1,3); scatter(Likelihood(:,1),squared_error(:,1))
figure(); sgtitle('Likelihood Gaussianized')
subplot(3,1,1); scatter(1:n_subjects,Likelihood_gauss(:,1))
subplot(3,1,2); scatter(1:n_subjects,sort(Likelihood_gauss(:,1)))
subplot(3,1,3); scatter(Likelihood_gauss(:,1),squared_error(:,1))


% check correlations
corr(squared_error(:,1),entropy_simu_norm(:,1))
corr(squared_error(:,1),Entropy(:,1))
for i = 1:5
    corr(sort(squared_error(:,i)),Entropy_gauss(:,i))
    corr(sort(squared_error(:,i)),Likelihood_gauss(:,i))
end

% clearly our metafeature is skewed so let's Gaussianize our data
%Entropy_gauss
%%
%Entropy_gaussianized(:,[2 6 14 25 44])
Likelihood_gaussianized(:,[2 6 14 25 44])


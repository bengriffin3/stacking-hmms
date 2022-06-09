%% Plotting results

% Plot distance matrices

load('KLdistances_ICA50')
figure(1)
subplot(3,2,1);
imagesc(DistStatic);
title('Static distances between subjects');
xlabel('Subjects'); ylabel('Subjects');

for i = 2:6
    subplot(3,2,i);
    imagesc(DistMat(:,:,i-1));
    title('HMM distances between subjects - Repitition',i-1);
    xlabel('Subjects'); ylabel('Subjects');
end

%% Load and store predictions
load('staticFC_predictions')
explained_variance_static = explained_variance; % r²
vars_hat_static = vars_hat;
load('HMM_predictions')
explained_variance_hmm = explained_variance; % r²
vars_hat_hmm = vars_hat;

%plot predictions
cat_reorder = 1:679';
Y = -explained_variance_static;
Z = explained_variance_hmm(:,1);
figure(2)
scatter(cat_reorder,Y,10,repmat([115,213,246]/255,679,1),'filled')
grid on; hold on;
scatter(cat_reorder,Z,10,repmat([220,94,146]/255,679,1),'filled')
title('Uncorrected');
legend('Static','HMM');
xlabel('Variable'); ylabel('r^2')

%% plot predictions just for intelligence
load('headers_grouped_category.mat');
x = cell2mat(headers_grouped_category(:,4));
x(x~=2) = 0;
x(x == 2) = 1;

Y_intelligence = x.*Y;
Y_intelligence(Y_intelligence == 0) = [];
Z_intelligence = x.*Z;
Z_intelligence(Z_intelligence == 0) = [];

X = [1:1:size(Y_intelligence,1)];

figure(8)
%scatter(cat_reorder,Y_intelligence,10,repmat([115,213,246]/255,679,1),'filled')
scatter(X,Y_intelligence,10,repmat([115,213,246]/255,size(Y_intelligence,1),1),'filled')
grid on; hold on;
%scatter(cat_reorder,Z_intelligence,10,repmat([220,94,146]/255,679,1),'filled')
scatter(X,Z_intelligence,10,repmat([220,94,146]/255,size(Y_intelligence,1),1),'filled')
title('Uncorrected (Intelligence)');
legend('Static','HMM');
xlabel('Variable'); ylabel('r^2')

%%
figure(9)
x = 1:size(explained_variance,1)
y = explained_variance(:,1)
scatter(x,y,10,repmat([115,213,246]/255,679,1),'filled')
grid on; hold on;
title('Uncorrected (Intelligence)');
legend('Static','HMM');
xlabel('Variable'); ylabel('r^2')

%%
figure(3)
Diff = explained_variance_hmm(:,1)-explained_variance_static
Diff_colour(Diff<0) = -1; ;Diff_colour(Diff>0) = 1
Diff_cell = num2cell(Diff_colour)'
Diff_cell(Diff_colour == 1) = {[220,94,146]/255};
Diff_cell(Diff_colour == -1) = {[115,213,246]/255}
Diff_cell(Diff_colour == 0) = {[0 0 0 ]}
Diff_mat = cell2mat(Diff_cell);
scatter(cat_reorder,Diff,10,Diff_mat,'filled')
grid on
title('Difference HMM - average FC')
xlabel('Variable'); ylabel('r^2')

%
for p = 1:5
Diff = explained_variance_hmm(:,p)-explained_variance_static;
Diff(isnan(Diff)) = 0;
average_diff_vec = zeros(6,1);
% Demographics = 1
% Intelligence = 2
% Affective = 3
% Personality = 4
% Sleep = 5
% Anatomy = 6
% Other = 7

for i = 1:7
    average_diff_vec(i) = sum((category_index_vec == i).*Diff)/...
    nnz((category_index_vec == i).*Diff);
end

figure(4)
subplot(3,2,p)
cat = categorical({'Demographics','Intelligence','Affective','Personality','Sleep','Anatomy','Other'});
cat_reorder = reordercats(cat,{'Demographics','Intelligence','Affective','Personality','Sleep','Anatomy','Other'});
average_diff = -average_diff_vec;
b = bar(cat_reorder,average_diff);
b.FaceColor = 'flat';
b.CData(1,:) = [239,48,37]/255; b.CData(2,:) = [240,136,50]/255; b.CData(3,:) = [228,230,90]/255;
b.CData(4,:) = [113,226,86]/255; b.CData(5,:) = [151,33,245]/255; b.CData(6,:) = [.2 .6 .5]/255;
xlabel('Category'); ylabel('Average Difference')
title('Repitition',p);
end

% Reorder index

cat_reorder = 1:679';
Y = -explained_variance_static;
Z = explained_variance_hmm(:,1);
figure(5)
title('Trying to reorder so demographics first etc.')
subplot(2,1,1)
scatter(cat_reorder,Y,10,repmat([115,213,246]/255,679,1),'filled')
grid on
hold on
scatter(cat_reorder,Z,10,repmat([220,94,146]/255,679,1),'filled')
legend('Static','HMM')
xlabel('Variable'); ylabel('r^2')
subplot(2,1,2)
Y_2 = Y(idx)
Z_2 = Z(idx)
scatter(cat_reorder,Y_2,10,repmat([115,213,246]/255,679,1),'filled')
grid on
hold on
scatter(cat_reorder,Z_2,10,repmat([220,94,146]/255,679,1),'filled')
legend('Static','HMM')
xlabel('Variable'); ylabel('r^2')
    

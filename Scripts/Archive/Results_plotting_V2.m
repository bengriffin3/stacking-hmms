%% Plotting results
% Plot distance matrices
iterations = 3;

load('KLdistances_ICA50_30_09_2021_pm_run')
figure(1)
subplot(2,2,1);
imagesc(DistStatic);
title('Static distances between subjects');
xlabel('Subjects'); ylabel('Subjects');

for i = 2:iterations+1
    subplot(2,2,i);
    imagesc(DistMat(:,:,i-1));
    title('HMM distances between subjects - Repitition',i-1);
    xlabel('Subjects'); ylabel('Subjects');
end

%% Load and store predictions

load('staticFC_predictions_30_09_2021_pm_run')
vars_hat_static = vars_hat;
explained_variance_static = explained_variance; % r²
load('HMM_predictions_30_09_2021_pm_run')
vars_hat_hmm = vars_hat;
explained_variance_hmm = explained_variance; % r²

% Reorder vectors in order to group the categories of behavioural variables
% together
cat_reorder = idx;

vars_hat_hmm_reorg = vars_hat_hmm(:,cat_reorder,:);
vars_hat_static_reorg = vars_hat_static(:,cat_reorder);
explained_variance_hmm_reorg = explained_variance_hmm(cat_reorder,:);
explained_variance_static_reorg = explained_variance_static(cat_reorder);

% Now that we have reordered the behvavioural variables, we keep only the first
% 219 in order to remove the 'Anatomy' and 'Other' ones
vars_hat_hmm_reorg_trim = vars_hat_hmm_reorg(:,1:219,:);
vars_hat_static_reorg_trim = vars_hat_static(:,1:219);
explained_variance_hmm_reorg_trim = explained_variance_hmm_reorg(1:219,:);
explained_variance_static_reorg_trim = explained_variance_static_reorg(1:219);

n_variables = 219;



%%
%plot predictions
X = (1:n_variables)';
Y = -explained_variance_static_reorg_trim;
figure(2)
for i = 1:iterations
    subplot(3,1,i)
    Z = explained_variance_hmm_reorg_trim(:,i);
    scatter(X,Y,10,repmat([115,213,246]/255,n_variables,1),'filled')
    grid off; hold on;
    scatter(X,Z,10,repmat([220,94,146]/255,n_variables,1),'filled')
    title('Uncorrected');
    xlabel('Variable'); ylabel('r^2')
    ylim([-0.2 0.2])
    
    
    % Divide the data into categories, remembering:
    % Demographics = 1; Intelligence = 2; Affective = 3; Personality = 4
    % Sleep = 5; Anatomy = 6; Other = 7;
    [GC,GR] = groupcounts(x);
    GC_cumulative = cumsum(GC);
    % We have:
    % 13 demographic variables; 99 intelligence variables; 17 affectivex variables
    % 65 personality variables; 25 sleep variables; 199 anatomy variables
    % 261 'other' variables
    % Therefore, we want category lines after 13, 112, 129, 194, 219, 418, 679
    
    % plot category divider lines
    hold on;
    y1 = [-0.2 0.2]; % y-range for the partition lines 
    x1 = [GC_cumulative(1) GC_cumulative(1)];         % 1st dashed line
    x2 = [GC_cumulative(2) GC_cumulative(2)];         % 2nd dashed line
    x3 = [GC_cumulative(3) GC_cumulative(3)];         % 3rd dashed line
    x4 = [GC_cumulative(4) GC_cumulative(4)];         % 4th dashed line
    x5 = [GC_cumulative(5) GC_cumulative(5)];         % 5th dashed line
    %x6 = [GC_cumulative(6) GC_cumulative(6)];         % 6th dashed line
    %x7 = [GC_cumulative(7) GC_cumulative(7)];         % 7th dashed line
    
    
    p(1) = plot(x1,y1); p(2) = plot(x2,y1); p(3) = plot(x3,y1); p(4) = plot(x4,y1); 
    p(5) = plot(x5,y1); %p(6) = plot(x6,y1); p(7) = plot(x7,y1);
    hold on;
    legend('Static','HMM','Demographic','Intelligence','Affective','Personality','Sleep');%,'Anatomy','Other');

end



% %% plot predictions just for intelligence
% load('headers_grouped_category.mat');
% x = cell2mat(headers_grouped_category(:,4));
% x(x~=2) = 0;
% x(x == 2) = 1;
% 
% Y_intelligence = x.*Y;
% Y_intelligence(Y_intelligence == 0) = [];
% Z_intelligence = x.*Z;
% Z_intelligence(Z_intelligence == 0) = [];
% 
% X = [1:1:size(Y_intelligence,1)];
% 
% figure(8)
% %scatter(cat_reorder,Y_intelligence,10,repmat([115,213,246]/255,679,1),'filled')
% scatter(X,Y_intelligence,10,repmat([115,213,246]/255,size(Y_intelligence,1),1),'filled')
% grid on; hold on;
% %scatter(cat_reorder,Z_intelligence,10,repmat([220,94,146]/255,679,1),'filled')
% scatter(X,Z_intelligence,10,repmat([220,94,146]/255,size(Y_intelligence,1),1),'filled')
% title('Uncorrected (Intelligence)');
% legend('Static','HMM');
% xlabel('Variable'); ylabel('r^2')
% 
% figure(9)
% x = 1:size(explained_variance,1)
% y = explained_variance(:,1)
% scatter(x,y,10,repmat([115,213,246]/255,n_variables,1),'filled')
% grid on; hold on;
% title('Uncorrected (Intelligence)');
% legend('Static','HMM');
% xlabel('Variable'); ylabel('r^2')

%%
figure(3)
for i = 1:iterations
    subplot(3,1,i)
    Diff = explained_variance_hmm_reorg_trim(:,i)-explained_variance_static_reorg_trim
    
    Diff_colour(Diff<0) = -1; ;Diff_colour(Diff>0) = 1
    Diff_cell = num2cell(Diff_colour)'
    Diff_cell(Diff_colour == 1) = {[220,94,146]/255};
    Diff_cell(Diff_colour == -1) = {[115,213,246]/255};
    Diff_cell(Diff_colour == 0) = {[0 0 0 ]};
    Diff_mat = cell2mat(Diff_cell);

    positiveIndexes = Diff >= 0;
    negativeIndexes = Diff < 0;
    scatter(X(positiveIndexes),Diff(positiveIndexes),10,Diff_mat(positiveIndexes,:),'filled')
    hold on; grid on;
    scatter(X(negativeIndexes),Diff(negativeIndexes),10,Diff_mat(negativeIndexes,:),'filled')
    p(1) = plot(x1,y1); p(2) = plot(x2,y1); p(3) = plot(x3,y1); p(4) = plot(x4,y1); 
    p(5) = plot(x5,y1);
    hold on;
    title('Difference HMM - average FC')
    xlabel('Variable'); ylabel('Difference HMM - average-FC')
    legend('Better HMM','Better time-averaged FC','Demographic','Intelligence','Affective','Personality','Sleep');

    

end

%%
figure(4)
for p = 1:iterations
    Diff = explained_variance_hmm_reorg_trim(:,p)-explained_variance_static_reorg_trim;
    Diff(isnan(Diff)) = 0; % why do I have NaNs? Demographic variables not working?
    average_diff_vec = zeros(5,1);
    % Demographics = 1; % Intelligence = 2; % Affective = 3
    % Personality = 4; % Sleep = 5
    % Anatomy = 6; % Other = 7
    sort_categories = sort(category_index_vec);
    category_index_vec_ordered = sort_categories(1:219);
    
    for i = 1:5
        average_diff_vec(i) = sum((category_index_vec_ordered == i).*Diff)/...
        nnz((category_index_vec_ordered == i).*Diff);
    end
    
    subplot(3,1,p)
    cat = categorical({'Demographics','Intelligence','Affective','Personality','Sleep'});
    cat_reorder = reordercats(cat,{'Demographics','Intelligence','Affective','Personality','Sleep'});
    average_diff = -average_diff_vec;
    b = bar(cat_reorder,average_diff);
    b.FaceColor = 'flat';
    b.CData(1,:) = [239,48,37]/255; b.CData(2,:) = [240,136,50]/255; b.CData(3,:) = [228,230,90]/255;
    b.CData(4,:) = [113,226,86]/255; b.CData(5,:) = [151,33,245]/255; %b.CData(6,:) = [.2 .6 .5]/255;
    xlabel('Behavioural Groups'); ylabel('Average Difference')
    title('Repitition',p);
end

%%
figure(5)
    

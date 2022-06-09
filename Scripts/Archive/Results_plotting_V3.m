%% Load and plot distance matrices (DM)
load('KLdistances_ICA50_01_10_2021_run')
n_iterations = size(DistMat,3);

% Plot Structural-DM
figure(1)
subplot(2,2,1);
imagesc(DistStatic); 
title('Static distances between subjects');
xlabel('Subjects'); ylabel('Subjects');

% Plot HMM-DM (for each iteration)
for i = 1:n_iterations
    subplot(2,2,i+1);
    imagesc(DistMat(:,:,i));
    title('HMM distances between subjects - Repitition',i);
    xlabel('Subjects'); ylabel('Subjects');
end

%% Load and store predictions
load('staticFC_predictions_01_10_2021_run')
vars_hat_static = vars_hat;
explained_variance_static = explained_variance; % r²
load('HMM_predictions_01_10_2021_run_NEW')
vars_hat_hmm = vars_hat;
explained_variance_hmm = explained_variance; % r²

% Determine number of variables per behaviour category
n_variables = size(vars_hat_static,2);
load('type_beh.mat')
[GC,GR] = groupcounts(type_beh');
cumsum_groups = cumsum(GC);

%% Plot predictions (r2) for HMM & time-averaged FC
% Store variables that are reused for all three charts
X = (1:n_variables)';
Y = -explained_variance_static;
colour_cells = {[239,48,37]/255 [240,136,50]/255 [228,230,90]/255 [113,226,86]/255 [151,33,245]/255};
y_plot = [-0.2 0.2];
figure(2)
for i = 1:n_iterations

    % Plot HMM r^2
    subplot(3,1,i)
    Z = explained_variance_hmm(:,i);
    scatter(X,Y,10,repmat([115,213,246]/255,n_variables,1),'filled')
    grid on; hold on;
    scatter(X,Z,10,repmat([220,94,146]/255,n_variables,1),'filled')
   
    % Plot category divider lines
    for j = 1:5
        x_plot = [cumsum_groups(j) cumsum_groups(j)];
        plot(x_plot,y_plot,'Color',colour_cells{j});
    end

    % Format chart
    title('Uncorrected'); xlabel('Variable'); ylabel('r^2')
    ylim([-0.2 0.2])
    legend('Static','HMM','Demographic','Intelligence','Affective','Personality','Sleep');
    hold on
end

%% Perform t-tests on r2 for HMM & time-averaged FC results
% for var = 1:n_variables
%     [h,p] = ttest(explained_variance_hmm(var,1));
% end



%% Plot difference HMM - average-FC
figure(3)
y_plot = [-0.2 0.2];

for i = 1:n_iterations
    
    % Find the difference between the hmm and the static predictions
    Diff = explained_variance_hmm(:,i)-explained_variance_static;

    % Colour variables red for better HMM and blue for better time-average
    positiveIndexes = Diff >= 0; red_variables = positiveIndexes*([220,94,146]/255);
    negativeIndexes = Diff < 0; blue_variables = negativeIndexes*([115,213,246]/255);
    Diff_colour = red_variables + blue_variables;

    % Plot differences
    subplot(n_iterations,1,i)
    scatter(X(positiveIndexes),Diff(positiveIndexes),10,Diff_colour(positiveIndexes,:),'filled')
    hold on; grid on;
    scatter(X(negativeIndexes),Diff(negativeIndexes),10,Diff_colour(negativeIndexes,:),'filled')

    % Plot category divider lines
    for j = 1:5
        x_plot = [cumsum_groups(j) cumsum_groups(j)];
        plot(x_plot,y_plot,'Color',colour_cells{j});
    end

    % Format chart
    title('Difference HMM - average FC')
    xlabel('Variable'); ylabel('Difference HMM - average-FC')
    legend('Better HMM','Better time-averaged FC','Demographic','Intelligence','Affective','Personality','Sleep');

end

%% Plot the bar charts of differences between HMM and time-average FC representations
figure(4)
category_numbers = [1 2 3 4 6];
colour_cells = {[239,48,37]/255 [240,136,50]/255 [228,230,90]/255 [113,226,86]/255 [151,33,245]/255};

for i = 1:n_iterations
    
    % Calculate the average difference by behaviour category
    Diff = explained_variance_hmm(:,i)-explained_variance_static;
    average_diff_vec = zeros(5,1);
    for j = 1:5
        average_diff_vec(j) = sum((type_beh' == category_numbers(j)).*Diff)/...
        nnz((type_beh == category_numbers(j)).*Diff);
    end
    
    % Organise behaviour categories for bar chart
    cat = categorical({'Demographics','Intelligence','Affective','Personality','Sleep'});
    cat_reorder = reordercats(cat,{'Demographics','Intelligence','Affective','Personality','Sleep'});

    % Plot bar charts
    subplot(3,1,i)
    b = bar(cat_reorder,average_diff_vec);

    % Format chart (inc. colour of bars)
    xlabel('Behavioural Groups'); ylabel('Average Difference')
    title('Repitition',i);
    b.FaceColor = 'flat';
    for k = 1:5
        b.CData(k,:) = colour_cells{k};
    end

end

%% Perform permutation test on the average for each group (10000 permutations)
p_value_by_category = zeros(5,3);
observeddifference_by_category = zeros(5,3);
cumsum_groups_new = [1; cumsum_groups];

for i = 1:3
    % Note the explained variance for current HMM iteration
    explained_variance_hmm_iter = explained_variance_hmm(:,i);

    % Perform permutation tests for each behaviour category
    for j = 1:5
        hmm_vec_data = explained_variance_hmm_iter(cumsum_groups_new(j):cumsum_groups_new(j+1));
        [p_value_by_category(j,i), observeddifference_by_category(j,i)] = permutationTest(hmm_vec_data,explained_variance_static,10000);
    end
end

% Display p-values and mean differences
p_value_by_category
observeddifference_by_category

%% Plot explained variance r2 for FC-HMM vs Mean-HMM (and Var-HMM???)
% Load and store explained variance matrices for different HMM runs:
% zeromean = 1 (cell 1), covariance matrix uniquefull (cell 2) and diag (cell 3)
explained_variance_store = cell(3,1);
load('HMM_predictions_zeromean.mat'); explained_variance_store{1} = explained_variance;
load('HMM_predictions_covtype_uniquefull.mat'); explained_variance_store{2} = explained_variance;
load('HMM_predictions_covtype_diag.mat'); explained_variance_store{3} = explained_variance;

% Store variables that are reused for various charts
colour_cells = {[239,48,37]/255 [240,136,50]/255 [228,230,90]/255 [113,226,86]/255 [151,33,245]/255};
x = [0 0.35];

for h = 1:3
    figure(4 + h)
    for p = 1:3
        % Store the explained varianced from the different HMM runs
        explained_variance_store = explained_variance_store{h};
        X = explained_variance_store(:,p); Y = explained_variance_hmm(:,p);
        
        % Plot  the variables with a different colour for each category
        subplot(3,1,p)
        for i = 1:5
            X_plot = X(cumsum_groups_new(i):cumsum_groups_new(i+1));
            Y_plot = Y(cumsum_groups_new(i):cumsum_groups_new(i+1));
            scatter(X_plot,Y_plot,10,repmat(colour_cells{i},size(X_plot,1),1),'filled')
            hold on
        end
    
        % Plot the line y = x and format the chart
        plot(x,x, 'Color', [0.75 0.75 0.75]);
        legend('Demographic','Intelligence','Affective','Personality','Sleep');
        xlabel('Explained variance r^2 for Mean-HMM'); ylabel('Explained variance r^2 for FC-HMM');
    
    end
end



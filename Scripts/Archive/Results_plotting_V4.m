%%%%%%%%%%%%%%%%%%%%
%%% In this script, we plot the graphs found in Vidaurre et al.
%%% (NeuroImage, 2021). The majority of the graphs focus on the FC-HMM,
%%% which models each state with a covariance matrix (but no mean), so that
%%% we focus on functional connectivity. This unearths information that the
%%% time-averaged approach cannot.
%%%%%%%%%%%%%%%%%%%%
clear;clc;
% Load and plot distance matrices (DM)
DirResults = ['Dokumenter/MATLAB/HMMMAR_BG/HMMMAR Results/'];
DirFC = ['FC_HMM_zeromean_1_covtype_full_8_states/'];
load([DirResults DirFC 'KLdistances_ICA50.mat'])
n_iterations = size(DistMat,3);

% Plot Structural-DM
figure()
subplot(3,2,1);
imagesc(DistStatic); 
title('Static distances between subjects');
xlabel('Subjects'); ylabel('Subjects');

% Plot HMM-DM (for each iteration)
for i = 1:n_iterations
    subplot(3,2,i+1);
    imagesc(DistMat(:,:,i));
    title('HMM distances between subjects - Repitition',i);
    xlabel('Subjects'); ylabel('Subjects');
end

%% Load and store predictions
load([DirResults DirFC 'staticFC_predictions.mat'])
vars_hat_static = vars_hat;
explained_variance_static = explained_variance; % r²
load([DirResults DirFC 'hmm_predictions.mat'])
vars_hat_hmm = vars_hat;
explained_variance_hmm = explained_variance; % r²
n_variables = size(vars_hat_static,2);

% Determine number of variables per behaviour category
load('type_beh.mat')
load('feature_groupings.mat')
% [GC,GR] = groupcounts(type_beh');
% feature_groupings = cumsum(GC);
% feature_groupings = [1; feature_groupings];


%% Plot predictions (r2) for HMM & time-averaged FC
% Store variables that are reused for all three charts
X = (1:n_variables)';
Y = -explained_variance_static;
colour_cells = {[239,48,37]/255 [240,136,50]/255 [228,230,90]/255 [113,226,86]/255 [151,33,245]/255};
y_plot = [-0.3 0.3];

figure()
for i = 1:n_iterations

    % Plot HMM r^2
    subplot(5,1,i)
    Z = explained_variance_hmm(:,i);
    scatter(X,Y,10,repmat([115,213,246]/255,n_variables,1),'filled')
    grid on; hold on;
    scatter(X,Z,10,repmat([220,94,146]/255,n_variables,1),'filled')
   
    % Plot category divider lines
    for j = 1:length(feature_groupings)-1
        x_plot = [feature_groupings(j+1) feature_groupings(j+1)];
        plot(x_plot,y_plot,'Color',colour_cells{j});
    end

    % Format chart
    title('Uncorrected FC-HMM'); xlabel('Variable'); ylabel('r^2')
    ylim([-0.3 0.3])
    %legend('Static','HMM','Demographic','Intelligence','Affective','Personality','Sleep');
    hold on

end


%% Plot difference HMM - average-FC
figure()
y_plot = [-0.2 0.3];

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
    for j = 1:length(feature_groupings)-1
        x_plot = [feature_groupings(j+1) feature_groupings(j+1)];
        plot(x_plot,y_plot,'Color',colour_cells{j});
    end

    % Format chart
    title('Difference FC-HMM - average FC')
    xlabel('Variable'); %ylabel('Difference HMM - average-FC')
    %legend('Better HMM','Better time-averaged FC','Demographic','Intelligence','Affective','Personality','Sleep');

end

%% Plot the bar charts of differences between HMM and time-average FC representations (inc permutation tests)
figure()
category_numbers = [1 2 3 4 6];
colour_cells = {[239,48,37]/255 [240,136,50]/255 [228,230,90]/255 [113,226,86]/255 [151,33,245]/255};

% Initialise variables for permutation tests
p_value_by_category = zeros(5,3);
obs_diff = zeros(5,3);
load('feature_groupings.mat')

for i = 1:n_iterations
    
    % Calculate the average difference by behaviour category
    explained_variance_hmm_iter = explained_variance_hmm(:,i);
    Diff = explained_variance_hmm_iter-explained_variance_static;
    average_diff_vec = zeros(5,1);
    for j = 1:length(feature_groupings)-1
        average_diff_vec(j) = sum((type_beh' == category_numbers(j)).*Diff)/...
        nnz((type_beh == category_numbers(j)).*Diff);
    end

    % Perform permutation test on the average for each behaviour category (10,000 permutations)
    for j = 1:length(feature_groupings)-1
        hmm_vec_data = explained_variance_hmm_iter(feature_groupings(j):feature_groupings(j+1));
        [p_value_by_category(j,i), obs_diff(j,i)] = permutationTest(hmm_vec_data,explained_variance_static,10000);
    end

    % Create a matrix with asterisks for statisticaly significant results
    p_value_sig = zeros(5,1);
    p_value_sig(p_value_by_category(:,i) < 0.05) = 1;
    p_value_sig(p_value_by_category(:,i) < 0.01) = 2;
    p_value_str = num2cell(p_value_sig);
    p_value_str(p_value_sig == 1) = {'*'};
    p_value_str(p_value_sig == 2) = {'**'};
    p_value_str(p_value_sig == 0) = {' '};

    % Organise behaviour categories for bar chart
    cat = categorical({'Demographics','Intelligence','Affective','Personality','Sleep'});
    cat_reorder = reordercats(cat,{'Demographics','Intelligence','Affective','Personality','Sleep'});

    % Plot bar charts
    subplot(5,1,i)
    b = bar(cat_reorder,average_diff_vec);

    % Format chart (inc. colour of bars)
    xlabel('Behavioural Groups'); ylabel('Average Difference');
    title('Repetition',i);
    b.FaceColor = 'flat';
    for k = 1:5
        b.CData(k,:) = colour_cells{k};
    end

    % Add asterisk labels to represent statistical significance
    xtips1 = b(1).XEndPoints;
    ytips1 = b(1).YEndPoints;
    labels1 = p_value_str;
    text(xtips1,ytips1,labels1,'HorizontalAlignment','center',...
        'VerticalAlignment','bottom', 'FontSize', 20)
end


%% Plot explained variance r2 for FC-HMM vs (Mean/Var/Mean-FC)
% FC-HMM: Only covariance (zeromean = 1; covtype = 'full')
load([DirResults 'FC_HMM_zeromean_1_covtype_full_8_states/HMM_predictions.mat']); explained_variance_FC_HMM = explained_variance;

% (i)   Mean-HMM: Only mean (zeromean = 0; covtype = 'uniquefull')
% (ii)  Var-HMM: Only variance (zeromean = 1; covtype = 'diag')
% (iii)   Mean-FC-HMM: Both mean and covariance (zeromean = 0; covtype = 'full')
explained_variance_store = cell(3,1);
load([DirResults 'Mean_HMM_zeromean_0_covtype_uniquefull/HMM_predictions.mat']); explained_variance_store{1} = explained_variance; % (ii) Mean-HMM
load([DirResults 'VAR_HMM_zeromean_1_covtype_diag/HMM_predictions.mat']); explained_variance_store{2} = explained_variance; % (iii) Var-HMM
load([DirResults 'Mean_FC_HMM_zeromean_0_covtype_full/HMM_predictions.mat']); explained_variance_store{3} = explained_variance; % (iv) Mean-FC-HMM
labels = ["Mean-HMM","Var-HMM","Mean-FC-HMM"];

% Store variables that are reused for various charts
colour_cells = {[239,48,37]/255 [240,136,50]/255 [228,230,90]/255 [113,226,86]/255 [151,33,245]/255};
x = [0 0.35];

for h = 1:3
    figure()
    for p = 1:5
        % Store the explained varianced from the different HMM runs
        explained_variance_hmm_run = explained_variance_store{h};
        X = explained_variance_hmm_run(:,p); Y = explained_variance_FC_HMM(:,p);
        
        % Plot  the variables with a different colour for each category
        %subplot(3,1,p)
        for i = 1:5
            X_plot = X(feature_groupings(i):feature_groupings(i+1));
            Y_plot = Y(feature_groupings(i):feature_groupings(i+1));
            scatter(X_plot,Y_plot,10,repmat(colour_cells{i},size(X_plot,1),1),'filled')
            hold on
        end
    
        % Plot the line y = x and format the chart
        plot(x,x, 'Color', [0.75 0.75 0.75]);
        legend('Demographic','Intelligence','Affective','Personality','Sleep');
        xlabel(sprintf("Explained variance r^2 for %s",labels(h))); ylabel('Explained variance r^2 for FC-HMM');
    
    end
end



%% Comparing predictions (from same type of HMMs)
clear; clc;
% Set directory with predictions
DirOut = 'C:\Users\au699373\OneDrive - Aarhus Universitet\Dokumenter\MATLAB\HMMMAR_BG\HMMMAR Results\MAR_2022\All\Repetition_6_SC\';
target_vars = load('vars_target.mat','vars');


% load prediction 1
HMM_predictions1 = load([DirOut 'HMM_predictions_same_folds.mat'], 'vars_hat','subject_squared_error');

% load prediction 2
HMM_predictions2 = load([DirOut 'HMM_predictions_distmat_14.mat'], 'vars_hat','subject_squared_error');

% we begin by comparing the standard predictions and those where we have
% normalised the distance matrices for corresponding repetitions (it
% actually doesn't matter if corresponding as the reps are just random)

for j = 1%[1 5 10 15] % pick intelligence feature
    
    var_target = target_vars.vars(:,j);

    for i = 1:size(HMM_predictions1.vars_hat,3)
        % plot everything in blue
        vars_hat_plot1 = squeeze(HMM_predictions1.vars_hat(:,j,i)); mse_plot1 = squeeze(HMM_predictions1.subject_squared_error(:,j,i));
        vars_hat_plot2 = squeeze(HMM_predictions2.vars_hat(:,j,i)); mse_plot2 = squeeze(HMM_predictions2.subject_squared_error(:,j,i));

        % Predictions
%         figure(700); subplot(5,5,i); scatter(vars_hat_plot1,vars_hat_plot2,'x'); hold on; xlabel('Yhat (HMM 1)'); ylabel('Yhat (HMM 2)'); % legend('Subject prediction (inside range)','Subject prediction (outside range)');
%         sgtitle(sprintf('Predictions for variable %i with max = %.4g, mean = %.4g, min = %.4g',j,max(var_target), mean(var_target,'omitnan'),min(var_target)));
        % Errors
%         figure(701); subplot(5,5,i); scatter(mse_plot1,mse_plot2,'x'); hold on; xlabel('Error (HMM 1)'); ylabel('Error (HMM 2)'); 
%         sgtitle(sprintf('Error for variable %i with max = %.4g, mean = %.4g, min = %.4g',j,max(var_target), mean(var_target,'omitnan'),min(var_target)));
%         % Targets + errors
        figure(702); subplot(5,5,i); scatter(var_target,mse_plot1,'x'); hold on; xlabel(sprintf('Y (var %i)',j)); ylabel('Error');
        sgtitle(sprintf('Error for variable %i with max = %.4g, mean = %.4g, min = %.4g',j,max(var_target), mean(var_target,'omitnan'),min(var_target)));
        %sgtitle(sprintf('Error for age with max = %.4g, mean = %.4g, min = %.4g',max(var_target), mean(var_target),min(var_target)));
        
        coefficients = polyfit(var_target, mse_plot1, 2); % Get coefficients of a line fit through the data.
        xFit = linspace(min(var_target), max(var_target), 1000); % Create a new x axis with exactly 1000 points (or whatever you want)
        yFit = polyval(coefficients, xFit); % Get the estimated yFit value for each of those 1000 new x locations.
        %plot(x, y, 'b.', 'MarkerSize', 15); % Plot training data.
        %hold on; % Set hold on so the next plot does not blow away the one we just drew.
        plot(xFit, yFit, 'k-'); % Plot fitted line.
        grid on;


        % find values outside of range
        vars_hat_plot1_out = (squeeze(HMM_predictions1.vars_hat(:,j,i))>max(var_target)) + (squeeze(HMM_predictions1.vars_hat(:,j,i))<min(var_target));
        vars_hat_plot2_out = (squeeze(HMM_predictions2.vars_hat(:,j,i))>max(var_target)) + (squeeze(HMM_predictions2.vars_hat(:,j,i))<min(var_target));
        nnz_idx_vh = (vars_hat_plot1_out + vars_hat_plot2_out)>0;
        nnz_idx_vh_v1 = vars_hat_plot1_out>0;
        
        % plot values outside of range in red
        if sum(nnz_idx_vh) ~= 0
%             figure(700); scatter(vars_hat_plot1(nnz_idx_vh),vars_hat_plot2(nnz_idx_vh),'x','r'); 
%             figure(701); scatter(mse_plot1(nnz_idx_vh),mse_plot2(nnz_idx_vh),'x','r');
            figure(702); scatter(var_target(nnz_idx_vh_v1),mse_plot1(nnz_idx_vh_v1),'x','r');
        end
        
    end
    
end
% Question: why do some subjects have a high error, but they aren't a red
% cross (which implies they are outside the range of actual data points)?
% Basically these subjects are just predicted terrible, e.g. for
% HMM_predictions2 = load([DirOut 'HMM_predictions_same_folds.mat'], 'vars_hat','subject_squared_error');
% [HMM_predictions2.vars_hat(935,11,14) var_target(935)]
% We get 
% [84.3148  126.5796]
% so the predicted value (84.3) is terrible compared to the actual subject
% value (126.6) yet it is just within the range of data points (min = 84.2)

%%
DirOut = 'C:\Users\au699373\OneDrive - Aarhus Universitet\Dokumenter\MATLAB\HMMMAR_BG\HMMMAR Results\MAR_2022\All\Repetition_6_SC\';
target_vars = load('vars_target.mat','vars'); % load targets
HMM_predictions = load([DirOut 'HMM_predictions_norm_feature.mat'], 'vars_hat','subject_squared_error'); % load predictions


figure;
for j = 1:34%[1 5 10 15] % pick intelligence feature
    var_target = target_vars.vars(:,j);
    for i = 1:size(HMM_predictions.vars_hat,3)
        mse_plot1 = squeeze(HMM_predictions.subject_squared_error(:,j,i));
%         scatter(var_target,mse_plot1,'x'); hold on; xlabel(sprintf('Y (var %i)',j)); ylabel('Error');
%         sgtitle(sprintf('Error for variable %i with max = %.4g, mean = %.4g, min = %.4g',j,max(var_target), mean(var_target),min(var_target)));
        
        idx = ~isnan(var_target);
        coefficients = polyfit(var_target(idx), mse_plot1(idx), 2); % Get coefficients of a line fit through the data.
        xFit = linspace(min(var_target), max(var_target), 1000); % Create a new x axis with exactly 1000 points (or whatever you want)
        yFit = polyval(coefficients, xFit); % Get the estimated yFit value for each of those 1000 new x locations.
        subplot(6,6,j)
        plot(xFit, yFit, 'k-'); % Plot fitted line.
        %ylim([0 1000])
        grid on; hold on
        xlabel(sprintf('Intelligence Feature %i',j)); ylabel('Error')
        sgtitle('Prediction error curves for multiple HMM predictions')
        legend('Error curve')
        

    end
    
end


%%
j = 9;
maxk(squeeze(HMM_predictions1.vars_hat(:,j,:)),3)
maxk(squeeze(HMM_predictions2.vars_hat(:,j,:)),3)


%% Comparing HMM inferences 
% Here we use the same parameters for the HMM model, and the same folds for
% the prediction, so that the only difference is the stochasticity of the
% HMM inference

DirOut = 'C:\Users\au699373\OneDrive - Aarhus Universitet\Dokumenter\MATLAB\HMMMAR_BG\HMMMAR Results\MAR_2022\All\Repetition_6_SC\';
%DirOut = 'C:\Users\au699373\OneDrive - Aarhus Universitet\Dokumenter\MATLAB\HMMMAR_BG\HMMMAR Results\MAR_2022\All\Repetition_6_bias_correct\';
%target_vars = load('vars_target.mat','vars');
%target_vars = vars;

% load predictions
HMM_predictions = load([DirOut 'HMM_predictions_all_vars.mat'], 'vars_hat','subject_squared_error');

% figure;
% for j = 1:34
%     var_target = target_vars.vars(:,j);
%     scatter(j,max(var_target),'x','r'); hold on
%     scatter(j,maxk(squeeze(HMM_predictions.vars_hat(:,j,:)),1)','x','b');
% end

figure;
for j = 1:9%34
    subplot(3,3,j)
    %var_target = target_vars.vars(:,j);
    var_target = vars(:,j)
    %bar(j,max(var_target),'x','r'); hold on
    bar(1:size(HMM_predictions.vars_hat,3),maxk(squeeze(HMM_predictions.vars_hat(:,j,:)),1)'); hold on
    plot(xlim,[max(var_target) max(var_target)], 'r')
    title(sprintf('Intelligence Feature %i',j)); xlabel('HMM repetition'); ylabel('Prediction')
    legend('Max predicted value','Max feature value')
    sgtitle('Maximum predicted value vs the maximum subject value for each intelligence feature')

end



%% Ridge Regression Explained Variance
HMM_predictions = load([DirOut 'HMM_predictions_np5_rr_bias_correct.mat'], 'vars_hat','subject_squared_error','explained_variance');
n_vars = 34;
figure;
scatter(1:n_vars,HMM_predictions.explained_variance,'x','b');
xlabel('Intelligence Feature'); ylabel('Explained Variance, r^2')
legend('Predictions from HMMs'); title('Explained variance of predictions using ridge regression')


%% Ridge Regression (compare bias correct to non-bias correct)

DirOut = 'C:\Users\au699373\OneDrive - Aarhus Universitet\Dokumenter\MATLAB\HMMMAR_BG\HMMMAR Results\MAR_2022\All\Repetition_6_SC\';
target_vars = load('vars_target.mat','vars');
%HMM_predictions = load([DirOut 'HMM_predictions_np5_rr_bias_correct.mat'], 'vars_hat','subject_squared_error');
HMM_predictions = load([DirOut 'HMM_predictions_np5_rr.mat'], 'vars_hat','subject_squared_error');


count = 0;
for j = 11%[1 5 10 15] % pick intelligence feature
    var_target = target_vars.vars(:,j);

    for i = 1:size(HMM_predictions.vars_hat,3)
        count = count + 1;
        % plot everything in blue
        mse_plot1 = squeeze(HMM_predictions.subject_squared_error(:,j,i));

        % Targets + errors
        figure(802); subplot(5,5,count); scatter(var_target,mse_plot1,'x'); hold on; xlabel(sprintf('Y (var %i)',j)); ylabel('Error');
        sgtitle(sprintf('Error for variable %i with max = %.4g, mean = %.4g, min = %.4g',j,max(var_target), mean(var_target),min(var_target)));
        
        coefficients = polyfit(var_target, mse_plot1, 2); % Get coefficients of a line fit through the data.
        xFit = linspace(min(var_target), max(var_target), 1000); % Create a new x axis with exactly 1000 points (or whatever you want)
        yFit = polyval(coefficients, xFit); % Get the estimated yFit value for each of those 1000 new x locations.
        plot(xFit, yFit, 'k-'); grid on; % Plot fitted line.
        

        % find values outside of range
        vars_hat_plot1_out = (squeeze(HMM_predictions.vars_hat(:,j,i))>max(var_target)) + (squeeze(HMM_predictions.vars_hat(:,j,i))<min(var_target));
        nnz_idx_vh = vars_hat_plot1_out>0;
        
        % plot values outside of range in red
        if sum(nnz_idx_vh) ~= 0
            figure(802); scatter(var_target(nnz_idx_vh),mse_plot1(nnz_idx_vh),'x','r');
        end
        
    end
    
end



%% Are the bad predictions the same subjects?
HMM_predictions = load([DirOut 'HMM_predictions.mat'], 'vars_hat','subject_squared_error');
for j = 11
    % normalise squared error so that all intelligence features are on the
    % same scale
    %norm_sse = normc(squeeze(subject_squared_error(:,j,:)));
    A = squeeze(subject_squared_error(:,j,:));
    norm_sse = A./sum(A);

    % plot normalised squared error
    figure;
    imagesc(norm_sse); colorbar
end

%% Table of terrible predictions
clc
%bad_subject_predictions_high = zeros(1001,1);
%bad_subject_predictions_low = zeros(1001,1);

bad_subject_predictions_high_var = zeros(1001,34);
bad_subject_predictions_low_var = zeros(1001,34);

for j = 1:34  % pick intelligence feature
    var_target = target_vars.vars(:,j);
    bad_subject_predictions_high = zeros(1001,1);
    bad_subject_predictions_low = zeros(1001,1);

    for i = 1:size(HMM_predictions.vars_hat,3)
        % plot everything in blue
        vars_hat_store = squeeze(HMM_predictions.vars_hat(:,j,i));

        % find values (far) outside of range
        vars_hat_high = vars_hat_store>1.2*max(var_target);
        vars_hat_low = vars_hat_store<0.8*min(var_target);

        % note indexes out out-of-range predictions
        %idx_high = find(vars_hat_high == 1);
        %idx_low = find(vars_hat_low == 1);

        % Add 1 to each subject that has been predicted badly
        bad_subject_predictions_high = bad_subject_predictions_high + vars_hat_high;
        bad_subject_predictions_low = bad_subject_predictions_low + vars_hat_low;

    end

    % make leaderboard of subjects
    % columns of each variable
    % row is subject IDs
    %[M, I] = maxk(bad_subject_predictions_high,10);
    %[M I]
    bad_subject_predictions_high_var(:,j) = bad_subject_predictions_high; 

    figure(900);
    % plot histogram
    subplot(6,6,j)
    x = histogram(bad_subject_predictions_high,'BinWidth',1);
    %x = bar(bad_subject_predictions_high)
%     E = x.BinEdges;
%     y = x.BinCounts;
%     xloc = E(1:end-1)+diff(E)/2;
%     text(xloc, y+1, string(y))
    sgtitle('How many times are subjects badly predicted for each intelligence feature across 24 predictions?')
    title(sprintf('Intelligence Feature %i',j));
    xlabel('No. bad predictions per subject'); ylabel('Frequency');

   


end
% plot bar charts of number of times each subject is badly predicted
% (across all 34 variables and 24 repetitions)
figure; bar(sum(bad_subject_predictions_high_var,2))
title('Badly predicted subjects across 24 repetitions of HMM and 34 intelligence variables')
xlabel('Subjects'); ylabel('Number of (badly) out-of-range predictions')


% Generate leaderboard of poorly predicted subjects
bad_subject_predictions_high_var_new = [bad_subject_predictions_high_var sum(bad_subject_predictions_high_var,2)]; % add total bad predictions per subject
bad_subject_predictions_high_var_new= [(1:1001)' bad_subject_predictions_high_var_new]; % add idx of subjects
bad_subject_predictions_high_var_sort = sortrows(bad_subject_predictions_high_var_new,36,'descend'); % order by worst predicted

T = array2table(bad_subject_predictions_high_var_sort(1:50,[1 12 13 14 15 end]));
% T.Properties.VariableNames = ["Subject No.","Var1","Var2","Var3","Var4","Var5","Var6","Var7","Var8","Var9","Var10"...
%     ,"Var11","Var12","Var13","Var14","Var15","Var16","Var17","Var18","Var19","Var20","Var21","Var22","Var23","Var24","Var25","Var26","Var27","Var28"...
%     ,"Var29","Var30","Var31","Var32","Var33","Var34","Total bad predictions"];
T.Properties.VariableNames = ["Subject No.","Var11","Var12","Var13","Var14","Total bad predictions"];
figure; uitable('Data',T{:,:},'ColumnName',T.Properties.VariableNames,...
    'RowName',T.Properties.RowNames,'Units', 'Normalized', 'Position',[0, 0, 1, 1]);

%% Age error
DirOut = 'C:\Users\au699373\OneDrive - Aarhus Universitet\Dokumenter\MATLAB\HMMMAR_BG\HMMMAR Results\MAR_2022\All\Repetition_6_SC\';
HMM_predictions = load([DirOut 'HMM_predictions_all_vars.mat'], 'vars_hat','subject_squared_error'); % load predictions
load vars_126.mat % load targets


for j = 1 % Select age
    var_target = vars(:,j);

    for i = 1:size(HMM_predictions.vars_hat,3)
        % plot everything in blue
        mse_plot1 = squeeze(HMM_predictions.subject_squared_error(:,j,i));
        % Targets + errors
        figure(752); subplot(5,5,i); scatter(var_target,mse_plot1,'x'); hold on; xlabel(sprintf('Y (var %i)',j)); ylabel('Error');
        sgtitle(sprintf('Error for age with max = %.4g, mean = %.4g, min = %.4g',max(var_target), mean(var_target),min(var_target)));
        
        coefficients = polyfit(var_target, mse_plot1, 2); % Get coefficients of a line fit through the data.
        xFit = linspace(min(var_target), max(var_target), 1000); % Create a new x axis with exactly 1000 points (or whatever you want)
        yFit = polyval(coefficients, xFit); % Get the estimated yFit value for each of those 1000 new x locations.
        plot(xFit, yFit, 'k-'); grid on; % Plot fitted line.
        

        % find values outside of range
        nnz_idx_vh_v1 = ((squeeze(HMM_predictions.vars_hat(:,j,i))>max(var_target)) + (squeeze(HMM_predictions.vars_hat(:,j,i))<min(var_target)))>0;
        
        % plot values outside of range in red
        if sum(nnz_idx_vh_v1) ~= 0; figure(752); scatter(var_target(nnz_idx_vh_v1),mse_plot1(nnz_idx_vh_v1),'x','r'); end
        
    end
end

















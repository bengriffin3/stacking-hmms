clear; clc;
% Let's predict age as that's quite an easy one
%load('vars_126.mat')
load('vars_target.mat')
for variable_ex = 11%1:34
%variable_ex = 4;
var_target = vars(:,variable_ex);
n_subjects = size(var_target,1);

simulation = 102;
sim_params.stack_method = 'least_squares';
%sim_params.stack_method = 'random_forest';

% Generate predictions that we are going to stack (would be cross-validated predictions)

% Simulation 1
% 2 predictions, each bad for 1/2 the subjects
if simulation == 1
    hmm_1 = [var_target(1:500) + randn(500,1); var_target(501:end) + 5*ones(501,1)];
    hmm_2 = [var_target(1:500) + 5*ones(500,1); var_target(501:end) + randn(501,1)];
    X = [hmm_1 hmm_2];
end
% Result: 

% Simulation 2
% 5 predictions, each bad for 1/5 the subjects
if simulation == 2
    hmm_1 = [var_target(1:200) + 5*ones(200,1); var_target(201:end) + randn(801,1)];
    hmm_2 = [var_target(1:200) + randn(200,1); var_target(201:400) + 5*ones(200,1); var_target(401:end) + randn(601,1)];
    hmm_3 = [var_target(1:400) + randn(400,1); var_target(401:600) + 5*ones(200,1); var_target(601:end) + randn(401,1)];
    hmm_4 = [var_target(1:600) + randn(600,1); var_target(601:800) + 5*ones(200,1); var_target(801:end) + randn(201,1)];
    hmm_5 = [var_target(1:800) + randn(800,1); var_target(801:end) + 5*ones(201,1);];
    X = [hmm_1 hmm_2 hmm_3 hmm_4 hmm_5];
end
% Result: 

% Simulation 3
% 2 predictions, 1 good for all subjects, 1 bad for 500 subjects
if simulation == 3
    hmm_1 = [var_target(1:500) + randn(500,1); var_target(501:end)];
    hmm_2 = [var_target(1:500) + 5*ones(500,1); var_target(501:end) + randn(501,1)];
    X = [hmm_1 hmm_2];
end
% Result: 

% Simulation 4
% 5 predictions, 1 good for all subjects, 4 bad for 200 subjects
if simulation == 4
    hmm_1 = [var_target(1:200) + randn(200,1); var_target(201:end)  + randn(801,1)];
    hmm_2 = [var_target(1:200) + randn(200,1); var_target(201:400) + 5*ones(200,1); var_target(401:end) + randn(601,1)];
    hmm_3 = [var_target(1:400) + randn(400,1); var_target(401:600) + 5*ones(200,1); var_target(601:end) + randn(401,1)];
    hmm_4 = [var_target(1:600) + randn(600,1); var_target(601:800) + 5*ones(200,1); var_target(801:end) + randn(201,1)];
    hmm_5 = [var_target(1:800) + randn(800,1); var_target(801:end) + 5*ones(201,1);];
    X = [hmm_1 hmm_2 hmm_3 hmm_4 hmm_5];
end
% Result: 

% Simulation 5
% 5 predictions, 4 good for all subjects, 1 bad for 200 subjects
if simulation == 5
    hmm_1 = [var_target(1:200) + randn(200,1); var_target(201:end)];
    hmm_2 = [var_target(1:200) + randn(200,1); var_target(201:400) ; var_target(401:end) + randn(601,1)];
    hmm_3 = [var_target(1:400) + randn(400,1); var_target(401:600) ; var_target(601:end) + randn(401,1)];
    hmm_4 = [var_target(1:600) + randn(600,1); var_target(601:800) ; var_target(801:end) + randn(201,1)];
    hmm_5 = [var_target(1:800) + randn(800,1); var_target(801:end) + 5*ones(201,1);];
    X = [hmm_1 hmm_2 hmm_3 hmm_4 hmm_5];
end
% Result: it knows to choose the one without much noise

% Simulation 6
% 2 predictions, 1 bad for 1/5 the subjects, 1 bad for 4/5s
if simulation == 6
    hmm_1 = [var_target(1:200) + randn(200,1); var_target(201:end) + 5*ones(801,1)];
    hmm_2 = [var_target(1:200) + 5*ones(200,1); var_target(201:end) + randn(801,1)];
    X = [hmm_1 hmm_2];
end
% Result: weights are roughly 0.2 and 0.8


% Simulation 7
% 2 predictions, 1 (very slightly) worse than the other
if simulation == 7
    hmm_1 = var_target + 3.000001*ones(1001,1);
    hmm_2 = var_target + 3*ones(1001,1);
    X = [hmm_1 hmm_2];
end
% Result: weights are basically 1 and 0 - the model knows to ignore the
% worse prediction


% Simulation 8
% We have 5 HMMs, all with quite a bit of noise - can the stacking help
% ignore the noise?
noise_amp_all = 0.3;
if simulation == 8
    hmm_1 = var_target + noise_amp_all*randn(1001,1);
    hmm_2 = var_target + noise_amp_all*randn(1001,1);
    hmm_3 = var_target + noise_amp_all*randn(1001,1);
    hmm_4 = var_target + noise_amp_all*randn(1001,1);
    hmm_5 = var_target + noise_amp_all*randn(1001,1);
    X = [hmm_1 hmm_2 hmm_3 hmm_4 hmm_5];
end
% Result: 


% Simulation 9
% Now, as we decrease/increase this noise, how does this affect the
% stacking? How is the correlation changing during this?
%%% See end of script


%%% Run Simulation 10 vs Simulation 11 vs Simulation 12


% Simulation 10
if simulation == 10
    hmm_1 = [var_target(1:200) + randn(200,1); var_target(201:end) + 5*ones(801,1)];
    hmm_2 = [var_target(1:200) + 5*ones(200,1); var_target(201:400)  + randn(200,1); var_target(401:end) + 5*ones(601,1)];
    hmm_3 = [var_target(1:400) + 5*ones(400,1); var_target(401:600)  + randn(200,1); var_target(601:end) + 5*ones(401,1)];
    hmm_4 = [var_target(1:600) + 5*ones(600,1); var_target(601:800)  + randn(200,1); var_target(801:end) + 5*ones(201,1)];
    hmm_5 = [var_target(1:800) + 5*ones(800,1); var_target(801:end) + randn(201,1);];
    X = [hmm_1 hmm_2 hmm_3 hmm_4 hmm_5];
end
% Result: 

% Simulation 11
if simulation == 11
    hmm_1 = [var_target(1:200) + randn(200,1); var_target(201:end) + 5*ones(801,1)];
    hmm_2 = [var_target(1:100) + 5*ones(100,1); var_target(101:300) + randn(200,1); var_target(301:end) + 5*ones(701,1)];
    hmm_3 = [var_target(1:200) + 5*ones(200,1); var_target(201:400) + randn(200,1); var_target(401:end) + 5*ones(601,1)];
    hmm_4 = [var_target(1:300) + 5*ones(300,1); var_target(301:500) + randn(200,1); var_target(501:end) + 5*ones(501,1)];
    hmm_5 = [var_target(1:400) + 5*ones(400,1); var_target(401:600) + randn(200,1); var_target(601:end) + 5*ones(401,1)];
    X = [hmm_1 hmm_2 hmm_3 hmm_4 hmm_5];
end
% Result: 


% Simulation 12
%%% let's simulate changing the number of subjects predictions each HMM has
%%% done well and the number of subjects that are commonly predicted well
%%% by different the different HMMs. Then make a 3D plot where we show
%%% something about number of accurately predicted subjects (overall?) vs
%%% number of unique subjects predicted accurately (overall?) and then
%%% maybe correlation between HMMs or overall error of HMMs

% See end of script

% % Simulation 13
% % Nonlinear errors (random forests better than least squares?
% if simulation == 13
%     hmm_1 = [var_target(1:200) + randn(200,1); var_target(201:end) + exp(0:log(5)/800:log(5))*ones(801,1)];
%     hmm_2 = [var_target(1:100) + exp(0:log(5)/99:log(5))*ones(100,1); var_target(101:300) + randn(200,1); var_target(301:end) + exp(0:log(5)/700:log(5))*ones(701,1)];
%     hmm_3 = [var_target(1:200) + exp(0:log(5)/199:log(5))*ones(200,1); var_target(201:400) + randn(200,1); var_target(401:end) + exp(0:log(5)/600:log(5))*ones(601,1)];
%     hmm_4 = [var_target(1:300) + exp(0:log(5)/299:log(5))*ones(300,1); var_target(301:500) + randn(200,1); var_target(501:end) + exp(0:log(5)/500:log(5))*ones(501,1)];
%     hmm_5 = [var_target(1:400) + exp(0:log(5)/399:log(5))*ones(400,1); var_target(401:600) + randn(200,1); var_target(601:end) + exp(0:log(5)/400:log(5))*ones(401,1)];
%     X = [hmm_1 hmm_2 hmm_3 hmm_4 hmm_5];
% end
% % Result: 

% Simulation 14
% Nonlinear errors (random forests better than least squares?
% Exponential
if simulation == 14
    hmm_1 = [var_target(1:200) + randn(200,1); var_target(201:end) + exp(0:log(5)/800:log(5))'.*ones(801,1)];
    hmm_2 = [var_target(1:100) + exp(0:log(5)/99:log(5))'.*ones(100,1); var_target(101:300) + randn(200,1); var_target(301:end) + exp(0:log(5)/700:log(5))'.*ones(701,1)];
    hmm_3 = [var_target(1:200) + exp(0:log(5)/199:log(5))'.*ones(200,1); var_target(201:400) + randn(200,1); var_target(401:end) + exp(0:log(5)/600:log(5))'.*ones(601,1)];
    hmm_4 = [var_target(1:300) + exp(0:log(5)/299:log(5))'.*ones(300,1); var_target(301:500) + randn(200,1); var_target(501:end) + exp(0:log(5)/500:log(5))'.*ones(501,1)];
    hmm_5 = [var_target(1:400) + exp(0:log(5)/399:log(5))'.*ones(400,1); var_target(401:600) + randn(200,1); var_target(601:end) + exp(0:log(5)/400:log(5))'.*ones(401,1)];
    X = [hmm_1 hmm_2 hmm_3 hmm_4 hmm_5];
end
% Result: 

% % Simulation 15
% % Nonlinear errors (random forests better than least squares?
% % Cubic relationship
% if simulation == 15
%     hmm_1 = [var_target(1:200) + randn(200,1); var_target(201:end) + exp(0:log(5)/800:log(5))'.*ones(801,1)];
%     hmm_2 = [var_target(1:100) + exp(0:log(5)/99:log(5))'.*ones(100,1); var_target(101:300) + randn(200,1); var_target(301:end) + exp(0:log(5)/700:log(5))'.*ones(701,1)];
%     hmm_3 = [var_target(1:200) + exp(0:log(5)/199:log(5))'.*ones(200,1); var_target(201:400) + randn(200,1); var_target(401:end) + exp(0:log(5)/600:log(5))'.*ones(601,1)];
%     hmm_4 = [var_target(1:300) + exp(0:log(5)/299:log(5))'.*ones(300,1); var_target(301:500) + randn(200,1); var_target(501:end) + exp(0:log(5)/500:log(5))'.*ones(501,1)];
%     hmm_5 = [var_target(1:400) + exp(0:log(5)/399:log(5))'.*ones(400,1); var_target(401:600) + randn(200,1); var_target(601:end) + exp(0:log(5)/400:log(5))'.*ones(401,1)];
%     X = [hmm_1 hmm_2 hmm_3 hmm_4 hmm_5];
% end
% % Result: 

% Simulation 100
% Real data from predicting age
if simulation == 100
    DirOut = 'C:\Users\au699373\OneDrive - Aarhus Universitet\Dokumenter\MATLAB\HMMMAR_BG\HMMMAR Results\MAR_2022\All\Repetition_6_SC\';
    %HMM_predictions = load([DirOut 'HMM_predictions_same_folds.mat'], 'vars_hat');
    HMM_predictions = load([DirOut 'HMM_predictions_np5_rr_bias_correct.mat'], 'vars_hat');
    X_all = squeeze(HMM_predictions.vars_hat(:,1,:)); % note 24 predictions
    hmmKEEP = ~(max(X_all) > max(var_target));
    X = X_all(:,hmmKEEP);% remove bad predictions

end
% Results: as we know, since the same HMM parameters are used for these, we
% get super correlated predictions and so stacking doesn't improve anything

% Simulation 101
% Real data from predicting intelligence variable 1
if simulation == 101
    DirOut = 'C:\Users\au699373\OneDrive - Aarhus Universitet\Dokumenter\MATLAB\HMMMAR_BG\HMMMAR Results\MAR_2022\All\Repetition_3\';
    HMM_predictions = load([DirOut 'HMM_predictions.mat'], 'vars_hat','explained_variance');
    X_all = squeeze(HMM_predictions.vars_hat(:,variable_ex,:)); % note 24 predictions
    hmmKEEP = ~(max(X_all) > max(var_target)*1.1);
    X = X_all(:,hmmKEEP);% remove bad predictions

end

% Simulation 102
% Real data from predicting intelligence variable 1
if simulation == 102
    DirOut = 'C:\Users\au699373\OneDrive - Aarhus Universitet\Dokumenter\MATLAB\HMMMAR_BG\HMMMAR Results\MAR_2022\All\Repetition_6_SC\';
    HMM_predictions = load([DirOut 'HMM_predictions.mat'], 'vars_hat','explained_variance');
    HMM_predictions_bias_correct = load([DirOut 'HMM_predictions_np5_rr_bias_correct.mat']);
    HMM_predictions_LOO_hyper_tune = load([DirOut 'HMM_predictions_LOO_hyper_tune.mat']);

    X_all = [squeeze(HMM_predictions.vars_hat(:,variable_ex,:))  squeeze(HMM_predictions_bias_correct.vars_hat(:,variable_ex,:)) squeeze(HMM_predictions_LOO_hyper_tune.vars_hat(:,variable_ex,:))]; % note 24 predictions
    %X_all = squeeze(HMM_predictions_bias_correct.vars_hat(:,variable_ex,:));
    hmmKEEP = ~(max(X_all) > max(var_target)*1.1);
    X = X_all(:,hmmKEEP);% remove bad predictions

end




%% Now we form new prediction using cross-validated least-squares

% remove NaNs
X = rmmissing(X);
var_target = rmmissing(var_target);


y = var_target;
%y = y-mean(y)



% stack predictions
[yhat,W] = stacking_regress_ben(X,y,sim_params);

% test model
stack_sse = (yhat - y).^2; stack_mse = sum(stack_sse)/n_subjects; stack_ev = corr(yhat,var_target).^2;


%%
% Calculate errors for original predictions
n_models = size(X,2);
X_sse = (X - y).^2; X_mse = sum(X_sse)/n_subjects; X_ev = corr(X,y).^2;
% 
% % SCATTER PLOTS
% % Plot errors
% figure;
% for i = 1:n_models
%     subplot(2,n_models+1,i); scatter(var_target, X_sse(:,i),'x','b'); title(sprintf('HMM %i, MSE = %f',i, X_mse(i)));
%     xlabel('Target feature'); ylabel('Subject squared error'); legend('Subject');
% end
% 
% % plot stacked errors
% subplot(2,n_models+1,i+1); scatter(y, stack_sse,'x','b'); title(sprintf('Stacked, MSE = %f',stack_mse));
% xlabel('Target feature'); ylabel('Subject squared error'); legend('Subject');
% 
% sgtitle(sprintf('Subject feature target vs squared error (individual and stacked prediction (%s)) ',sim_params.stack_method))
% % HISTOGRAM PLOTS
% % Plot errors
% %figure;
% for i = 1:n_models
%     subplot(2,n_models+1,i+n_models+1); histogram(X_sse(:,i)); title(sprintf('HMM %i, MSE = %f',i, X_mse(i)));
%     xlabel('Error of predictions'); ylabel('Frequency');
% end
% % plot stacked errors
% subplot(2,n_models+1,i+n_models+2); histogram(stack_sse); title(sprintf('Stacked, MSE = %f',stack_mse));
% xlabel('Error of predictions'); ylabel('Frequency');
% 
% %%
% % Plot errors
% figure;
% for i = 1:n_models
%     subplot(n_models+1,1,i); imagesc(X_sse(:,i)); colorbar; title(sprintf('Prediction %i, MSE = %0.2f, EV = %0.2f',i,X_mse(i), X_ev(i))); ylabel('Subject Number');
%     
% end
% 
% % plot stacked errors
% subplot(n_models+1,1,i+1); imagesc(stack_sse); colorbar; title(sprintf('Stacked prediction, MSE= %0.2f, EV = %0.2f',stack_mse, stack_ev)); ylabel('Subject Number');
%%
% Form x tick labels
xhmmticks = cell(n_models+1,1);
for i = 1:n_models; xhmmticks{i} = sprintf('hmm %i',i); end
xhmmticks{i+1} = 'Stacked';

% plot mean squared errors
figure;
subplot(3,1,1);
Y = [X_mse stack_mse];
bar(Y);set(gca, 'XTick', 1:length(xhmmticks), 'XTickLabels', xhmmticks);
set(gca, 'XTick', 1:length(xhmmticks), 'XTickLabels', xhmmticks)
xlabel('HMM Prediction'); ylabel('MSE'); title('Mean Squared Error');
text(1:length(Y),Y,num2str(Y'),'vert','bottom','horiz','center');
ylim([0 max(X_mse)+01])

% plot explained variances
subplot(3,1,2);
Y = [X_ev' stack_ev];
bar(Y); set(gca, 'XTick', 1:length(xhmmticks), 'XTickLabels', xhmmticks);
xlabel('HMM Prediction'); ylabel('EV'); title('Explained Variance');
text(1:length(Y),Y,num2str(Y'),'vert','bottom','horiz','center');

% plot weights
subplot(3,1,3);
Y = W';
bar(Y,'FaceColor',[0.8500 0.3250 0.0980]);
set(gca, 'XTick', 1:length(xhmmticks), 'XTickLabels', xhmmticks);
xlabel('HMM Prediction'); ylabel('Weights'); title('Stacking weights');
text(1:length(Y),Y,num2str(Y'),'vert','bottom','horiz','center');

sgtitle(sprintf('%s', sim_params.stack_method))
end
% %% Correlation between HMMs
% % theoretically, lower correlation should lead to better stacking (makes more sense with
% % multiple HMMs)
% X_corr = corr(X);
% mean_corr = sum(sum(triu(X_corr,1))) / nnz(triu(X_corr,1));
% figure;
% % correlation between all HMM predictions
% subplot(2,1,1); imagesc(X_corr); colorbar;
% xlabel('HMM prediction'); ylabel('HMM prediction');
% title(sprintf('Correlation between HMM predictions to be stacked, mean correlation = %0.2f',mean_corr))
% 
% if W == 0; disp('No weights for random forest algorithm'); else
%     % correlations between HMM predictions that are (mildly) used in the stacking
%     % (weights above 0.05)
%     idxLargeWeights = W>0.05;
%     subplot(2,1,2); imagesc(corr(X(:,idxLargeWeights))); colorbar;
%     xlabel('HMM prediction'); ylabel('HMM prediction');
%     title(sprintf('Correlation between HMM predictions to be stacked, mean correlation = %0.2f',mean_corr))
% end
% 
% %%
% figure;
% for i = 1:n_models
%     subplot(1,n_models+1,i); scatter(var_target, X(:,i),'x')
%     xlabel('Subject target'); ylabel('Subject prediction')
%     title(sprintf('HMM prediction %i',i))
% end
% subplot(1,n_models+1,i+1); scatter(var_target, yhat,'x')
% xlabel('Subject target'); ylabel('Subject prediction')
% title('Stacked prediction')
% 
% sgtitle('Targets and predictions original and stacked prediction')
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% %% Plot histogram of actual variables and predictions
% if simulation == 100 || simulation == 101
%     figure;
%     % let's plot for first variable for now
%     for i = 1:34
%         subplot(6,6,i)
%         histogram(vars(:,i))
%         title(sprintf('Variable no. %i',i))
%     end
%     sgtitle('Distribution of target features for 34 intelligence variables')
%     figure;
%     % let's plot for first variable for now
%     var_plot = squeeze(HMM_predictions.vars_hat(:,11,:));
%     for i = 1:15 % check out the first 15
%         subplot(5,3,i)
%         histogram(var_plot(:,i))
%         title(sprintf('HMM repetition %i', i))
%     end
%     sgtitle('Predictions for variable 1')
% 
%     %%% NON-STACKED
% %     % Plot histograms of predictions as well as actual variables (non-stacked predictions)
% %     % Let's just show the first prediction of each one
% %     % let's plot for first variable for now
% %     figure;
% %     for i = 1:34
% %         subplot(6,6,i)
% %         histogram(vars(:,i)); hold on
% %         var_plot = squeeze(HMM_predictions.vars_hat(:,i,:));
% %         histogram(var_plot(:,2))
% % 
% %         title(sprintf('Variable no. %i',i))
% %     end
% %     sgtitle('Distribution of target features and predictions for 34 intelligence variables')
%     vars_126 = load('vars_126.mat','vars');
%     figure;
%     for i = 1:24
%         subplot(5,5,i)
%         histogram(vars_126.vars(:,1)); hold on;
%         histogram(X(:,i))
%         xlabel('Prediction'); ylabel('Frequency');
%         title(sprintf('Variable no. %i',i))
%     end
%     sgtitle('Predictions for different HMM repetitions')
% 
% else
%     figure;
%     for i = 1:n_models
%         subplot(1,n_models+1,i)
%         histogram(var_target(:,1)); hold on;
%         histogram(X(:,i))
%         xlabel('Prediction'); ylabel('Frequency');
%         title(sprintf('Prediction %i',i))
%     end
%     sgtitle('Predictions for different HMM repetitions')
% 
%     subplot(1,n_models+1,i+1)
%     histogram(var_target(:,1)); hold on;
%     histogram(yhat)
%     xlabel('Prediction'); ylabel('Frequency');
%     title('Stacked prediction')
% 
% end
%% SIMULATION 9

if simulation == 9
    for j = 2%1:2
        switch j
            case 1
                sim_params.stack_method = 'random_forest'; noise_amp_all = 0:1:15;
            case 2
                sim_params.stack_method = 'least_squares'; noise_amp_all = 0:0.1:15;
        end

    n_models = 5;

    stack_sse_all = NaN(1001,length(noise_amp_all)); stack_mse_all = NaN(1,length(noise_amp_all)); stack_ev_all = NaN(1,length(noise_amp_all));
    X_sse_all = NaN(1001,n_models,length(noise_amp_all)); X_mse_all = NaN(n_models,length(noise_amp_all)); X_ev_all = NaN(n_models,length(noise_amp_all));
    mean_corr_all = NaN(1,length(noise_amp_all));

    for i = 1:length(noise_amp_all)
        %i
        noise_amp = noise_amp_all(i);
        hmm_1 = var_target + noise_amp*randn(1001,1);
        hmm_2 = var_target + noise_amp*randn(1001,1);
        hmm_3 = var_target + noise_amp*randn(1001,1);
        hmm_4 = var_target + noise_amp*randn(1001,1);
        hmm_5 = var_target + noise_amp*randn(1001,1);
        X = [hmm_1 hmm_2 hmm_3 hmm_4 hmm_5];
        y = var_target;

        X_sse_all(:,:,i) = (X - y).^2;
        X_mse_all(:,i) = sum(X_sse_all(:,:,i))/n_subjects;
        X_ev_all(:,i) = corr(X,y).^2;

        [yhat,W] = stacking_regress_ben(X,y,sim_params); % stack predictions
        stack_sse_all(:,i) = (yhat - y).^2;
        stack_mse_all(i) = sum(stack_sse_all(:,i))/n_subjects;
        stack_ev_all(i) = corr(yhat,var_target).^2; % test model

        % Note correlation between HMMs
        X_corr = corr(X);
        mean_corr_all(i) = sum(sum(triu(X_corr,1))) / nnz(triu(X_corr,1));
        W

    end

    % Calculate errors for original predictions
    X_sse = (X - y).^2; X_mse = sum(X_sse)/n_subjects; X_ev = corr(X,y).^2;

    % plot mean squared errors as noise increases
    x_plot = noise_amp_all;
    y_plot = [X_mse_all; stack_mse_all];

    if strcmp(sim_params.stack_method, 'random_forest')
         figure;
         yyaxis left
         plot(x_plot,y_plot(6,:),'--'); hold on
    else
        yyaxis left
        plot(x_plot,y_plot(6,:),':'); hold on

        for i = 1:size(y_plot,1)-1
            plot(x_plot,y_plot(i,:),'-'); hold on
        end

        xlabel('Magnitude of noise, M'); ylabel('Mean Squared Error');
        title('The effect that adding M*N(0,1) noise to individual predictions has on the error of (stacked + individual predictions)')


        yyaxis right
        plot(x_plot,mean_corr_all)
        ylabel('Mean correlation between individual predictions')
        legend('Stacked Prediction (RF)','Stacked Prediction(LS)','Individual Predictions')
    end

    end
    

%     % plot explained variance as noise increases
%     x_plot = noise_amp_all;
%     y_plot = [X_ev_all; stack_ev_all];
%     figure; plot(x_plot,y_plot); xlabel('Magnitude of noise'); ylabel('Explained Variance');
%     legend('Prediction 1','Prediction 2','Prediction 3','Prediction 4','Prediction 5','Stacked Prediction')
%     title('The effect that adding N(0,1) noise to individual predictions has on the explained variance of each prediction and their stacked prediction')


end

%% SIMULATION 12

if simulation == 12
    sim_params.stack_method = 'least_squares';
    %sim_params.stack_method = 'random_forest';

    n_models = 5;
    error_amp = 5;
    n_acc_sub_all = 100%1:10:191; % max 200 since using 5 HMMs and 1001 subjects
    n_cross_sub_all = 10%0:10:191; % number of subjects that AREN'T crossovered

    stack_sse_all = NaN(1001,length(n_acc_sub_all),length(n_cross_sub_all)); stack_mse_all = NaN(length(n_acc_sub_all),length(n_cross_sub_all)); stack_ev_all = NaN(length(n_acc_sub_all),length(n_cross_sub_all));
    X_sse_all = NaN(1001,n_models,length(n_acc_sub_all),length(n_cross_sub_all)); X_mse_all = NaN(n_models,length(n_acc_sub_all)); X_ev_all = NaN(n_models,length(n_acc_sub_all));
    mean_corr_all = NaN(length(n_acc_sub_all),length(n_cross_sub_all)); n_unique_good_subjects = NaN(length(n_acc_sub_all),length(n_cross_sub_all));

    for i = 1:length(n_acc_sub_all)
        i
        for j = 1:length(n_cross_sub_all)
            n_acc_sub = n_acc_sub_all(i);
            n_cross_sub = n_cross_sub_all(j);

            hmm_1 = [var_target(1:n_acc_sub) + randn(n_acc_sub,1); var_target(n_acc_sub+1:end) + 5*ones(length(var_target(n_acc_sub+1:end)),1)];
            hmm_2 = [var_target(1:1*(n_acc_sub-n_cross_sub)) + error_amp*ones(1*(n_acc_sub-n_cross_sub),1); var_target(1*(n_acc_sub-n_cross_sub)+1:1*(n_acc_sub-n_cross_sub)+n_acc_sub) + randn(n_acc_sub,1); var_target(1*(n_acc_sub-n_cross_sub)+n_acc_sub+1:end) + error_amp*ones(length(var_target(1*(n_acc_sub-n_cross_sub)+n_acc_sub+1:end)),1)];
            hmm_3 = [var_target(1:2*n_cross_sub) + 5*ones(2*n_cross_sub,1); var_target(2*n_cross_sub+1:2*n_cross_sub+n_acc_sub) + randn(n_acc_sub,1); var_target(2*n_cross_sub+n_acc_sub+1:end) + 5*ones(length(var_target(2*n_cross_sub+n_acc_sub+1:end)),1)];
            hmm_4 = [var_target(1:3*n_cross_sub) + 5*ones(3*n_cross_sub,1); var_target(3*n_cross_sub+1:3*n_cross_sub+n_acc_sub) + randn(n_acc_sub,1); var_target(3*n_cross_sub+n_acc_sub+1:end) + 5*ones(length(var_target(3*n_cross_sub+n_acc_sub+1:end)),1)];
            hmm_5 = [var_target(1:4*n_cross_sub) + 5*ones(4*n_cross_sub,1); var_target(4*n_cross_sub+1:4*n_cross_sub+n_acc_sub) + randn(n_acc_sub,1); var_target(4*n_cross_sub+n_acc_sub+1:end) + 5*ones(length(var_target(4*n_cross_sub+n_acc_sub+1:end)),1)];
            X = [hmm_1 hmm_2 hmm_3 hmm_4 hmm_5];
            y = var_target;

            % note individual errors
            X_sse_all(:,:,i,j) = (X - y).^2;
            X_mse_all(:,i,j) = sum(X_sse_all(:,:,i,j))/n_subjects;
            X_ev_all(:,i,j) = corr(X,y).^2;

            % note stacked errors
            [yhat,W] = stacking_regress_ben(X,y,sim_params); % stack predictions
            stack_sse_all(:,i,j) = (yhat - y).^2;
            stack_mse_all(i,j) = sum(stack_sse_all(:,i,j))/n_subjects;
            stack_ev_all(i,j) = corr(yhat,var_target).^2; % test model

            % note correlation between HMMs
            X_corr = corr(X);
            mean_corr_all(i,j) = sum(sum(triu(X_corr,1))) / nnz(triu(X_corr,1));

            %%%% NEED TO SORT THIS OUT
            n_unique_good_subjects(i,j) = 5*n_acc_sub - 8*(n_acc_sub-n_cross_sub);
        end
    end
    stack_mse_all
    n_unique_good_subjects

    figure; imagesc(stack_mse_all); colorbar;
    xlabel('Number of crossover subjects'); ylabel('Number of accurate subjects');
    title('Mean squared error as (distinct) accurate subjects varies')
    xticklabels({'10','30','50','70','90','110','130','150','170','190'})
    yticklabels({'10','30','50','70','90','110','130','150','170','190'})
end



%% Testing Diego's function
% clear; clc;
% rng('default')
% % X = design matrix 
% % y = targets
% % epsilon = ridge regression penalty
% %[yhat_star,yhat,w,beta] = stack_regress_BG(X,y,epsilon)
% 
% X_new = cell(2,1);
% load carsmall
% x1 = Weight;
% x2 = Horsepower;    % Contains NaN data
% 
% % Form data
% X{1} = [x1 x2]; %Population of states
% y = MPG; %Accidents per state
% 
% % Remove NaNs
% idxNaN = sum([isnan(X{1}) isnan(y)],2)>0;
% X_new{1} = X{1}(~idxNaN,:);
% X_new{2} = X{1}(~idxNaN,:)+5;
% y = y(~idxNaN);
% 
% epsilon = 0.001;
% [yhat_star,yhat,w,beta] = stack_regress_BG(X_new,y,epsilon)
% 
% % yhat = predictions for each learner
% % w = weights to stack the learners
% % yhat_star = stacked prediction
% % beta = the weights used for each learner to combine all the features from
% % the design matrix
% 
% yhat_star == yhat_star2
% 







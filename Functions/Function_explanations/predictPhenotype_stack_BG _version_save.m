function [predictedY,predictedY_ST,W,W_max]...
    = predictPhenotype_stack_BG (Yin,Din,options,varargin)
%we use vargin so that specifying correlation structure and confounds is
%optional

% function [predictedY,predictedYD,YD,stats,...
%     predictedY_max,predictedYD_max,stats_max,W,W_max]...
%     = predictPhenotype_stack (Yin,Din,options,varargin)
%
% Kernel ridge regression or nearest-neighbour estimation using
% a distance matrix using (stratified) LOO.
% Using this means that the HMM was run once, out of the cross-validation loop
%
% INPUT
% Yin       (no. subjects by 1) vector of phenotypic values to predict,
%           which can be continuous or binary. If a multiclass variable
%           is to be predicted, then Yin should be encoded by a
%           (no. subjects by no. classes) matrix, with zeros or ones
%           indicator entries.
% Din       (no. subjects by no. subjects) matrix of distances between
%           subjects, calculated (for example) by computeDistMatrix or
%           computeDistMatrix_AVFC
% options   Struct with the prediction options, with fields:
%   + alpha - for method='KRR', a vector of weights on the L2 penalty on the regression
%           By default: [0.0001 0.001 0.01 0.1 0.4 0.7 1.0 10 100]
%   + sigmafact - for method='KRR', a vector of parameters for the kernel; in particular,
%           this is the factor by which we will multiply a data-driven estimation
%           of the kernel parameter. By default: [1/5 1/3 1/2 1 2 3 5];
%   + K - for method='NN', a vector with the number of nearest neighbours to use
%   + CVscheme - vector of two elements: first is number of folds for model evaluation;
%             second is number of folds for the model selection phase (0 in both for LOO)
%   + CVfolds - prespecified CV folds for the outer loop
%   + biascorrect - whether we correct for bias in the estimation
%                   (Smith et al. 2019, NeuroImage)
%   + verbose -  display progress?
% cs        optional (no. subjects X no. subjects) dependency structure matrix with
%           specifying possible relations between subjects (e.g., family
%           structure), or a (no. subjects X 1) vector defining some
%           grouping, with (1...no.groups) or 0 for no group
% confounds     (no. subjects by  no. of confounds) matrix of features that
%               potentially influence the phenotypes and we wish to control for
%               (optional)
%
% OUTPUT
% predictedY    predicted response,in the original (non-decounfounded) space
% predictedYD    predicted response,in the decounfounded space
% YD    response,in the decounfounded space
% stats         structure, with fields
%   + pval - permutation-based p-value, if permutation is run;
%            otherwise, correlation-based p-value
%   + cod - coeficient of determination
%   + corr - correlation between predicted and observed Y
%   + baseline_corr - baseline correlation between predicted and observed Y for null model
%   + sse - sum of squared errors
%   + baseline_sse - baseline sum of squared errors
%   PLUS: All of the above +'_deconf' in the deconfounded space, if counfounds were specified
%   + alpha - selected values for alpha at each CV fold
%   + sigmaf - selected values for sigmafact at each CV fold
%
% Author: Diego Vidaurre, Aarhus University

%rng('default') % set seed for reproducibility

M = length(Din); % M = number of repetitions of HMM to combine
for m = 1:M, Din{m}(eye(size(Din,1))==1) = 0; end % set the diagonal elements of the distance matrices to 0

[N,q] = size(Yin);
% N = number of subjects
% q = number of classes (features to predict I think?)

% remove subjects with NaN values
which_nan = false(N,1);
if q == 1
    which_nan = isnan(Yin);
    if any(which_nan)
        Yin = Yin(~which_nan);
        for m = 1:M, Din{m} = Din{m}(~which_nan,~which_nan); end
        warning('NaN found on Yin, will remove...')
    end
    N = size(Yin,1); % number of subjects without NaNs
end

% if options not set by user, set the following L2 penalty values
if nargin < 3 || isempty(options), options = struct(); end
% if ~isfield(options,'method')
%     if isfield(options,'K')
%         method = 'K';
%     else
%         method = 'KRR';
%     end
% else
%     method = 'KRR';
% end
if ~isfield(options,'alpha')
    alpha = [0.0001 0.001 0.01 0.1 0.4 0.7 1.0 10 100];
    % alpha = a vector of weights on the L2 penalty on the kernel ridge regression
else
    alpha = options.alpha;
end

% if ~isfield(options,'lambda') % lambda is not used
%     lambda = 0.0001;
% else
%     lambda = options.lambda;
% end
if ~isfield(options,'sigmafact')
    sigmafact = [1/5 1/3 1/2 1 2 3 5]; % a vector of parameters for the kernel (in KRR)
else
    sigmafact = options.sigmafact;
end
% if ~isfield(options,'K')
%     K = 1:min(50,round(0.5*N)); % if KNN used, value of K (e.g. 50 nearest neighbours)
% else
%     K = options.K;
% end

if ~isfield(options,'CVscheme'), CVscheme = [10 10]; % 10-fold CV scheme (first element is for model evaluation and second element is for model selection)
else, CVscheme = options.CVscheme; end
if ~isfield(options,'CVfolds'), CVfolds = [];
else, CVfolds = options.CVfolds; end
% if ~isfield(options,'biascorrect'), biascorrect = 0;
% else, biascorrect = options.biascorrect; end
if ~isfield(options,'verbose'), verbose = 1; % display progress
else, verbose = options.verbose; end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CHECK CORRELATION STRUCTURE
% I think this basically makes sures families stay together, so in varargin
% there would be some defined family correlation structure, then this
% checks the structure and stores it in allcs (for us, this is twins!)
allcs = [];
if (nargin>3) && ~isempty(varargin{1}) % if correlation structure is defined (it is defined in nargin = 4 and then we check it's not empty)
    cs = varargin{1}; % let cs be the pre-defined correlation structure
    if ~isempty(cs)
        is_cs_matrix = (size(cs,2) == size(cs,1)); % 1 if matrix, 0 if vector
        if is_cs_matrix % if it's a square matrix then do this...
            if any(which_nan) % if there are any NaN values...
                cs = cs(~which_nan,~which_nan); % remove subjects with NaN values
            end
            % Now we find indices of positive values of correlation structure
            % [length(cs) length(cs)] is a n_subjects x n_subjects array
            % so the goal is to get the subscripts of cs where it is
            % greater than 0
            [allcs(:,2),allcs(:,1)]=ind2sub([length(cs) length(cs)],find(cs>0));
            % e.g. output , allcs = [4 83; 5 24; 16 61;....; 24 5; 61 16; 83 4]
            % this means that subjects 4 and 83 are related, 5 and 24 are related, and 16 adn 61 are related
            % notice the symmetry because if subjects 4 and 83 are related,
            % then 83 and 4 are related
        else %else if it's not a square matrix (then it's a vector?)
            if any(which_nan)
                cs = cs(~which_nan); % remove subjects with NaN values (from vector rather than matrix)
            end
            allcs = find(cs > 0); % find positive values of the correlation structure
        end
    end
else, cs = [];
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% get confounds
if (nargin>4) && ~isempty(varargin{2})
    confounds = varargin{2};
    confounds = confounds - repmat(mean(confounds),N,1);
    deconfounding = 1;
    if any(which_nan)
        confounds = confounds(~which_nan,:);
    end
else
    confounds = []; deconfounding = 0;
end

% Initialise variables
Ymean = zeros(N,q);
YD = zeros(N,q); % deconfounded signal
YmeanD = zeros(N,q); % mean in deconfounded space
predictedY = zeros(N,q); predictedY_max = zeros(N,q); % To store the final predictions
predictedY_ST = zeros(N,M,q);
% if deconfounding, predictedYD = zeros(N,q); end
predictedYD_max = zeros(N,q);

% create the inner CV structure - stratified for family=multinomial
% i.e. if there are families we want to make sure are kept together, then
% we create a stratified inner CV structure (stratified just means we
% dictate some sort of structure onto the data e.g. keep families together)
% CVscheme = [10 10] means 10-fold CV for hyperparameter selection and
% 10-fold CV for model evaluation
if isempty(CVfolds)
    if CVscheme(1)==1
        folds = {1:N};
        % CVscheme = [1 1] for LOO CV (hence why the folds just become each individual subject)
    elseif q == 1
        Yin_copy = Yin; Yin_copy(isnan(Yin)) = realmax;
        % Realmax returns the largest finite floating-point number in IEEE® double precision.
        folds = cvfolds(Yin_copy,CVscheme(1),allcs);
    else % no stratification
        folds = cvfolds(randn(size(Yin,1),1),CVscheme(1),allcs);
        % CVscheme(1) = 10 (i.e. do 10 folds)
        % cvfolds is a Diego function that creates a series of K cells (if
        % K-fold CV) and each cell has one of the folds of data
    end
else
    folds = CVfolds;
end

% Intialise weights of stacking (a weight for each repetition of HMM and each CV)
W = zeros(M,length(folds));
W_max = zeros(M,length(folds));


for ifold = 1:length(folds)
   
    if verbose, fprintf('CV iteration %d \n',ifold); end
   
    J = folds{ifold}; % get the indices of the test data
    if isempty(J), continue; end % if j is empty that skip the entire ifold for loop
    if length(folds)==1 % if there's only 1 fold then let the training data be the same as the test data
       ji = J;
    else
        ji = setdiff(1:N,J); % get the indices of the training data (i.e. all indices except those in J)
    end
   
    D = cell(M,1); D2 = cell(M,1);
    for m = 1:M, D{m} = Din{m}(ji,ji); D2{m} = Din{m}(J,ji); end
    Y = Yin(ji,:); % Note training feature values
   
    % family structure for this fold
    Qallcs=[];
    if (~isempty(cs))
        if is_cs_matrix
            [Qallcs(:,2),Qallcs(:,1)] = ...
                ind2sub([length(cs(ji,ji)) length(cs(ji,ji))],find(cs(ji,ji)>0));
            %ind2sub converts linear indices to subscripts
        else
            Qallcs = find(cs(ji) > 0);
        end
    end
       
    
    for ii = 1%:q % for each feature that we want to predict
        
        % remove the subjects with NaN values for specified feature (from the actual feature values that we are trying to predict)
         ind = find(~isnan(Y(:,ii))); 
         Yii = Y(ind,ii);

       
        QDin = cell(M,1);
        for m = 1:M, QDin{m} = D{m}(ind,ind); end % get the distance matrices of the subjects that didn't have NaN values for the selected variable
        QN = length(ind); % Note the remaining number of subjects after removing NaNs

        % I've already removed NaNs so don't need to remove using above code
        Yii = Y;
        QDin = D;
        QN = length(Yii);
       
        Qfolds = cvfolds(Yii,CVscheme(2),Qallcs); % we stratify

%         % deconfounding business
%         if deconfounding
%             Cii = confounds(ji,:); Cii = Cii(ind,:);
%             [betaY,interceptY,Yii] = deconfoundPhen(Yii,Cii);
%         end
%        
        % centering response
        my = mean(Yii);
        Yii = Yii - repmat(my,size(Yii,1),1);
        Ymean(J,ii) = my;
        QYin = Yii;
       
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Determining sigma and alpha
        Dev = Inf(length(alpha),length(sigmafact),M);
       
        % inner loop for parameters sigma and alpha
        for isigm = 1:length(sigmafact)
            %   sigmafact - for method='KRR', a vector of parameters for the kernel
            %   sigmafact = [1/5 1/3 1/2 1 2 3 5];
            %   we test each of our sigmafact (via a cross-validation scheme) to choose the best
         
            sigmf = sigmafact(isigm);
            QpredictedY = Inf(QN,length(alpha),M);
           
            % Inner CV loop
            for Qifold = 1:length(Qfolds)
               
                QJ = Qfolds{Qifold}; % Get the indices of the test data
                Qji=setdiff(1:QN,QJ); % Get the indices of the training data
                QY = QYin(Qji,:);  % Get the feature values of the training data
                Qmy = mean(QY); QY=QY-Qmy; % Centre the variables
                Nji = length(Qji); % Note the number of subjects we are training the data on
               
                for m = 1:M % for each repetition of HMM / model to combine
   
                    QD = QDin{m}(Qji,Qji); % Note the distance matrices of the subjects in the training data set
                    QD2 = QDin{m}(QJ,Qji); % Note the distance matrices of the subjects in the test data set
                   
                    sigmabase = auto_sigma(QD);
                    sigma = sigmf * sigmabase;
                   
                    K = gauss_kernel(QD,sigma); % Radial basis kernel function
                    K2 = gauss_kernel(QD2,sigma);
                    I = eye(Nji);
                    ridg_pen_scale = mean(diag(K));
                   
                    %   alpha - for method='KRR', a vector of weights on the L2 penalty on the regression
                    %   alpha = [0.0001 0.001 0.01 0.1 0.4 0.7 1.0 10 100]
                    %   we test each of our alpha (via a cross-validation scheme) to choose the best
                    for ialph = 1:length(alpha)
                        alph = alpha(ialph);
                        beta = (K + ridg_pen_scale * alph * I) \ QY;
                        % From NeuroImage 2021 paper:
                        % K = H
                        % QY = h
                        % alpha (proportional to?) lambda (weights on the L2 penalty on the regression)
                        % Then beta = the symbol alpha in the paper
                        % (confusingly)
                        QpredictedY(QJ,ialph,m) = K2 * beta + repmat(Qmy,length(QJ),1);
                    end
                   
                end
            end
           
            for m = 1:M % for each repetition of the HMM
                % Deviance  is a goodness-of-fit statistic for a statistical model 
                % It is a generalization of the idea of using the sum of squares of residuals (RSS) in ordinary 
                % least squares to cases where model-fitting is achieved by maximum likelihood.
                Dev(:,isigm,m) = (sum(( QpredictedY(:,:,m) - ...
                    repmat(QYin,1,length(alpha))).^2) / QN)';
            end
           
        end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%       

        % Now we have test all the alpha and sigma values, we need to
        % choose the best, we choose the ones with the lowest deviance...
        sigma = zeros(1,M); %alph = zeros(1,M);
        for m = 1:M % choose the best alpha and sigma for each repetition of the HMM
            dev = Dev(:,:,m);
            [~,mindev] = min(dev(:)); % Get the indices of the lowest deviation
            [ialph,isigm] = ind2sub(size(Dev),mindev);
            sigmf = sigmafact(isigm);
            sigmabase = auto_sigma(D{m});
            sigma(m) = sigmf * sigmabase;
            alph(m) = alpha(ialph);
        end
        % Now we have chosen the best alpha's and sigma's!

       
        % inner loop for stacking estimation
        QpredictedY = Inf(QN,M); % we will make a prediction for each subject and each repetition of the HMM
        for Qifold = 1:length(Qfolds)

            QJ = Qfolds{Qifold}; % Get the indices of the test data
            Qji = setdiff(1:QN,QJ); % Get the indices of the training data - 1:QN is all the subjects and then we remove the test data
            QY = QYin(Qji,:); % Get the feature values of the training data
            Qmy = mean(QY); QY = QY-Qmy; % Centre the variables
            Nji = length(Qji); % Note the number of subjects we are training the data on

            for m = 1:M % for each repetition of the HMM we get new predictions

                QD = QDin{m}(Qji,Qji); % Get the distances between subjects in the training data
                QD2 = QDin{m}(QJ,Qji); % Get the distances between subjects in the test data and subjects in the training data

                % Find the best model parameters using KRR
                K = gauss_kernel(QD,sigma(m));
                K2 = gauss_kernel(QD2,sigma(m));
                I = eye(Nji);
                ridg_pen_scale = mean(diag(K));
                beta = (K + ridg_pen_scale * alph(m) * I) \ QY;

                % We now use the beta's to make the predictions (then add
                % the mean back on to reverse the centering of the
                % variables)
                QpredictedY(QJ,m) = K2 * beta + repmat(Qmy,length(QJ),1);
            end
        end
        
        % Okay so at this point we have basically just done
        % 'predictPhenotype' 5 times, once for each of our HMM repetitions
        % (which we got using CV, so we can use all the data to train our
        % meta-model when we stack our predictions).
        % Next, we combine them using stacked regression
        % Note that we are still in the outer CV loop, so all this is still
        % using 90% of the data to predict the last 10%, then repeated.

        %w = (QpredictedY' * QpredictedY + lambda * eye(M)) \ (QpredictedY' * QYin);
        opts1 = optimset('display','off');
        % find the stacking weights (for this particular feature) for the
        % repetitions of HMM
        w = lsqlin(QpredictedY,QYin,[],[],ones(1,M),1,zeros(M,1),ones(M,1),[],opts1);
        W(:,ifold) = w;
       
        % This bit basically finds the best predictor (of the e.g. 5
        % repetitions of the HMM)
        [~,best] = min(sum((QpredictedY-QYin).^2));
        W_max(best,ifold) = 1;
        

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Now we make the actual predictions for each of our HMM repetitions that
% we will then use to combine into a new prediction using the weights.
% We do this on all the data.
        for m = 1:M % for each repetition of the HMM
             Dii = D{m}(ind,ind); D2ii = D2{m}(:,ind); 
             % here we removed subjects with NaN (I've already done this outside the fct)


            K = gauss_kernel(Dii,sigma(m));
            K2 = gauss_kernel(D2ii,sigma(m));
            
            % I've already removed NaNs so ignore this code
            Nji = length(ind);
            I = eye(Nji);

            ridg_pen_scale = mean(diag(K)); % This always seems to be 1 currently?
            beta = (K + ridg_pen_scale * alph(m) * I) \ Yii;

            % Get the predictions of each HMM repetition (FOR THE OUTER
            % LOOP TEST DATA)
            predictedY_ST(J,m,ii) = K2 * beta + my;
        end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % predict the test fold (by combining the prediction of each HMM
        % repetition with the respective weights)
        % Note that this is done within the outer cross-validation loop
        % because we are trying to determine how good our method is (rather
        % than actually make new predictions). So we are using 10-fold CV
        % to check how accurate the model is. If we then, in the future,
        % received new data f  rom HCP and wanted to make predictions, we
        % would retrain the model on ALL data, rather than using 10-fold
        % CV, as we have done here.
        predictedY(J,ii) = predictedY_ST(J,:,ii) * w; % MULTIPLY THE TEST PREDICTIONS WITH THE BEST WEIGHTS ACROSS THE TRAINING DATA
        % Here we note the predictions using the single best repetition of the HMM to compare vs stacked predictions
        %predictedY_max(J,ii) = predictedY_ST(J,best,ii)
        

%         % predictedYD and YD in deconfounded space; Yin and predictedY are confounded
%         predictedYD(J,ii) = predictedY(J,ii);
%         predictedYD_max(J,ii) = predictedY_max(J,ii);
%         YD(J,ii) = Yin(J,ii);
%         YmeanD(J,ii) = Ymean(J,ii);
%         if deconfounding % in order to later estimate prediction accuracy in deconfounded space
%             [~,~,YD(J,ii)] = deconfoundPhen(YD(J,ii),confounds(J,:),betaY,interceptY);
%             % original space
%             predictedY(J,ii) = confoundPhen(predictedY(J,ii),confounds(J,:),betaY,interceptY);
%             predictedY_max(J,ii) = confoundPhen(predictedY_max(J,ii),confounds(J,:),betaY,interceptY);
%             Ymean(J,ii) = confoundPhen(YmeanD(J,ii),confounds(J,:),betaY,interceptY);
%         end
   
    end
   
    disp(['Fold ' num2str(ifold) ])
   
end

% % the below is all just some stats we can take away from this stacking
% % prediction if we want!
% 
% stats = struct();
% stats.sse = zeros(q,1);
% stats.cod = zeros(q,1);
% stats.corr = zeros(q,1);
% stats.baseline_corr = zeros(q,1);
% stats.pval = zeros(q,1);
% if deconfounding
%     stats.sse_deconf = zeros(q,1);
%     stats.cod_deconf = zeros(q,1);
%     stats.corr_deconf = zeros(q,1);
%     stats.baseline_corr_deconf = zeros(q,1);
%     stats.pval_deconf = zeros(q,1);
% end
% stats_max = stats;
% 
% for ii = 1:q
%     ind = find(~isnan(Yin(:,ii)));
%     stats.sse(ii) = sum((Yin(ind,ii)-predictedY(ind,ii)).^2);
%     nullsse = sum((Yin(ind,ii)-Ymean(ind,ii)).^2);
%     stats.cod(ii) = 1 - stats.sse(ii) / nullsse;
%     stats.corr(ii) = corr(Yin(ind,ii),predictedY(ind,ii));
%     stats.baseline_corr(ii) = corr(Yin(ind,ii),Ymean(ind,ii));
%     [~,pv] = corrcoef(Yin(ind,ii),predictedY(ind,ii)); % original space
%     if corr(Yin(ind,ii),predictedY(ind,ii))<0, stats.pval(ii) = 1;
%     else, stats.pval(ii) = pv(1,2);
%     end
%     if deconfounding
%         stats.sse_deconf(ii) = sum((YD(ind,ii)-predictedYD(ind,ii)).^2);
%         nullsse_deconf = sum((YD(ind,ii)-YmeanD(ind,ii)).^2);
%         stats.cod_deconf(ii) = 1 - stats.sse_deconf(ii) / nullsse_deconf;
%         stats.corr_deconf(ii) = corr(YD(ind,ii),predictedYD(ind,ii));
%         stats.baseline_corr_deconf(ii) = corr(YD(ind,ii),YmeanD(ind,ii));
%         [~,pv] = corrcoef(YD(ind,ii),predictedYD(ind,ii)); % original space
%         if corr(YD(ind,ii),predictedYD(ind,ii))<0, stats.pval_deconf(ii) = 1;
%         else, stats.pval_deconf(ii) = pv(1,2);
%         end
%     end
%     stats_max.sse(ii) = sum((Yin(ind,ii)-predictedY_max(ind,ii)).^2);
%     nullsse = sum((Yin(ind,ii)-Ymean(ind,ii)).^2);
%     stats_max.cod(ii) = 1 - stats_max.sse(ii) / nullsse;
%     stats_max.corr(ii) = corr(Yin(ind,ii),predictedY_max(ind,ii));
%     stats_max.baseline_corr(ii) = corr(Yin(ind,ii),Ymean(ind,ii));
%     [~,pv] = corrcoef(Yin(ind,ii),predictedY_max(ind,ii)); % original space
%     if corr(Yin(ind,ii),predictedY_max(ind,ii))<0, stats_max.pval(ii) = 1;
%     else, stats_max.pval(ii) = pv(1,2);
%     end
%     if deconfounding
%         stats_max.sse_deconf(ii) = sum((YD(ind,ii)-predictedYD_max(ind,ii)).^2);
%         nullsse_deconf = sum((YD(ind,ii)-YmeanD(ind,ii)).^2);
%         stats_max.cod_deconf(ii) = 1 - stats_max.sse_deconf(ii) / nullsse_deconf;
%         stats_max.corr_deconf(ii) = corr(YD(ind,ii),predictedYD_max(ind,ii));
%         stats_max.baseline_corr_deconf(ii) = corr(YD(ind,ii),YmeanD(ind,ii));
%         [~,pv] = corrcoef(YD(ind,ii),predictedYD_max(ind,ii)); % original space
%         if corr(YD(ind,ii),predictedYD_max(ind,ii))<0, stats_max.pval_deconf(ii) = 1;
%         else, stats_max.pval_deconf(ii) = pv(1,2);
%         end
%     end
%    
% end

end



function K = gauss_kernel(D,sigma)
% Gaussian kernel
D = D.^2; % because distance is sqrt-ed
K = exp(-D/(2*sigma^2));
end


function sigma = auto_sigma (D)
% gets a data-driven estimation of the kernel parameter
D = D(triu(true(size(D,1)),1)); 
% The triu(xxx) creates a DxD matrix where the elements above the main diagonal are 1 and everything else is 0.
% Therefore, this keeps only the upper triangular portion of D (excluding the main diagonal)
sigma = median(D);
end


function [betaY,my,Y] = deconfoundPhen(Y,confX,betaY,my)
if nargin<3, betaY = []; end
if isempty(betaY)
    my = mean(Y);
    Y = Y - my;
    betaY = (confX' * confX + 0.00001 * eye(size(confX,2))) \ confX' * Y;
end
res = Y - confX*betaY;
Y = res;
end


function Y = confoundPhen(Y,conf,betaY,my)
Y = Y+conf*betaY+my;
end
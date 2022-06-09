function [predictedY,predictedYD,YD,stats,...
    predictedY_max,predictedYD_max,stats_max,W,W_max,predictedY_ST]...
    = predictPhenotype_stack (Yin,Din,options,varargin)
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

rng('default') % set seed for reproducibility

M = length(Din);
for m = 1:M, Din{m}(eye(size(Din,1))==1) = 0; end

[N,q] = size(Yin);

which_nan = false(N,1);
if q == 1
    which_nan = isnan(Yin);
    if any(which_nan)
        Yin = Yin(~which_nan);
        for m = 1:M, Din{m} = Din{m}(~which_nan,~which_nan); end
        warning('NaN found on Yin, will remove...')
    end
    N = size(Yin,1);
end

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
else
    alpha = options.alpha;
end

if ~isfield(options,'lambda')
    lambda = 0.0001;
else
    lambda = options.lambda;
end
if ~isfield(options,'sigmafact')
    sigmafact = [1/5 1/3 1/2 1 2 3 5];
else
    sigmafact = options.sigmafact;
end
if ~isfield(options,'K')
    K = 1:min(50,round(0.5*N));
else
    K = options.K;
end

if ~isfield(options,'CVscheme'), CVscheme = [10 10];
else, CVscheme = options.CVscheme; end
if ~isfield(options,'CVfolds'), CVfolds = [];
else, CVfolds = options.CVfolds; end
% if ~isfield(options,'biascorrect'), biascorrect = 0;
% else, biascorrect = options.biascorrect; end
if ~isfield(options,'verbose'), verbose = 1;
else, verbose = options.verbose; end

% check correlation structure
allcs = [];
if (nargin>3) && ~isempty(varargin{1})
    cs = varargin{1};
    if ~isempty(cs)
        is_cs_matrix = (size(cs,2) == size(cs,1));
        if is_cs_matrix
            if any(which_nan)
                cs = cs(~which_nan,~which_nan);
            end
            [allcs(:,2),allcs(:,1)]=ind2sub([length(cs) length(cs)],find(cs>0));    
        else
            if any(which_nan)
                cs = cs(~which_nan);
            end
            allcs = find(cs > 0);
        end
    end
else, cs = [];
end

% % get confounds
% if (nargin>4) && ~isempty(varargin{2})
%     confounds = varargin{2};
%     confounds = confounds - repmat(mean(confounds),N,1);
%     deconfounding = 1;
%     if any(which_nan)
%         confounds = confounds(~which_nan,:);
%     end
% else
%     confounds = []; deconfounding = 0;
% end

Ymean = zeros(N,q);
YD = zeros(N,q); % deconfounded signal
YmeanD = zeros(N,q); % mean in deconfounded space
predictedY = zeros(N,q); predictedY_max = zeros(N,q); predictedY_ST = zeros(N,M,q);
% if deconfounding, predictedYD = zeros(N,q); predictedYD_max = zeros(N,q); end
predictedYD_max = zeros(N,q);

% create the inner CV structure - stratified for family=multinomial
if isempty(CVfolds)
    if CVscheme(1)==1
        folds = {1:N};
    elseif q == 1
        Yin_copy = Yin; Yin_copy(isnan(Yin)) = realmax;
        folds = cvfolds(Yin_copy,CVscheme(1),allcs);
    else % no stratification
        folds = cvfolds(randn(size(Yin,1),1),CVscheme(1),allcs);
    end
else
    folds = CVfolds;
end

W = zeros(M,length(folds));
W_max = zeros(M,length(folds));


for ifold = 1%:length(folds)
   
    if verbose, fprintf('CV iteration %d \n',ifold); end
   
    J = folds{ifold}; % test
    if isempty(J), continue; end
    if length(folds)==1
        ji = J;
    else
        ji = setdiff(1:N,J); % train
    end
   
    D = cell(M,1); D2 = cell(M,1);
    for m = 1:M, D{m} = Din{m}(ji,ji); D2{m} = Din{m}(J,ji); end
    Y = Yin(ji,:);
   
    % family structure for this fold
    Qallcs=[];
    if (~isempty(cs))
        if is_cs_matrix
            [Qallcs(:,2),Qallcs(:,1)] = ...
                ind2sub([length(cs(ji,ji)) length(cs(ji,ji))],find(cs(ji,ji)>0));
        else
            Qallcs = find(cs(ji) > 0);
        end
    end
       
    for ii = 1:q
               
        ind = find(~isnan(Y(:,ii)));
        Yii = Y(ind,ii);
        %mean_Y = mean(Y)
        %mean_Yii = mean(Yii)
        
        QDin = cell(M,1);
        for m = 1:M, QDin{m} = D{m}(ind,ind); end
        QN = length(ind);
       
        Qfolds = cvfolds(Yii,CVscheme(2),Qallcs); % we stratify

        % deconfounding business
%         if deconfounding
%             Cii = confounds(ji,:); Cii = Cii(ind,:);
%             [betaY,interceptY,Yii] = deconfoundPhen(Yii,Cii);
%         end
       
        % centering response
        my = mean(Yii);
        Yii = Yii - repmat(my,size(Yii,1),1);
        %mean_Yii = mean(Yii)
        Ymean(J,ii) = my;
        QYin = Yii;
        %mean_QYin = mean(QYin)
        Dev = Inf(length(alpha),length(sigmafact),M);
       
        % inner loop for parameters sigma and alpha
        for isigm = 1:length(sigmafact)
           
            sigmf = sigmafact(isigm);
            QpredictedY = Inf(QN,length(alpha),M);
           
            % Inner CV loop
            for Qifold = 1:length(Qfolds)
               
                QJ = Qfolds{Qifold}; Qji=setdiff(1:QN,QJ);
                
                QY = QYin(Qji,:); %training data
                %mean_QY = mean(QY)
                Qmy = mean(QY); QY=QY-Qmy;
                %mean_QY_2 = mean(QY)
                Nji = length(Qji);
                
               
                for m = 1:M
   
                    QD = QDin{m}(Qji,Qji); % training data
                    QD2 = QDin{m}(QJ,Qji); % test data
                   
                    sigmabase = auto_sigma(QD);
                    sigma = sigmf * sigmabase;
                   
                    K = gauss_kernel(QD,sigma); % training kernel
                    K2 = gauss_kernel(QD2,sigma); % test kernel
                    I = eye(Nji);
                    ridg_pen_scale = mean(diag(K));
                   
                    for ialph = 1:length(alpha)
                        alph = alpha(ialph);
                        beta = (K + ridg_pen_scale * alph * I) \ QY; % determine weights using training data
                        QpredictedY(QJ,ialph,m) = K2 * beta + repmat(Qmy,length(QJ),1); % make predictions for test data
                    end
                   
                end
            end
            isigm
            size(QpredictedY)
            size(QYin)
            for i = 1:5
                mean_QpredictedY = mean(QpredictedY(:,:,i))
                mean_QYin = mean(QYin)
                QpredictedY(1:10,:,i)
                
            end
           
            for m = 1:M
                Dev(:,isigm,m) = (sum(( QpredictedY(:,:,m) - ...
                    repmat(QYin,1,length(alpha))).^2) / QN)';
            end
           
        end
       
        % choose the sigma and alpha with the lowest deviance
        sigma = zeros(1,M); alph = zeros(1,M);
        for m = 1:M
            dev = Dev(:,:,m)
            [~,mindev] = min(dev(:));
            [ialph,isigm] = ind2sub(size(Dev),mindev);
            sigmf = sigmafact(isigm);
            best_sigma = sigmf
            sigmabase = auto_sigma(D{m});
            sigma(m) = sigmf * sigmabase;
            alph(m) = alpha(ialph);
        end
        
       
        % inner loop for stacking estimation
        QpredictedY = Inf(QN,M);
        for Qifold = 1:length(Qfolds)
            QJ = Qfolds{Qifold}; Qji = setdiff(1:QN,QJ);
            QY = QYin(Qji,:); Qmy = mean(QY);
            QY = QY-Qmy;
            Nji = length(Qji);
            for m = 1:M
                QD = QDin{m}(Qji,Qji);
                QD2 = QDin{m}(QJ,Qji);
                K = gauss_kernel(QD,sigma(m));
                K2 = gauss_kernel(QD2,sigma(m));
                I = eye(Nji);
                ridg_pen_scale = mean(diag(K));
                beta = (K + ridg_pen_scale * alph(m) * I) \ QY;
                QpredictedY(QJ,m) = K2 * beta + repmat(Qmy,length(QJ),1);%Qmy;
            end
        end
 
              
        for m = 1:M
            Dii = D{m}(ind,ind); D2ii = D2{m}(:,ind);
            K = gauss_kernel(Dii,sigma(m));
            K2 = gauss_kernel(D2ii,sigma(m));
            Nji = length(ind);
            I = eye(Nji);
            ridg_pen_scale = mean(diag(K));
            beta = (K + ridg_pen_scale * alph(m) * I) \ Yii;
            predictedY_ST(J,m,ii) = K2 * beta + my;
        end
        
        
        
        lambda = 0.001;
        
        %QpredictedY = [ones(size(QpredictedY,1),1) QpredictedY];
        %size(QpredictedY)
        %size(QYin)
        w = (QpredictedY' * QpredictedY + lambda * eye(M)) \ (QpredictedY' * QYin);
        %opts1 = optimset('display','off');
        %w = lsqlin(QpredictedY,QYin,[],[],ones(1,M),1,zeros(M,1),ones(M,1),[],opts1);
        W(:,ifold) = w;
        

        
        
        % predict the test fold
        predictedY(J,ii) = predictedY_ST(J,:,ii) * w;
        %predictedY(J,ii) = predictedY_ST(J,:,ii) * w(2:end);
        %predictedY_max(J,ii) = predictedY_ST(J,best,ii);
        
        mean_predictors = mean(nonzeros(predictedY_ST(J,m,ii)))
        mean_predictions = mean(predictedY(J,ii))
        
        mean_predictors - mean_predictions
        predictedY(J,ii) = predictedY(J,ii) + (mean_predictors - mean_predictions);

       
        predictedYD = [];
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
 
stats = struct();
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
stats_max = stats;
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
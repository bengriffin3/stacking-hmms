function [predictedY_ST, predictedY_stack,predictedY_stack_static,predictedY_FWLS,predictedY_FWLS_norm,predictedY_ridge,W_stack,W_stack_static,W_FWLS,W_FWLS_norm,W_ridge]...
    = predictPhenotype_stack_meta(Yin,Din_HMM,Din_static,metafeatures,options,varargin)
%%%%%%% Metafeature code %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% lines 194,310,324,349


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


M = length(Din_HMM);
for m = 1:M, Din_HMM{m}(eye(size(Din_HMM,1))==1) = 0; end
Din_static(eye(size(Din_static,1))==1) = 0;

[N,q] = size(Yin);

which_nan = false(N,1);
if q == 1
    which_nan = isnan(Yin);
    if any(which_nan)
        Yin = Yin(~which_nan);
        for m = 1:M, Din_HMM{m} = Din_HMM{m}(~which_nan,~which_nan); end
        Din_static = Din_static(~which_nan,~which_nan);
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

Ymean = zeros(N,q);
YD = zeros(N,q); % deconfounded signal
YmeanD = zeros(N,q); % mean in deconfounded space
predictedY = zeros(N,q); predictedY_max = zeros(N,q);
predictedY_ST = zeros(N,M,q);

predictedY_stack = zeros(N,q); predictedY_ST_static = zeros(N,q); predictedY_stack_static = zeros(N,q);
predictedY_FWLS = zeros(N,q); predictedY_FWLS_norm = zeros(N,q);
predictedY_ridge = zeros(N,q);

if deconfounding, predictedYD = zeros(N,q); predictedYD_max = zeros(N,q); end

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

n_metafeatures = size(metafeatures,2)/5;
W_stack = zeros(M,length(folds));
W_stack_static = zeros(M+1,length(folds));
W_FWLS = zeros(M*n_metafeatures,length(folds));
W_FWLS_norm = zeros(M*n_metafeatures,length(folds));
W_ridge = zeros(M,length(folds));


lambda = 0:0.1:1; %%% regularization parameter

for ifold = 1:length(folds)
   
    if verbose, fprintf('CV iteration %d \n',ifold); end
   
    J = folds{ifold}; % test
    if isempty(J), continue; end
    if length(folds)==1
        ji = J;
    else
        ji = setdiff(1:N,J); % train
    end
   
    D = cell(M,1); D2 = cell(M,1);
    for m = 1:M, D{m} = Din_HMM{m}(ji,ji); D2{m} = Din_HMM{m}(J,ji); end
    D_static = Din_static(ji,ji); D2_static = Din_static(J,ji);
    Y = Yin(ji,:);

   
    meta_train = metafeatures(ji,:); %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    meta_test = metafeatures(J,:); %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   
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
       
        QDin = cell(M,1);
        for m = 1:M, QDin{m} = D{m}(ind,ind); end
        QDin_static = D_static(ind,ind);  
        QN = length(ind);
       
        Qfolds = cvfolds(Yii,CVscheme(2),Qallcs); % we stratify

        % deconfounding business
        if deconfounding
            Cii = confounds(ji,:); Cii = Cii(ind,:);
            [betaY,interceptY,Yii] = deconfoundPhen(Yii,Cii);
        end
       
        % centering response
        my = mean(Yii);
        Yii = Yii - repmat(my,size(Yii,1),1);
        Ymean(J,ii) = my;
        QYin = Yii;
       
%         Dev = Inf(length(alpha),length(sigmafact),M);
%        
%         % inner loop for parameters sigma and alpha
%         for isigm = 1:length(sigmafact)
%            
%             sigmf = sigmafact(isigm);
%             QpredictedY = Inf(QN,length(alpha),M);
%            
%             % Inner CV loop
%             for Qifold = 1:length(Qfolds)
%                
%                 QJ = Qfolds{Qifold}; Qji=setdiff(1:QN,QJ);
%                 QY = QYin(Qji,:); Qmy = mean(QY); QY=QY-Qmy;
%                 Nji = length(Qji);
%                
%                 for m = 1:M
%    
%                     QD = QDin{m}(Qji,Qji);
%                     QD2 = QDin{m}(QJ,Qji);
%                    
%                     sigmabase = auto_sigma(QD);
%                     sigma = sigmf * sigmabase;
%                    
%                     K = gauss_kernel(QD,sigma);
%                     K2 = gauss_kernel(QD2,sigma);
%                     I = eye(Nji);
%                     ridg_pen_scale = mean(diag(K));
%                    
%                     for ialph = 1:length(alpha)
%                         alph = alpha(ialph);
%                         beta = (K + ridg_pen_scale * alph * I) \ QY;
%                         QpredictedY(QJ,ialph,m) = K2 * beta + repmat(Qmy,length(QJ),1);
%                     end
%                    
%                 end
%             end
%            
%             for m = 1:M
%                 Dev(:,isigm,m) = (sum(( QpredictedY(:,:,m) - ...
%                     repmat(QYin,1,length(alpha))).^2) / QN)';
%             end
%            
%         end
%        
%         % choose the sigma and alpha with the lowest deviance
%         sigma = zeros(1,M); alph = zeros(1,M);
%         for m = 1:M
%             dev = Dev(:,:,m);
%             [~,mindev] = min(dev(:));
%             [ialph,isigm] = ind2sub(size(Dev),mindev);
%             sigmf = sigmafact(isigm);
%             sigmabase = auto_sigma(D{m});
%             sigma(m) = sigmf * sigmabase;
%             alph(m) = alpha(ialph);
%         end
       

        sigma = 1.9621e+04*ones(m,1);
        alph = [1 1 1 1 1];


        % inner loop for stacking estimation
        QpredictedY_HMM = Inf(QN,M);
        QpredictedY_static = Inf(QN,1);
        for Qifold = 1:length(Qfolds)
            QJ = Qfolds{Qifold}; Qji = setdiff(1:QN,QJ);
            QY = QYin(Qji,:); Qmy = mean(QY); QY = QY-Qmy;
            Nji = length(Qji);
            for m = 1:M
                % HMM predictions
                QD = QDin{m}(Qji,Qji);
                QD2 = QDin{m}(QJ,Qji);
                K = gauss_kernel(QD,sigma(m));
                K2 = gauss_kernel(QD2,sigma(m));
                I = eye(Nji);
                ridg_pen_scale = mean(diag(K));
                beta = (K + ridg_pen_scale * alph(m) * I) \ QY;
                QpredictedY_HMM(QJ,m) = K2 * beta + repmat(Qmy,length(QJ),1);

            end
            % NB: static and HMM predictions would use different alpha and
            % sigma, so if I want to do this for real I would have to
            % investigate this?
            
            % Static prediction (m will just equal 5, but since I've set alpha and
            % sigma currently, this is fine)...
            QD_static = QDin_static(Qji,Qji);
            QD2_static = QDin_static(QJ,Qji);
            K = gauss_kernel(QD_static,sigma(m));
            K2 = gauss_kernel(QD2_static,sigma(m));
            I = eye(Nji);
            ridg_pen_scale = mean(diag(K));
            beta = (K + ridg_pen_scale * alph(m) * I) \ QY;
            QpredictedY_static(QJ) = K2 * beta + repmat(Qmy,length(QJ),1);

        end
        
        % Matrix of metfeatures and predictions combined %%%%%%%%%%%%%%%%%%
        A_train = NaN(size(ji,2),n_metafeatures*M);
        A_train_norm = NaN(size(ji,2),n_metafeatures*M);
        for i = 1:n_metafeatures
              A_train(:,M*i-M+1:M*i) = meta_train(:,M*i-M+1:M*i).*QpredictedY_HMM;
              A_train_norm(:,M*i-M+1:M*i) = normalize(meta_train(:,M*i-M+1:M*i)).*normalize(QpredictedY_HMM);
        end
 
        %w = (QpredictedY' * QpredictedY + lambda * eye(M)) \ (QpredictedY' * QYin);
        opts1 = optimset('display','off');
        
        % Stacking weights
        w_stack = lsqlin(QpredictedY_HMM,QYin,[],[],ones(1,M),1,zeros(M,1),ones(M,1),[],opts1);
        W_stack(:,ifold) = w_stack;

        % Stacking weights with static and HMM repetitions
        w_stack_static = lsqlin([QpredictedY_HMM QpredictedY_static],QYin,[],[],ones(1,M+1),1,zeros(M+1,1),ones(M+1,1),[],opts1);

        % if we just want the weights for the 5 repetitions to sum to 1
        % (and not include the static prediction)
        %w_stack_static = lsqlin([QpredictedY_HMM QpredictedY_static],QYin,[],[],[ones(1,M) zeros(1,1),1,zeros(M+1,1),ones(M+1,1),[],opts1); 
        W_stack_static(:,ifold) = w_stack_static;



%%%%%%%%%%%%%%%% Stacking w/ metafeatures weights %%%%%%%%%%%%%%%%%%%%%%%%%
        %w_FWLS = lsqlin(A_train,QYin,[],[],ones(1,M*n_metafeatures),1,zeros(M*n_metafeatures,1),ones(M*n_metafeatures,1),[],opts1);
        
        % Let's try it where only the prediction weights have to sum to 1,
        % the other weights can be different, that way adding a zero
        % metafeature doesn't change the solution.
        w_FWLS = lsqlin(A_train,QYin,[],[],[ones(1,M) zeros(1,M*(n_metafeatures-1))],1,zeros(M*n_metafeatures,1),ones(M*n_metafeatures,1),[],opts1);

        % Let's try letting the weights for the metafeatures not be between 0 and 1
        % (weights for predictions still between 0 and 1)
        %w_FWLS = lsqlin(A_train,QYin,[],[],[ones(1,M) zeros(1,M*(n_metafeatures-1))],1,zeros(M,1),ones(M,1),[],opts1);


        % Let's stack without the sum to 1 constraint
        %w_FWLS = lsqlin(A_train,QYin,[],[],[],[],zeros(M*n_metafeatures,1),ones(M*n_metafeatures,1),[],opts1);

        % Let's stack without the sum to 1 or non-negativity constraint
        %w_FWLS = lsqlin(A_train,QYin,[],[],[],[],[],[],[],opts1);
        w_FWLS_norm = lsqlin(A_train,normalize(QYin),[],[],[],[],[],[],[],opts1);

        % let's test running the model just on the metafeatures and not on our predictions at all!
        %w_FWLS = lsqlin(meta_train,QYin,[],[],ones(1,M*n_metafeatures),1,zeros(M*n_metafeatures,1),ones(M*n_metafeatures,1),[],opts1);
        %w_FWLS = lsqlin(meta_train(:,6:10),QYin,[],[],ones(1,M*(n_metafeatures-1)),1,zeros(M*(n_metafeatures-1),1),ones(M*(n_metafeatures-1),1),[],opts1);
        
        
        W_FWLS(:,ifold) = w_FWLS;
        W_FWLS_norm(:,ifold) = w_FWLS_norm;

%%%%%%%%%%%%%%%% Stacking w/ ridge regression %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%         
%         inner loop for ridge regression regulatrization parameter
%         for Qifold = 1:length(Qfolds)
% 
%             QJ = Qfolds{Qifold}; % test fold
%             Qji=setdiff(1:QN,QJ); % training fold
%             QY = QYin(Qji,:); % training values (predictions)
%             Qmy = mean(QY); QY=QY-Qmy; % standardise training predictions
%             Nji = length(Qji);
% 
% 
%             QpredictedY_ridge_ST = Inf(QN,length(lambda),M);
%             QpredictedY_ridge = Inf(QN,length(lambda));
% 
%             for ilambda = 1:length(lambda)
%                 lamb = lambda(ilambda);
% 
%                 determine weights for stacking
%                 w_ridge_cv = ridge(QYin,QpredictedY_HMM,lamb);
% 
% 
%                 for m = 1:M
%                     QD = QDin{m}(Qji,Qji); QD2 = QDin{m}(QJ,Qji);
%                     K = gauss_kernel(QD,sigma(m));
%                     K2 = gauss_kernel(QD2,sigma(m));
%                     I = eye(Nji);
%                     ridg_pen_scale = mean(diag(K));
%                     beta = (K + ridg_pen_scale * alph(m) * I) \ QY;
%                     QpredictedY_ridge_ST(QJ,ilambda,m) = K2 * beta + Qmy;
%                 end
%   
%                 QpredictedY_ridge(QJ,ilambda,ii) = squeeze(QpredictedY_ridge_ST(QJ,ilambda,:)) * w_ridge_cv;
%             end
% 
%         end
%         explained_variance_ridge = corr(squeeze(QpredictedY_ridge),QYin).^2;
%         [M,I] = max(explained_variance_ridge);
%         lambda_best = lambda(I)
% 
%         lambda = 0:1e-5:5e-3;
%         lambda = 0.005;
%         w_ridge = ridge(QYin,QpredictedY_HMM,lambda_best);
%         W_ridge(:,ifold) = w_ridge;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       
        % Get predictions for 5 repetitions
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
        
        % predict test fold using 5 repetitions and combination weights
        predictedY_stack(J,ii) = predictedY_ST(J,:,ii) * w_stack;

        % stacked with static prediction
        Dii_static = D_static(ind,ind); D2ii_static = D2_static(:,ind);
        K = gauss_kernel(Dii_static,sigma(m));
        K2 = gauss_kernel(D2ii_static,sigma(m));
        Nji = length(ind);
        I = eye(Nji);
        ridg_pen_scale = mean(diag(K));
        beta = (K + ridg_pen_scale * alph(m) * I) \ Yii;
        predictedY_ST_static(J,ii) =  K2 * beta + my;

       
        % predict test fold (stacked + static)

        predictedY_stack_static(J,ii) = [predictedY_ST(J,:,ii) predictedY_ST_static(J,ii)] * w_stack_static;


        %%%% Predict test fold (w/ metafeatures) %%%%%%%%%%%%%%%%%%%%%%%%%%
        A_test = NaN(size(J,2),n_metafeatures*M);
        A_test_norm = NaN(size(J,2),n_metafeatures*M);
        for i = 1:n_metafeatures
            A_test(:,M*i-M+1:M*i) = meta_test(:,M*i-M+1:M*i).*predictedY_ST(J,:,ii);

            % we set all features that have std of 0 (same for all
            % subjects) to 1 (is this correct??)
            meta_test_norm = normalize(meta_test(:,M*i-M+1:M*i));
            meta_test_norm(isnan(normalize(meta_test(:,M*i-M+1:M*i)))) = 1;
            A_test_norm(:,M*i-M+1:M*i) = meta_test_norm.*normalize(predictedY_ST(J,:,ii));
        end
        predictedY_FWLS(J,ii) = A_test * w_FWLS;
        predictedY_FWLS_norm(J,ii) = A_test_norm * w_FWLS_norm;

        % Predict test fold using just the metafeatures
        %predictedY_FWLS(J,ii) = meta_test * w_FWLS;
        %predictedY_FWLS(J,ii) = meta_test(:,6:10) * w_FWLS;
        

%         % Predict test fold for ridge regression
%         predictedY_ridge(J,ii) = predictedY_ST(J,:,ii) * w_ridge;

        % predictedYD and YD in deconfounded space; Yin and predictedY are confounded
        predictedYD(J,ii) = predictedY(J,ii);
        predictedYD_max(J,ii) = predictedY_max(J,ii);
        YD(J,ii) = Yin(J,ii);
        YmeanD(J,ii) = Ymean(J,ii);
        if deconfounding % in order to later estimate prediction accuracy in deconfounded space
            [~,~,YD(J,ii)] = deconfoundPhen(YD(J,ii),confounds(J,:),betaY,interceptY);
            % original space
            predictedY(J,ii) = confoundPhen(predictedY(J,ii),confounds(J,:),betaY,interceptY);
            predictedY_max(J,ii) = confoundPhen(predictedY_max(J,ii),confounds(J,:),betaY,interceptY);
            Ymean(J,ii) = confoundPhen(YmeanD(J,ii),confounds(J,:),betaY,interceptY);
        end
   
    end
   
    disp(['Fold ' num2str(ifold) ])
   
end

stats = struct();
stats.sse = zeros(q,1);
stats.cod = zeros(q,1);
stats.corr = zeros(q,1);
stats.baseline_corr = zeros(q,1);
stats.pval = zeros(q,1);
if deconfounding
    stats.sse_deconf = zeros(q,1);
    stats.cod_deconf = zeros(q,1);
    stats.corr_deconf = zeros(q,1);
    stats.baseline_corr_deconf = zeros(q,1);
    stats.pval_deconf = zeros(q,1);
end
stats_max = stats;

for ii = 1:q
    ind = find(~isnan(Yin(:,ii)));
    stats.sse(ii) = sum((Yin(ind,ii)-predictedY(ind,ii)).^2);
    nullsse = sum((Yin(ind,ii)-Ymean(ind,ii)).^2);
    stats.cod(ii) = 1 - stats.sse(ii) / nullsse;
    stats.corr(ii) = corr(Yin(ind,ii),predictedY(ind,ii));
    stats.baseline_corr(ii) = corr(Yin(ind,ii),Ymean(ind,ii));
    [~,pv] = corrcoef(Yin(ind,ii),predictedY(ind,ii)); % original space
    if corr(Yin(ind,ii),predictedY(ind,ii))<0, stats.pval(ii) = 1;
    else, stats.pval(ii) = pv(1,2);
    end
    if deconfounding
        stats.sse_deconf(ii) = sum((YD(ind,ii)-predictedYD(ind,ii)).^2);
        nullsse_deconf = sum((YD(ind,ii)-YmeanD(ind,ii)).^2);
        stats.cod_deconf(ii) = 1 - stats.sse_deconf(ii) / nullsse_deconf;
        stats.corr_deconf(ii) = corr(YD(ind,ii),predictedYD(ind,ii));
        stats.baseline_corr_deconf(ii) = corr(YD(ind,ii),YmeanD(ind,ii));
        [~,pv] = corrcoef(YD(ind,ii),predictedYD(ind,ii)); % original space
        if corr(YD(ind,ii),predictedYD(ind,ii))<0, stats.pval_deconf(ii) = 1;
        else, stats.pval_deconf(ii) = pv(1,2);
        end
    end
    stats_max.sse(ii) = sum((Yin(ind,ii)-predictedY_max(ind,ii)).^2);
    nullsse = sum((Yin(ind,ii)-Ymean(ind,ii)).^2);
    stats_max.cod(ii) = 1 - stats_max.sse(ii) / nullsse;
    stats_max.corr(ii) = corr(Yin(ind,ii),predictedY_max(ind,ii));
    stats_max.baseline_corr(ii) = corr(Yin(ind,ii),Ymean(ind,ii));
    [~,pv] = corrcoef(Yin(ind,ii),predictedY_max(ind,ii)); % original space
    if corr(Yin(ind,ii),predictedY_max(ind,ii))<0, stats_max.pval(ii) = 1;
    else, stats_max.pval(ii) = pv(1,2);
    end
    if deconfounding
        stats_max.sse_deconf(ii) = sum((YD(ind,ii)-predictedYD_max(ind,ii)).^2);
        nullsse_deconf = sum((YD(ind,ii)-YmeanD(ind,ii)).^2);
        stats_max.cod_deconf(ii) = 1 - stats_max.sse_deconf(ii) / nullsse_deconf;
        stats_max.corr_deconf(ii) = corr(YD(ind,ii),predictedYD_max(ind,ii));
        stats_max.baseline_corr_deconf(ii) = corr(YD(ind,ii),YmeanD(ind,ii));
        [~,pv] = corrcoef(YD(ind,ii),predictedYD_max(ind,ii)); % original space
        if corr(YD(ind,ii),predictedYD_max(ind,ii))<0, stats_max.pval_deconf(ii) = 1;
        else, stats_max.pval_deconf(ii) = pv(1,2);
        end
    end
   
end

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
function [predictedY,predictedY_ST,W_FWLS,W,folds]...
    = stacking_predictions_BG_mf(Yin,Din,QpredictedY_store,QYin_store,sigma_store,alph_store,options,varargin)

rng('default') % set seed for reproducibility

M = length(Din); % M = number of repetitions of HMM to combine
for m = 1:M, Din{m}(eye(size(Din,1))==1) = 0; end % set the diagonal elements of the distance matrices to 0

[N,q] = size(Yin);

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
if nargin < 7 || isempty(options), options = struct(); end

if ~isfield(options,'CVscheme'), CVscheme = [10 10]; % 10-fold CV scheme (first element is for model evaluation and second element is for model selection)
else, CVscheme = options.CVscheme; end
if ~isfield(options,'CVfolds'), CVfolds = [];
else, CVfolds = options.CVfolds; end
% if ~isfield(options,'biascorrect'), biascorrect = 0;
% else, biascorrect = options.biascorrect; end
if ~isfield(options,'verbose'), verbose = 1; % display progress
else, verbose = options.verbose; end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
allcs = [];
if (nargin>7) && ~isempty(varargin{1}) % if correlation structure is defined (it is defined in nargin = 4 and then we check it's not empty)
    cs = varargin{1}; % let cs be the pre-defined correlation structure
    if ~isempty(cs)
        is_cs_matrix = (size(cs,2) == size(cs,1));
        if is_cs_matrix % if it's a square matrix then do this...
            if any(which_nan) % if there are any NaN values...
                cs = cs(~which_nan,~which_nan); % remove subjects with NaN values
            end
            [allcs(:,2),allcs(:,1)]=ind2sub([length(cs) length(cs)],find(cs>0)); % find indices of positive values of correlation structure
        else %else if it's not a square matrix (then it's a vector?)
            if any(which_nan)
                cs = cs(~which_nan); % remove subjects with NaN values
            end
            allcs = find(cs > 0); % find positive values of the correlation structure
        end
    end
else, cs = [];
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% get confounds
if (nargin>8) && ~isempty(varargin{2})
    confounds = varargin{2};
    confounds = confounds - repmat(mean(confounds),N,1);
    deconfounding = 1;
    if any(which_nan)
        confounds = confounds(~which_nan,:);
    end
else
    confounds = []; deconfounding = 0;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% get metafeatures
if (nargin>9) && ~isempty(varargin{3})
    metafeatures = varargin{3};
else
    metafeatures = ones(N,M);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

n_metafeatures = size(metafeatures,2)/M; % number of metafeatures


% Initialise variables
Ymean = zeros(N,q);
YD = zeros(N,q); % deconfounded signal
YmeanD = zeros(N,q); % mean in deconfounded space
predictedY = NaN(N,q); predictedY_max = zeros(N,q); % To store the final predictions
predictedY_old = NaN(N,q); %%%%%%%%%%%%%%% TEST CODE
predictedY_ST = zeros(N,M,q);
% if deconfounding, predictedYD = zeros(N,q); end
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

% Intialise weights of stacking (a weight for each repetition of HMM and each CV)
W = zeros(M,length(folds));
W_max = zeros(M,length(folds));
W_FWLS = zeros(M*n_metafeatures,length(folds));

% Convert predictions from cell array to full predictions

for ifold = [8 10] %1:length(folds)
   
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
   
       
    
    for ii = 1%:q
        
        % remove the subjects with NaN values for specified feature (from the actual feature values that we are trying to predict)
         ind = find(~isnan(Y(:,ii)));
         Yii = Y(ind,ii);

       
        QDin = cell(M,1);
        for m = 1:M, QDin{m} = D{m}(ind,ind); end % get the distance matrices of the subjects that didn't have NaN values for the selected variable

        % I've already removed NaNs so don't need to remove using above code
        Yii = Y;
        % centering response
        my = mean(Yii);
        Yii = Yii - repmat(my,size(Yii,1),1);
%          Ymean(J,ii) = my
       
        % Store the 'inner loop for stacking estimation' predictions
        QpredictedY = QpredictedY_store{ifold};
        QYin = QYin_store{ifold};

        
        % Create matrix, A, of learners and metafeatures
        F_train = metafeatures(ji,:);
        F_test = metafeatures(J,:);

        A_train = NaN(size(ji,2),n_metafeatures*M);
        for i = 1:n_metafeatures
              A_train(:,M*i-M+1:M*i) = F_train(:,M*i-M+1:M*i).*QpredictedY;
        end

        %w = (QpredictedY' * QpredictedY + lambda * eye(M)) \ (QpredictedY' * QYin);
        opts1 = optimset('display','off');
        % find the stacking weights (for this particular feature) for the
        % repetitions of HMM
        w = lsqlin(QpredictedY,QYin,[],[],ones(1,M),1,zeros(M,1),ones(M,1),[],opts1);
        W(:,ifold) = w;

        

        % find the stacking weights for the FWLS
        w_FWLS = lsqlin(A_train,QYin,[],[],ones(1,M*n_metafeatures),1,zeros(M*n_metafeatures,1),ones(M*n_metafeatures,1),[],opts1); %w_FWLS = lsqlin(A(ji,:),QYin,[],[],[],[],zeros(M*n_metafeatures,1),ones(M*n_metafeatures,1),[],opts1);
        W_FWLS(:,ifold) = w_FWLS;
        
       
        % This bit basically finds the best predictor (of the e.g. 5
        % repetitions of the HMM)
%         [~,best] = min(sum((QpredictedY-QYin).^2));
%         W_max(best,ifold) = 1;
        
        sigma = sigma_store{ifold};
        alph = alph_store{ifold};
        rng('default')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Now we make the actual predictions for each of our HMM repetitions that
% we will then use to combine into a new prediction using the weights.
% We do this on all the data.
        for m = 1:M % for each repetition of the HMM
            Dii = D{m}(ind,ind); D2ii = D2{m}(:,ind); 
            K = gauss_kernel(Dii,sigma(m));
            K2 = gauss_kernel(D2ii,sigma(m));
            
            Nji = length(ind);
            I = eye(Nji);

            ridg_pen_scale = mean(diag(K)); % This always seems to be 1 currently?
            beta = (K + ridg_pen_scale * alph(m) * I) \ Yii;

            % Get the predictions of each HMM repetition (FOR THE OUTER
            % LOOP TEST DATA)
            predictedY_ST(J,m,ii) = K2 * beta + my;
        end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
        predictedY_old(J,ii) = predictedY_ST(J,:,ii) * w; % MULTIPLY THE TEST PREDICTIONS WITH THE BEST WEIGHTS ACROSS THE TRAINING DATA


        A_test = NaN(size(J,2),n_metafeatures*M);
        for i = 1:n_metafeatures
              A_test(:,M*i-M+1:M*i) = F_test(:,M*i-M+1:M*i).*predictedY_ST(J,:,ii);
        end

        predictedY(J,ii) = A_test*w_FWLS;

    end
    ifold;
    F_test;
    predictedY_ST(J,:,ii);
    J';
    [predictedY_old(J,ii) predictedY(J,ii)]
    [w w_FWLS(1:5)];
    %disp(['Fold ' num2str(ifold) ])
   
end
%     sum(isnan(predictedY_old))
%     sum(isnan(predictedY))

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
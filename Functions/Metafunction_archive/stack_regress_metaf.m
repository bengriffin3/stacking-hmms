function [vars_hat_stacked,vars_hat_FWLS,exp_var_stacked,exp_var_FWLS, weights, weights_FWLS] = stack_regress_metaf(Yin,vars_hat,options,vargin,metafeatures)
%%% function to perform stacked ridge regression constraining the weights
%%% to be non-negative and sum to 1, as well as using metafeatures to
%%% improve prediction accuracy
% INPUT
% vars              (no. subjects x no. variables)  matrix of phenotypic values to predict
% vars_hat          (no. subjects x no. variables x no. repetitions of HMM) array of predicted phenotypes
% mf                (no. subjects x no. metafeatures) matrix of metafeatures
%                   where the first metafeature is the constant vector (a vector of 1s)
% OUTPUT
% vars_hat_stacked  (no. subjects x variables) matrix of stacked predictions for each variable
% vars_hat_FWLS     (no. subjects x variables) matrix of stacked predictions with metafeatures for each variable
% exp_var_stacked   (no. variables x 1) vector of r^2 of stacked predictions for each variable
% exp_var_FWLS      (no. variables x 1) vector of r^2 of stacked predictions with metafeatures for each variable

n_subjects = size(Yin,1); % number of subjects
n_var = size(Yin,2); % number of intelligence variables
n_models = size(vars_hat,3); % number models to combine


if nargin < 3 || isempty(options), options = struct(); end
if ~isfield(options,'CVscheme'), CVscheme = [10 10];
else, CVscheme = options.CVscheme; end
if ~isfield(options,'CVfolds'), CVfolds = [];
else, CVfolds = options.CVfolds; end

if nargin < 5
    metafeatures = ones(n_subjects,1);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CHECK CORRELATION STRUCTURE
which_nan = false(N,1);
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SET UP CROSS-VALIDATION FOLDS
if isempty(CVfolds)
    if CVscheme(1)==1
        folds = {1:N};
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

n_metafeatures = size(metafeatures,2)/n_models; % number of metafeatures

% Initialise stacking variables
weights = NaN(n_models,n_folds,n_var);
vars_hat_stacked = NaN(n_subjects,n_var);
exp_var_stacked = NaN(n_var,1);

% Initialise FWLS variables
weights_FWLS = NaN(n_models*n_metafeatures,n_folds,n_var);
vars_hat_FWLS = NaN(n_subjects,n_var);
exp_var_FWLS = NaN(n_var,1);


indices_all = crossvalind('Kfold',1:n_subjects,n_folds);

for m = 1:n_var

    % Save data for specific variable
    y_all = Yin(:,m);
    V_all = squeeze(vars_hat(:,m,:));

    % Remove subjects with NaN values
    y = y_all; V = V_all; F = metafeatures; indices = indices_all;
    which_nan = isnan(y);
    if any(which_nan)
        y = y_all(~which_nan);
        V = V_all(~which_nan,:);
        F = metafeatures(~which_nan,:);
        indices = indices_all(~which_nan);
        warning('NaN found on Yin, will remove...')
    end

    % Initialise variables for cross-validation
    yhat_star = NaN(sum(~which_nan),1);
    yhat_star_FWLS = NaN(sum(~which_nan),1);
    
    for ifold = 1:length(folds)

        test = folds{ifold};
        train = setdiff(1:N,test);

        % Note indices of testing and training data
        %test = (indices == ifold);
        train = ~test;

        % Split into train and test data
        y_train = y(train); %y_test = y(test);
        V_train = V(train,:); V_test = V(test,:);

        % Create A matrix of learners and metafeatures
        A = NaN(size(y,1),n_metafeatures*n_models);
        for i = 1:n_metafeatures % 2 metafeatures
              A(:,n_models*i-n_models+1:n_models*i) = F(:,n_models*i-n_models+1:n_models*i).*V;
        end

        % Stacking weights (determined by least squares)
        opts1 = optimset('display','off');
        weights(:,ifold,m) = lsqlin(V_train,y_train,[],[],ones(1,n_models),1,zeros(n_models,1),ones(n_models,1),[],opts1);
        weights_FWLS(:,ifold,m) = lsqlin(A(train,:),y_train,[],[],ones(1,n_models*n_metafeatures),1,zeros(n_models*n_metafeatures,1),ones(n_models*n_metafeatures,1),[],opts1);

        % Use the weights to form new predictions
        yhat_star(test) = V_test * weights(:,ifold,m); % stacking
        yhat_star_FWLS(test) = A(test,:) * weights_FWLS(:,ifold,m); % FWLS



    end
        % Store the new predictions
        vars_hat_stacked(~which_nan,m) = yhat_star;
        vars_hat_FWLS(~which_nan,m) = yhat_star_FWLS;

    % Calculate new levels of explained variance (for combined model)
    exp_var_stacked(m) = corr(vars_hat_stacked(~which_nan,m),y).^2;
    exp_var_FWLS(m) = corr(vars_hat_FWLS(~which_nan,m),y).^2;

end
end
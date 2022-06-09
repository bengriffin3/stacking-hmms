function [predictedY,predictedY_ST,W_FWLS,W,folds]...
    = stacking_predictions_meta(Yin,Din,QpredictedY_store,QYin_store,sigma_store,alph_store,options,cs,confounds,metafeatures,folds)
rng('default') % set seed for reproducibility

M = length(Din); % M = number of repetitions of HMM to combine
for m = 1:M, Din{m}(eye(size(Din,1))==1) = 0; end % set the diagonal elements of the distance matrices to 0

[N,q] = size(Yin);
% N = number of subjects
% q = number of classes (features to predict I think?)

verbose = 0;

n_metafeatures = size(metafeatures,2)/M; % number of metafeatures

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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



% Initialise variables
Ymean = zeros(N,q);
predictedY = zeros(N,q);  % To store the final predictions
predictedY_ST = zeros(N,M,q);




% Intialise weights of stacking (a weight for each repetition of HMM and each CV)
W = NaN(M,length(folds));
W_FWLS = NaN(M*n_metafeatures,length(folds));


for ifold = 1:length(folds)
        
    % Load the data from the previous run
    QpredictedY = QpredictedY_store{ifold};
    QYin = QYin_store{ifold};
    sigma = sigma_store{ifold};
    alph = alph_store{ifold};
   

    % Split the data into test and train data
    J = folds{ifold}; % get the indices of the test data
    ji = setdiff(1:N,J); % get the indices of the training data (i.e. all indices except those in J)
   
    D = cell(M,1); D2 = cell(M,1);
    for m = 1:M, D{m} = Din{m}(ji,ji); D2{m} = Din{m}(J,ji); end
    Y = Yin(ji,:); % Note training feature values
    
    % Create matrix, A, of learners and metafeatures
    F_train = metafeatures(ji,:);
    F_test = metafeatures(J,:);

    A_train = NaN(size(ji,2),n_metafeatures*M);
    for i = 1:n_metafeatures
        A_train(:,M*i-M+1:M*i) = F_train(:,M*i-M+1:M*i).*QpredictedY;
    end
       


    for ii = 1%:q % for each feature that we want to predict
        
        % remove the subjects with NaN values for specified feature (from the actual feature values that we are trying to predict)
         ind = find(~isnan(Y(:,ii))); 
         Yii = Y(ind,ii);

       
        QDin = cell(M,1);
        for m = 1:M, QDin{m} = D{m}(ind,ind); end % get the distance matrices of the subjects that didn't have NaN values for the selected variable

        %Yii = Y;

        % centering response
        my = mean(Yii);
        Yii = Yii - repmat(my,size(Yii,1),1);
        Ymean(J,ii) = my;
%         QYin = Yii;




        %w = (QpredictedY' * QpredictedY + lambda * eye(M)) \ (QpredictedY' * QYin);
        opts1 = optimset('display','off');
        % find the stacking weights (for this particular feature) for the
        % repetitions of HMM
        w = lsqlin(QpredictedY,QYin,[],[],ones(1,M),1,zeros(M,1),ones(M,1),[],opts1);
        W(:,ifold) = w;



        % find the stacking weights for the FWLS
        w_FWLS = lsqlin(A_train,QYin,[],[],ones(1,M*n_metafeatures),1,zeros(M*n_metafeatures,1),ones(M*n_metafeatures,1),[],opts1); %w_FWLS = lsqlin(A(ji,:),QYin,[],[],[],[],zeros(M*n_metafeatures,1),ones(M*n_metafeatures,1),[],opts1);
        W_FWLS(:,ifold) = w_FWLS;

        rng('default')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        for m = 1:M % for each repetition of the HMM
            Dii = D{m}(ind,ind); D2ii = D2{m}(:,ind);
            K = gauss_kernel(Dii,sigma(m));
            K2 = gauss_kernel(D2ii,sigma(m));
            Nji = length(ind);
            I = eye(Nji);
        
            ridg_pen_scale = mean(diag(K));
            beta = (K + ridg_pen_scale * alph(m) * I) \ Yii;
            predictedY_ST(J,m,ii) = K2 * beta + my;
        end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        predictedY_old(J,ii) = predictedY_ST(J,:,ii) * w; % MULTIPLY THE TEST PREDICTIONS WITH THE BEST WEIGHTS ACROSS THE TRAINING DATA

        A_test = NaN(size(J,2),n_metafeatures*M);
        for i = 1:n_metafeatures
            A_test(:,M*i-M+1:M*i) = F_test(:,M*i-M+1:M*i).*predictedY_ST(J,:,ii);
        end

        predictedY(J,ii) = A_test*w_FWLS;


    [predictedY_old(J,ii) predictedY(J,ii)]


   
    end
    
   
    %disp(['Fold ' num2str(ifold) ])
   
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
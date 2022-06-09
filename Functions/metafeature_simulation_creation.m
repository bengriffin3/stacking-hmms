function [meta_simu_norm,pearson_error_metafeature] = metafeature_simulation_creation(squared_error,Metafeature,simulation_options,vars_target)
% we simulate the metafeature by using the fact that the cossine similarity
% of two vectors, which have been normalized by subtracting the vector
% means (i.e. the centered cossine similarity), is equivalent to the Pearson 
% correlation coefficient
% see for more details: https://stats.stackexchange.com/questions/15011/generate-a-random-variable-with-a-defined-correlation-to-an-existing-variables/15035#15035
n_repetitions = size(squared_error,2);
n_subjects = size(squared_error,1);
n_metafeatures = size(Metafeature,2)/n_repetitions;

% intialise variables
meta_simu = NaN(n_subjects,n_repetitions*n_metafeatures);
pearson_error_metafeature = NaN(1,n_repetitions);
pearson_vars_metafeature = NaN(1,n_repetitions);

squared_error = repmat(squared_error,1,n_metafeatures);

if isempty(simulation_options.corr_specify)
    for rep = 1:n_repetitions*n_metafeatures
        [pearson_error_metafeature(rep), ~] = corr(squared_error(:,rep),Metafeature(:,rep),'type','pearson', 'rows','complete');
        [pearson_vars_metafeature(rep), ~] = corr(vars_target,Metafeature(:,rep),'type','pearson', 'rows','complete');
    end
    
    if strcmp(simulation_options.simulate_corr_with,'prediction_accuracy')
        p_meta = pearson_error_metafeature; % make correlation more favourable
        theta = acos(p_meta); % corresponding angle
        x1 = repmat(squared_error,1,n_metafeatures); % generate fixed given variable (this will be squared error for me) % THIS IS WHAT WE WANT TO GENERATE A VECTOR SIMILAR TO
    elseif strcmp(simulation_options.simulate_corr_with, 'vars_targets')
        p_meta = pearson_vars_metafeature;
        theta = acos(p_meta); % corresponding angle
        x1 =repmat(vars_target,n_metafeatures,n_repetitions); % generate fixed given variable (this will be squared error for me) % THIS IS WHAT WE WANT TO GENERATE A VECTOR SIMILAR TO
    end

else    
    p_meta = repmat(simulation_options.corr_specify,1,n_repetitions*n_metafeatures);
    theta = acos(p_meta); % corresponding angle
    x1 = repmat(squared_error,1,n_metafeatures); % generate fixed given variable (this will be squared error for me) % THIS IS WHAT WE WANT TO GENERATE A VECTOR SIMILAR TO
end


x2 = randn(n_subjects,n_repetitions*n_metafeatures); % generate rv variable that
Xctr1 = x1 - mean(x1); % center columns to mean 0
Xctr2 = x2 - mean(x2);% center columns to mean 0
Id = eye(n_subjects); % generate identity matrix size n_subjects x n_subjects

for i = 1:n_repetitions*n_metafeatures
    X = Xctr1(:,i);
    P = X*((X'*X)\X'); % get projection matrix onto space by defined by x1
    Xctr2_orth = (P-Id) * Xctr2(:,i); % make Xctr2 orthogonal to Xctr1 using projection matrix
    Y1 = Xctr1(:,i) * 1./(sqrt(sum(Xctr1(:,i).^2))); % scale columns to length 1
    Y2 = Xctr2_orth * 1./(sqrt(sum(Xctr2_orth.^2))); % scale columns to length 1
    meta_simu(:,i) = Y2 + (1/tan(theta(:,i))) * Y1; % final new vector
end
%corr(squared_error,meta_simu)  % check this is desired correlation

%[corr(Metafeature,vars_target, 'rows','complete') corr(meta_simu,vars_target, 'rows','complete')]
%round(corr(Metafeature,vars_target, 'rows','complete'),6) == round(corr(meta_simu,vars_target, 'rows','complete'),6)

meta_simu_norm = ((meta_simu-min(meta_simu))./(max(meta_simu)-min(meta_simu)));

end
function [vars_target_clean,Predictions_clean,Metafeature_clean,non_nan_idx] = ...
                        nan_subject_remove(var,vars,vars_hat,Metafeature_norm)%,Squared_error_gaussianized)

    % Store data for variable
    Predictions_clean = squeeze(vars_hat(:,var,:));
    vars_target_clean = vars(:,var);
    Metafeature_clean = Metafeature_norm;
    %squared_error_gaussianized_clean = Squared_error_gaussianized(:,:,var);
    
    % Remove subjects with missing values
    non_nan_idx = ~isnan(vars_target_clean);
    which_nan = isnan(vars_target_clean);
    if any(which_nan)
        vars_target_clean = vars_target_clean(~which_nan);
        Predictions_clean = Predictions_clean(~which_nan,:);
        Metafeature_clean = Metafeature_clean(~which_nan,:);
        %squared_error_gaussianized_clean = squared_error_gaussianized_clean(~which_nan,:);
        warning('NaN found on Yin, will remove...')
    end
    
end
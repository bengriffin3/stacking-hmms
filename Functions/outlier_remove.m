function [vars_target_clean,Predictions_clean,Metafeature_clean,squared_error_gaussianized_clean,non_outlier_idx] = ...
                        outlier_remove(vars_target,Predictions,Metafeature,squared_error_gaussianized)
                    
    vars_target_clean = vars_target;
    Predictions_clean = Predictions;
    Metafeature_clean = Metafeature;
    squared_error_gaussianized_clean = squared_error_gaussianized;
                    
    % Note index of outliers
    non_outlier_idx = find(~isoutlier(vars_target));
    which_outlier = isoutlier(vars_target);
    if any(which_outlier)
        vars_target_clean = vars_target(~which_outlier);
        Predictions_clean = Predictions(~which_outlier,:);
        Metafeature_clean = Metafeature(~which_outlier,:);
        squared_error_gaussianized_clean = squared_error_gaussianized(~which_outlier,:);
        warning('Outliers found on Yin, will remove...')
    end
                    
end
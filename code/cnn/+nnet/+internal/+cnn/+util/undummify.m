function labels = undummify( scores, labelCategories)
    % undummify   Convert scores into categorical output. 
    % labelCategories is supposed to be a column categorical array with
    % underlying categories already in the correct order and with given
    % ordinality.
    
    %   Copyright 2015-2017 The MathWorks, Inc.
    
    [mxValues, idx] = max(scores,[],2);
    labels = labelCategories(idx);
    
    % Replace NaN maxima with <undefined> labels
    nans = isnan(mxValues);
    if any(nans)
        labels(nans) = setcats(categorical(NaN,'Ordinal',isordinal(labels)), ...
            categories(labels));
    end
end
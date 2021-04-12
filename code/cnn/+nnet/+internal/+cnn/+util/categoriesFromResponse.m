function cats = categoriesFromResponse(response)
    % categoriesFromResponse   Extracts class names and ordinality from the
    % categorical array response and stores it in the column categorical array cats.
    % If response is not categorical, cats is an empty categorical array.
    %
    
    %   Copyright 2017 The MathWorks, Inc.
    
    if isa(response, 'categorical')    
        classNames = categories(response);
        ordinality = isordinal(response);
        cats = categorical(classNames, classNames, 'Ordinal', ordinality);
    else
        cats = categorical();
    end
end
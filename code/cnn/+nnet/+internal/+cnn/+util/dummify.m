function dummifiedOut = dummify(categoricalIn)
    % iDummify   Convert a categorical input into a dummified output.
    %
    % dummifiedOut(1,1,i,j)=1 if observation j is in class i, and zero
    % otherwise. Therefore, dummifiedOut will be of size [1, 1, K, N],
    % where K is the number of categories and N is the number of
    % observation in categoricalIn.
    
    %   Copyright 2015-2016 The MathWorks, Inc.
    
    numObservations = numel(categoricalIn);
    numCategories = numel(categories(categoricalIn));
    dummifiedSize = [1, 1, numCategories, numObservations];
    dummifiedOut = zeros(dummifiedSize);
    categoricalIn = iMakeHorizontal( categoricalIn );
    idx = sub2ind(dummifiedSize(3:4), int32(categoricalIn), 1:numObservations);
    dummifiedOut(idx) = 1;
end

function vec = iMakeHorizontal( vec )
    vec = reshape( vec, 1, numel( vec ) );
end
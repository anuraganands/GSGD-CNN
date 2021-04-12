classdef ResponseNameable < handle
    % ResponseNameable Abstract interface for declaring that a MiniBatchDatastore has named responses.
    
    %   Copyright 2017 The MathWorks, Inc.
    
    properties (Abstract)
       ResponseNames 
    end
    
end

classdef FourDArrayReadable < handle
    % FourDArrayReadable Abstract interface for declaring that a
    % MiniBatchDatastore uses an internal interface for read/readByIndex in
    % which read may return data as a 4-D array or cell and Response is a
    % field of the info struct. This is done as a performance optimization
    % for lightweight dispatching.
    
    %   Copyright 2017 The MathWorks, Inc.
    
    properties (Constant)
        UsesInternalReadInterface = true
    end
    
    methods
        function ds = FourDArrayReadable 
        end
    end
    
end

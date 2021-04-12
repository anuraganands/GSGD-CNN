classdef Precision
    % Precision     Class to handle data precision
    
    %   Copyright 2015-2017 The MathWorks, Inc.
    
    properties (SetAccess = private)
        % Type   Type of precision to be used.
        %        One of:
        %        'single'
        %        'double'
        Type
    end
    
    methods
        function this = Precision(type)
            this.Type = type;
        end
        
        function data = cast(this, data)
            % cast   Cast floating point data using the precision specified
            % at construction time
            data = cast(data, this.Type);
        end
        
        function data = zeros(this, dataSize)
            % zeros   Allocate a matrix of zeros of size dataSize 
            data = zeros(dataSize, this.Type);
        end
        
        function data = ones(this, dataSize)
            % ones   Allocate a matrix of all ones of size dataSize 
            data = ones(dataSize, this.Type);
        end
    end
end

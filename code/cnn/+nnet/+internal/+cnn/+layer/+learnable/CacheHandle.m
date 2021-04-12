classdef CacheHandle < handle
    % CacheHandle   Handle object used to cache values
    
    %   Copyright 2016-2017 The MathWorks, Inc.
    
    properties (SetAccess = private)
        Value = []
    end
    
    properties (Access=private)
        CacheEmpty;
    end
    
    methods
        function this = CacheHandle(value)
            if nargin
                this.Value = value;
                this.CacheEmpty = false;
            else
                this.CacheEmpty = true;
                this.Value = [];
            end
        end
        
        function tf = isEmpty(this)
            tf = this.CacheEmpty;
        end
        
        function fillCache(this, value)
            this.Value = value;
            this.CacheEmpty = false;
        end
        
        function clearCache(this)
            this.CacheEmpty = true;
            this.Value = [];
        end
    end
end

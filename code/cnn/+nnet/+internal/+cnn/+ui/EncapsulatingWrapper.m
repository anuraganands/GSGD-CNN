classdef EncapsulatingWrapper < handle
    % EncapsulatingWrapper   Holds onto any object as a private property
    
    properties(Access = private)
        % Object   The object being held
        Object
    end
    
    methods
        function this = EncapsulatingWrapper(obj)
            this.Object = obj; 
        end
    end
    
end


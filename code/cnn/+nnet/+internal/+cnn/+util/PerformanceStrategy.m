classdef(Abstract) PerformanceStrategy
    % PerformanceStrategy   Interface for summary performance execution strategies
    %
    %   A class that inherits from this interface can be used to implement
    %   different performance strategies, i.e. classification accuracy or
    %   root-mean-squared error, for different data types, i.e. image or
    %   vector.
    
    %   Copyright 2017 The MathWorks, Inc.
    
    methods(Abstract)
        % accuracy   Classification accuracy
        acc = accuracy(this, predictions, response)
        
        % rmse   Root-mean-squared error
        err = rmse(this, predictions, response)
    end
end
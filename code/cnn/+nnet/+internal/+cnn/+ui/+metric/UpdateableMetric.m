classdef(Abstract) UpdateableMetric < handle
    % UpdateableMetric   Interface for metrics which can be updated from a struct of information.
    
    %   Copyright 2017-2018 The MathWorks, Inc.
    
    methods(Abstract)
        update(this, infoStruct)
        % update   Updates the metric from the information given in the
        % infoStruct.
        
        updatePostTraining(this, infoStruct)
        % updatePostTraining   Updates the metric from the information
        % given in the infoStruct, outside the main training loop.
    end
    
end


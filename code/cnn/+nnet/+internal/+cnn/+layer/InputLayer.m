classdef(Abstract) InputLayer < nnet.internal.cnn.layer.Layer
    % InputLayer   Interface for input layers
    
    %   Copyright 2017 The MathWorks, Inc.
    
    properties(SetAccess = private)
        % InputNames   Input layers cannot have inputs, because nothing
        %              connects TO and input layer
        InputNames = {}
        
        % OutputNames   Input layers have one output
        OutputNames = {'out'}
    end
end
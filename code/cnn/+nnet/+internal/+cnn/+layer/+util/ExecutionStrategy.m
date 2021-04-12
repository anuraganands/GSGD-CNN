classdef(Abstract) ExecutionStrategy
    % ExecutionStrategy   Interface for execution strategies
    %
    %   A class that inherits from this interface can be used to implement
    %   different ways of computing forward propagation, backward
    %   propagation and gradient calculations using different hardware
    %   resources (e.g. host or GPU).
    
    %   Copyright 2016-2017 The MathWorks, Inc.
    
    methods(Abstract)
        % forward   The forward method for this strategy
        %   [Z, memory] = forward(this, X, varargin) takes an input array
        %   X, and possibly a series of learned parameters varargin, and
        %   returns an output Z. The second output memory is used by
        %   stochastic layers to record any randomized information that
        %   will be needed for backpropagation. It is not used by the
        %   majority of layers.
        [Z, memory] = forward(this, X, varargin)
        % backward   The backward method for this strategy
        %   dX = backward(this, Z, varargin) takes an input array Z, and
        %   possibly a series of further input arrays and learned
        %   parameters varargin, and returns an output array dX and output
        %   cell array of derivative w.r.t. the weights dW. For layers with
        %   no learnable parameters dW should be [].
        [dX,dW] = backward(this, Z, varargin)
    end
end
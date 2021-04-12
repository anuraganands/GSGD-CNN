classdef LSTMHostStrategy < nnet.internal.cnn.layer.util.ExecutionStrategy
    % LSTMHostStrategy   Execution strategy for running LSTM on the host
    
    %   Copyright 2017 The MathWorks, Inc.
    
    methods
        function [Y, memory] = forward(~, X, W, R, b, c0, y0)
            [Y, C, G] = nnet.internal.cnnhost.lstmForward(X, W, R, b, c0, y0);
            memory.CellState = C;
            memory.Gates = G;
        end
        
        function varargout = backward(~, X, W, R, b, c0, y0, Y, memory, dZ)
            C = memory.CellState;
            G = memory.Gates;
            [ varargout{1:nargout} ] =  nnet.internal.cnnhost.lstmBackward(X, W, R, b, c0, y0, Y, C, G, dZ);
        end
    end
end
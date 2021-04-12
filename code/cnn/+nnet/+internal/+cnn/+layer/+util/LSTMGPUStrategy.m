classdef LSTMGPUStrategy < nnet.internal.cnn.layer.util.ExecutionStrategy
    % LSTMGPUStrategy   Execution strategy for running LSTM on the GPU
    
    %   Copyright 2017 The MathWorks, Inc.
    
    methods
        function [Y, memory] = forward(this, X, W, R, b, c0, y0)
            % Prepare data
            [X, HX, CX, Wc] = this.prepareDataForForward(X, W, R, b, c0, y0);
            % Call cuDNN forward method
            [Y, HY, CY, Ws] = nnet.internal.cnngpu.lstmForwardTrain(X, HX, CX, Wc);
            % Pass variables into memory for backpropagation
            memory.CellState = CY;
            memory.HiddenState = HY;
            memory.Workspace = Ws;
        end
        
        function [dX, dW, dR, dB] = backward(this, X, W, R, b, c0, y0, Y, memory, dY)
            % Prepare data
            [Y, HY, CY, dY, dHY, dCY, X, y0, c0, Wc, Ws, H, D] = this.prepareDataForBackward(X, W, R, b, c0, y0, Y, memory, dY);
            % Call cuDNN backward method
            args = { Y, HY, CY, dY, dHY, dCY, X, y0, c0, Wc, Ws };
            needsWeightGradients = nargout > 1;
            if ~needsWeightGradients
                % TODO(g1655557): Remove additional output args when
                % supported
                [dX, ~, ~, ~] = nnet.internal.cnngpu.lstmBackward( args{:} );
            else
                [dX, ~, ~, dWc] = nnet.internal.cnngpu.lstmBackward( args{:} );
                % Convert to NNT weights format
                [dW, dR, dB] = nnet.internal.cnngpu.util.LSTMWeightsConverter.fromCudnnDerivative(dWc, H, D);
            end
        end
    end
    
    methods(Access = protected)
        function [X, HX, CX, Wc] = prepareDataForForward(~, X, W, R, b, c0, y0)
            % Prepare data passed into the strategy for the internal
            % forward method
            
            % Convert to cuDNN weights format
            Wc = nnet.internal.cnngpu.util.LSTMWeightsConverter.toCudnn(W, R, b);
            % Expand initial states along batch dimension if necessary
            N = size( X, 2 );
            HX = iExpandBatchDimension(y0, N);
            CX = iExpandBatchDimension(c0, N);
        end
        
        function [Y, HY, CY, dY, dHY, dCY, X, HX, CX, Wc, Ws, H, D] = prepareDataForBackward(~, X, W, R, b, c0, y0, Y, memory, dY)
            % Prepare data passed into the strategy for the internal
            % backward method
            
            % Convert to cuDNN weights format
            Wc = nnet.internal.cnngpu.util.LSTMWeightsConverter.toCudnn(W, R, b);
            % Determine problem dimensions
            N = iGetBatchDimension( X );
            [D, H] = iGetInputAndHiddenDimensions( W );
            % Expand initial states along batch dimension if necessary
            HX = iExpandBatchDimension(y0, N);
            CX = iExpandBatchDimension(c0, N);
            % Declare variables from memory required for backpropagation
            CY = memory.CellState;
            HY = memory.HiddenState;
            Ws = memory.Workspace;
            dHY = zeros( size(HY), 'like', HY );
            dCY = zeros( size(CY), 'like', CY );
        end
    end
end

function N = iGetBatchDimension( X )
N = size( X, 2 );
end

function [D, H] = iGetInputAndHiddenDimensions( W )
[fourH, D] = size( W );
H = fourH./4;
end

function V = iExpandBatchDimension(V, N)
if size(V, 2) == 1
    V = repmat(V, 1, N);
end
end
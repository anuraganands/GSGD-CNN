classdef BiLSTMGPUStrategy < nnet.internal.cnn.layer.util.ExecutionStrategy
    % BiLSTMGPUStrategy   Execution strategy for running BiLSTM on the GPU
    
    %   Copyright 2017 The MathWorks, Inc.
    
    methods
        function [Y, memory] = forward(this, X, W, R, b, c0, y0)
            % Prepare data
            [X, HX, CX, Wc] = this.prepareDataForForward(X, W, R, b, c0, y0);
            % Call cuDNN forward method
            [Y, HY, CY, Ws] = nnet.internal.cnngpu.biLSTMForwardTrain(X, HX, CX, Wc);
            % Create memory struct
            memory.CellState = cat(1, CY(:, :, 1), CY(:, :, 2));
            memory.HiddenState = cat(1, HY(:, :, 1), HY(:, :, 2));
            memory.Workspace = Ws;
        end
        
        function [dX, dW, dR, db] = backward(this, X, W, R, b, c0, y0, Y, memory, dY)
            % Prepare data
            [Y, HY, CY, dY, dHY, dCY, X, HX, CX, Wc, Ws, H, D] = this.prepareDataForBackward(X, W, R, b, c0, y0, Y, memory, dY);
            % Call cuDNN backward method
            args = { Y, HY, CY, dY, dHY, dCY, X, HX, CX, Wc, Ws };
            needsWeightGradients = nargout > 1;
            if ~needsWeightGradients
                % TODO(g1655557): Remove additional output args when
                % supported
                [dX, ~, ~, ~] = nnet.internal.cnngpu.biLSTMBackward( args{:} );
            else
                [dX, ~, ~, dWc] =  nnet.internal.cnngpu.biLSTMBackward( args{:} );
                % Convert to NNT weights format
                [dW, dR, db] = nnet.internal.cnngpu.util.BiLSTMWeightsConverter.fromCudnnDerivative(dWc, H, D);
            end
        end
    end
    
    methods(Access = protected)
        function [X, HX, CX, Wc] = prepareDataForForward(~, X, W, R, b, c0, y0)
            % Prepare data passed into the strategy for the internal
            % forward method
            
            % Convert to cuDNN weights format
            Wc = nnet.internal.cnngpu.util.BiLSTMWeightsConverter.toCudnn(W, R, b);
            
            % Expand initial states along batch dimension if necessary
            N = size( X, 2 );
            H = size( R, 2 );
            y0 = iExpandBatchDimension(y0, N);
            c0 = iExpandBatchDimension(c0, N);
            HX = cat(3, y0(1:H, :), y0(1+H:2*H, :));
            CX = cat(3, c0(1:H, :), c0(1+H:2*H, :));
        end
        
        function [Y, HY, CY, dY, dHY, dCY, X, HX, CX, Wc, Ws, H, D] = prepareDataForBackward(~, X, W, R, b, c0, y0, Y, memory, dY)
            % Prepare data passed into the strategy for the internal
            % backward method
            
            % Convert to cuDNN weights format
            Wc = nnet.internal.cnngpu.util.BiLSTMWeightsConverter.toCudnn(W, R, b);
            % Determine problem dimensions
            N = iGetBatchDimension( X );
            [D, H] = iGetInputAndHiddenDimensions( W );
            % Expand initial states along batch dimension if necessary
            y0 = iExpandBatchDimension(y0, N);
            c0 = iExpandBatchDimension(c0, N);
            HX = cat(3, y0(1:H, :), y0(1+H:2*H, :));
            CX = cat(3, c0(1:H, :), c0(1+H:2*H, :));
            % Declare variables from memory required for backpropagation
            cEnd = memory.CellState;
            CY = cat(3, cEnd(1:H, :), cEnd(1+H:2*H, :));
            hEnd = memory.HiddenState;
            HY = cat(3, hEnd(1:H, :), hEnd(1+H:2*H, :));
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
[eightH, D] = size( W );
H = eightH./8;
end

function V = iExpandBatchDimension(V, N)
if size(V, 2) == 1
    V = repmat(V, 1, N);
end
end
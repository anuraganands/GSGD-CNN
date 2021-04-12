classdef BiLSTMHostStrategy < nnet.internal.cnn.layer.util.ExecutionStrategy
    % BiLSTMHostStrategy   Execution strategy for running LSTM on the host
    
    %   Copyright 2017 The MathWorks, Inc.
    
    methods
        function [Y, memory] = forward(~, X, W, R, b, c0, y0)
            % Split data into forward/backward sequences
            [Wf, Rf, bf, Wb, Rb, bb] = iSplitWeights(W, R, b);
            [c0f, y0f, c0b, y0b] = iSplitStates(c0, y0);
            
            % Forward sequence
            [Yf, Cf, Gf] = nnet.internal.cnnhost.lstmForward(X, Wf, Rf, bf, c0f, y0f);
            
            % Backward sequence
            [Yb, Cb, Gb] = nnet.internal.cnnhost.lstmForward(flip(X, 3), Wb, Rb, bb, c0b, y0b);
            
            % Concatenate outputs
            Y = cat(1, Yf, flip(Yb, 3));
            C = cat(1, Cf, Cb);
            G = cat(1, Gf, Gb);
            
            % Allocate memory
            memory.CellState = C;
            memory.HiddenState = cat(1, Yf(:, :, end), Yb(:, :, end));
            memory.Gates = G;
        end
        
        function [dX, dW, dR, dB] = backward(~, X, W, R, b, c0, y0, Y, memory, dY)
            % Split data into forward/backward sequences
            [Wf, Rf, bf, Wb, Rb, bb] = iSplitWeights(W, R, b);
            [c0f, y0f, c0b, y0b] = iSplitStates(c0, y0);
            C = memory.CellState;
            G = memory.Gates;
            [Yf, Cf, Gf, dYf, Yb, Cb, Gb, dYb] = iSplitForwardOutputs(Y, C, G, dY);
            
            % Forward sequence
            [dXf, dWeightsf{1:(nargout-1)}] =  nnet.internal.cnnhost.lstmBackward(X, Wf, Rf, bf, c0f, y0f, Yf, Cf, Gf, dYf);
            
            % Backward sequence
            [dXb, dWeightsb{1:(nargout-1)}] =  nnet.internal.cnnhost.lstmBackward(flip(X, 3), Wb, Rb, bb, c0b, y0b, flip(Yb, 3), Cb, Gb, flip(dYb, 3));
            
            % Concatenate outputs
            dX = dXf + flip(dXb, 3);
            needsWeightGradients = nargout > 1;
            if needsWeightGradients
                % Concatenate weights derivatives
                [dWf, dRf, dBf] = deal( dWeightsf{:} );
                [dWb, dRb, dBb] = deal( dWeightsb{:} );
                dW = cat(1, dWf, dWb);
                dR = cat(1, dRf, dRb);
                dB = cat(1, dBf, dBb);
            end
        end
    end
end

function [Wf, Rf, bf, Wb, Rb, bb] = iSplitWeights(W, R, b)
[Wf, Wb] = iSplitAcrossFirstDimension( W );
[Rf, Rb] = iSplitAcrossFirstDimension( R );
[bf, bb] = iSplitAcrossFirstDimension( b );
end

function [c0f, y0f, c0b, y0b] = iSplitStates(c0, y0)
[c0f, c0b] = iSplitAcrossFirstDimension( c0 );
[y0f, y0b] = iSplitAcrossFirstDimension( y0 );
end

function [Yf, Cf, Gf, dYf, Yb, Cb, Gb, dYb] = iSplitForwardOutputs(Y, C, G, dY)
[Yf, Yb] = iSplitAcrossFirstDimension( Y );
[Cf, Cb] = iSplitAcrossFirstDimension( C );
[Gf, Gb] = iSplitAcrossFirstDimension( G );
[dYf, dYb] = iSplitAcrossFirstDimension( dY );
end

function [Zf, Zb] = iSplitAcrossFirstDimension( Z )
H = 0.5*size(Z, 1);
fInd = 1:H;
bInd = H + fInd;
Zf = Z(fInd, :, :);
Zb = Z(bInd, :, :);
end
classdef BiLSTMHostReturnLastStrategy < nnet.internal.cnn.layer.util.ExecutionStrategy
    % BiLSTMHostReturnLastStrategy   Execution strategy for running BiLSTM on the host
    
    %   Copyright 2017 The MathWorks, Inc.
    
    methods
        function [Y, memory] = forward(~, X, W, R, b, c0, y0)
            % Split data into forward/backward sequences
            [Wf, Rf, bf, Wb, Rb, bb] = iSplitWeights(W, R, b);
            [c0f, y0f, c0b, y0b] = iSplitStates(c0, y0);
            
            % Forward sequence
            [Yf, Cf, Gf] = nnet.internal.cnnhost.lstmForwardReturnLast(X, Wf, Rf, bf, c0f, y0f);
            
            % Backward sequence
            [Yb, Cb, Gb] = nnet.internal.cnnhost.lstmForward(flip(X, 3), Wb, Rb, bb, c0b, y0b);
            Yb = Yb(:, :, 1);
            
            % Concatenate outputs
            Y = cat(1, Yf, Yb);
            C = cat(1, Cf, Cb);
            G = cat(1, Gf, Gb);
            
            % Allocate memory
            memory.CellState = C;
            memory.HiddenState = cat(1, Yf(:, :, end), Yb(:, :, end));
            memory.Gates = G;
        end
        
        function [dX, dW, dR, db] = backward(~, X, W, R, b, c0, y0, Y, memory, dY)
            % Split data into forward/backward sequence components
            [Wf, Rf, bf, Wb, Rb, bb] = iSplitWeights(W, R, b);
            [c0f, y0f, c0b, y0b, H] = iSplitStates(c0, y0);
            C = memory.CellState;
            G = memory.Gates;
            [Yf, Cf, Gf, dYf, ~, Cb, Gb, dYb] = iSplitForwardOutputs(Y, C, G, dY);
            
            % Forward sequence
            [dXf, dWeightsf{1:(nargout-1)}] =  nnet.internal.cnnhost.lstmBackwardReturnLast(X, Wf, Rf, bf, c0f, y0f, Yf, Cf, Gf, dYf);
            
            % Backward sequence
            Ybfull = tanh( Cb ).*Gb(1+3*H:4*H, :, :);
            dYbfull = zeros( size(Ybfull), 'like', Ybfull );
            dYbfull(:, :, 1) = dYb;
            [dXb, dWeightsb{1:(nargout-1)}] =  nnet.internal.cnnhost.lstmBackward(flip(X, 3), Wb, Rb, bb, c0b, y0b, Ybfull, Cb, Gb, dYbfull);
            
            % Add data derivatives
            dX = dXf + flip(dXb, 3);
            
            needsWeightGradients = nargout > 1;
            if needsWeightGradients
                % Concatenate weights derivatives
                [dWf, dRf, dbf] = deal( dWeightsf{:} );
                [dWb, dRb, dbb] = deal( dWeightsb{:} );
                dW = cat(1, dWf, dWb);
                dR = cat(1, dRf, dRb);
                db = cat(1, dbf, dbb);
            end
        end
    end
end

function [Wf, Rf, bf, Wb, Rb, bb] = iSplitWeights(W, R, b)
[Wf, Wb] = iSplitAcrossFirstDimension( W );
[Rf, Rb] = iSplitAcrossFirstDimension( R );
[bf, bb] = iSplitAcrossFirstDimension( b );
end

function [c0f, y0f, c0b, y0b, H] = iSplitStates(c0, y0)
[c0f, c0b, H] = iSplitAcrossFirstDimension( c0 );
[y0f, y0b] = iSplitAcrossFirstDimension( y0 );
end

function [Yf, Cf, Gf, dYf, Yb, Cb, Gb, dYb] = iSplitForwardOutputs(Y, C, G, dY)
[Yf, Yb] = iSplitAcrossFirstDimension( Y );
[Cf, Cb] = iSplitAcrossFirstDimension( C );
[Gf, Gb] = iSplitAcrossFirstDimension( G );
[dYf, dYb] = iSplitAcrossFirstDimension( dY );
end

function [Zf, Zb, H] = iSplitAcrossFirstDimension( Z )
H = 0.5*size(Z, 1);
fInd = 1:H;
bInd = H + fInd;
Zf = Z(fInd, :, :);
Zb = Z(bInd, :, :);
end
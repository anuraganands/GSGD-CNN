classdef LSTMWeightsConverter
% LSTMWeightsConverter  Converts weights and biases for an LSTM layer
% between the separate format and the cuDNN packed format.
%
%   Separate format:
%   W - Input weights         (4*H)x(D) matrix
%   R - Recurrent weights     (4*H)x(H) matrix
%   b - Bias                  (4*H)x(1) vector
%
%   Packed format:
%   Wcudnn - Output weights   (1x1xP) array
%
% For cuDNN 5 P is numel(W)+numel(R)+(2*numel(b)).

%   Copyright 2017 The MathWorks, Inc.

    methods( Static )
        
        function Wcudnn = toCudnn(W, R, b)
            Wt = W';
            Rt = R';
            Wcudnn = [Wt(:); Rt(:); b(:); zeros(numel(b),1,'like',b)];
            Wcudnn = reshape(Wcudnn, 1, 1, []);
        end
        
        function [W, R, b] = fromCudnn(Wcudnn, H, D)
        % fromCudnn  Converts cuDNN parameter vector to separate weights
        % and biases, using the Hidden Size H and the Input Size D
            % Indices of matrices and biases in the param vector
            WInd = 1:(4*H*D);
            RInd = ( 1:(4*H*H) ) + WInd(end);
            bInputInd = ( 1:(4*H) ) + RInd(end);
            bRecurrentInd = ( 1:(4*H) ) + bInputInd(end);
            
            % Get the matrices. Cudnn weights are transposed wrt NNT's
            W = reshape( Wcudnn(WInd), D, 4*H )';
            R = reshape( Wcudnn(RInd), H, 4*H )';
            
            % Get the biases. cuDNN uses separate bias for inputs weights
            % and recurrent weights, so they need to be added
            b = reshape( Wcudnn(bInputInd) + Wcudnn(bRecurrentInd), [], 1 );
        end
        
        function [dW, dR, db] = fromCudnnDerivative(dWcudnn, H, D)
        % fromCudnn  Converts cuDNN parameter derivative vector to separate
        % weights and bias derivatives, using the Hidden Size H and the
        % Input Size D. For the derivatives, the resultant biases of the
        % fromCudnn() calculation must be halved
            % Call fromCudnn() method
            [dW, dR, db] = nnet.internal.cnngpu.util.LSTMWeightsConverter.fromCudnn(dWcudnn, H, D);
            db = 0.5.*db;
        end
        
    end

end

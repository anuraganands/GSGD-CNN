classdef BiLSTMWeightsConverter
% BiLSTMWeightsConverter  Converts weights and biases for a BiLSTM layer
% between the separate format or combined format, and the cuDNN packed
% format.
%
%   Separate format:
%   Wf - Forward input weights         (4*H)x(D) matrix
%   Rf - Forward recurrent weights     (4*H)x(H) matrix
%   bf - Forward bias                  (4*H)x(1) vector
%   Wb - Backward input weights        (4*H)x(D) matrix
%   Rb - Backward recurrent weights    (4*H)x(H) matrix
%   bb - Backward bias                 (4*H)x(1) vector
%
%   Forward/backward combined format:
%   W - Forward/backward input weights         (8*H)x(D) matrix
%   R - Forward/backward recurrent weights     (8*H)x(H) matrix
%   b - Forward/backward bias                  (8*H)x(1) vector
%
%   Packed format:
%   Wcudnn - Output weights   (1x1xP) array
%
% For cuDNN 5 P is 2*(numel(Wf)+numel(Rf)+(2*numel(bf))).

%   Copyright 2017 The MathWorks, Inc.

    methods( Static )
        function Wcudnn = toCudnn( varargin )
            if nargin == 6
                % Separate format
                [Wf, Rf, bf, Wb, Rb, bb] = varargin{:};
            elseif nargin == 3
                % Forward/backward combined format
                [W, R, b] = varargin{:};
                [Wf, Rf, bf, Wb, Rb, bb] = iSplitWeights(W, R, b);
            end
            % cuDNN format
            Wcudnn = iToCudnn(Wf, Rf, bf, Wb, Rb, bb);
        end
        
        function varargout = fromCudnn(Wcudnn, H, D)
            % fromCudnn  Converts cuDNN parameter vector to separate
            % weights and biases, using the Hidden Size H and the Input
            % Size D
            [Wf, Rf, bf, Wb, Rb, bb] = iFromCudnn(Wcudnn, H, D);
            if nargout == 6
                % Separate format
                varargout(1:6) = {Wf, Rf, bf, Wb, Rb, bb};
            elseif nargout == 3
                % Forward/backward combined format
                [W, R, b] = iCombineWeights(Wf, Rf, bf, Wb, Rb, bb);
                varargout(1:3) = {W, R, b};
            end
        end
        
        function varargout = fromCudnnDerivative(dWcudnn, H, D)
            % fromCudnnDerivative  Converts cuDNN parameter derivative
            % vector to separate weights and bias derivatives, using the
            % Hidden Size H and the Input Size D. For the derivatives, the
            % resultant biases of the fromCudnn() calculation must be
            % halved
            % Call fromCudnn() method
            [dWf, dRf, dbf, dWb, dRb, dbb] = nnet.internal.cnngpu.util.BiLSTMWeightsConverter.fromCudnn(dWcudnn, H, D);
            dbf = 0.5.*dbf;
            dbb = 0.5.*dbb;
            if nargout == 6
                % Separate format
                varargout(1:6) = {dWf, dRf, dbf, dWb, dRb, dbb};
            elseif nargout == 3
                % Forward/backward combined format
                [dW, dR, db] = iCombineWeights(dWf, dRf, dbf, dWb, dRb, dbb);
                varargout(1:3) = {dW, dR, db};
            end
        end
    end
end

function [Wf, Rf, bf, Wb, Rb, bb] = iSplitWeights(W, R, b)
H = size( R, 2 );
[fInd, bInd] = nnet.internal.cnn.util.forwardBackwardSequenceIndices( H );
Wf = W(fInd, :);
Rf = R(fInd, :);
bf = b(fInd, :);
Wb = W(bInd, :);
Rb = R(bInd, :);
bb = b(bInd, :);
end

function [W, R, b] = iCombineWeights(Wf, Rf, bf, Wb, Rb, bb)
W = cat(1, Wf, Wb);
R = cat(1, Rf, Rb);
b = cat(1, bf, bb);
end

function Wcudnn = iToCudnn(Wf, Rf, bf, Wb, Rb, bb)
Wtf = Wf';
Rtf = Rf';
Wtb = Wb';
Rtb = Rb';
Wcudnn = [ Wtf(:); Rtf(:); ...
    Wtb(:); Rtb(:); ...
    bf(:); zeros(numel(bf),1,'like',bf); ...
    bb(:); zeros(numel(bb),1,'like',bb) ];
Wcudnn = reshape(Wcudnn, 1, 1, []);
end

function [Wf, Rf, bf, Wb, Rb, bb] = iFromCudnn(Wcudnn, H, D)
% Specify weights indices
WfInd = 1:(4*H*D);
RfInd = ( 1:(4*H*H) ) + WfInd(end);
WbInd = ( 1:(4*H*D) ) + RfInd(end);
RbInd = ( 1:(4*H*H) ) + WbInd(end);
bfInputInd = ( 1:(4*H) ) + RbInd(end);
bfRecurrentInd = ( 1:(4*H) ) + bfInputInd(end);
bbInputInd = ( 1:(4*H) ) + bfRecurrentInd(end);
bbRecurrentInd = ( 1:(4*H) ) + bbInputInd(end);

% Get the matrices. Cudnn weights are transposed wrt NNT's
Wf = reshape( Wcudnn(WfInd), D, 4*H )';
Rf = reshape( Wcudnn(RfInd), H, 4*H )';
Wb = reshape( Wcudnn(WbInd), D, 4*H )';
Rb = reshape( Wcudnn(RbInd), H, 4*H )';

% Get the biases. cuDNN uses separate bias for inputs weights
% and recurrent weights, so they need to be added
bf = reshape( Wcudnn(bfInputInd) + Wcudnn(bfRecurrentInd), [], 1 );
bb = reshape( Wcudnn(bbInputInd) + Wcudnn(bbRecurrentInd), [], 1 );
end
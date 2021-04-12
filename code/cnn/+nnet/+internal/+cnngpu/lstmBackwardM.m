function [dX, dW, dR, db] = lstmBackwardM(X, W, R, b, c0, y0, Y, C, G, dZ)
% lstmBackwardM   Propagate Long Short-Term Memory layer backwards on the
% GPU with M-code
%   [dX, dW, dR, db] = lstmBackwardM(X, W, R, b, c0, y0, Y, C, G, dZ)
%   computes the backward propagation of the Long Short Term Memory layer
%   using input data X, input weights W, recurrent weights R, bias term b,
%   initial cell state c0, initial hidden units y0, output data Y, cell
%   state C, layer gates G and output derivative dZ.
%
%   Definitions:
%   D := Number of dimensions of the input data
%   N := Number of input observations (mini-batch size)
%   S := Sequence length
%   H := Hidden units size
%
%   Inputs:
%   X - Layer input               (D)x(N)x(S) array
%   W - Input weights             (4*H)x(D) matrix
%   R - Recurrent weights         (4*H)x(H) matrix
%   b - Bias                      (4*H)x(1) vector
%   c0 - Initial cell state       (H)x(1) vector
%   y0 - Initial hidden units     (H)x(1) vector
%   Y - Layer output              (H)x(N)x(S) array
%   C - Memory cell               (H)x(N) matrix
%   G - Layer gates               (4*H)X(N) matrix
%   dZ - Next layer derivative    (H)x(N)x(S) array
%
%   Outputs:
%   dX - Input data derivative            (D)x(N)x(S) array
%   dW - Input weights derivative         (4*H)x(D) array
%   dR - Recurrent weights derivative     (4*H)x(H) array
%   db - Bias derivative                  (4*H)x(1) vector

%   Copyright 2017 The MathWorks, Inc.

% Input dimensionality
[~, N, S] = size(X);

% Make sure X is on the GPU
X = gpuArray( X );

% Indexing helpers
H = size(R, 2);
[zInd, iInd, fInd, oInd] = nnet.internal.cnn.util.gateIndices(H);
ifoInd = [iInd fInd oInd];

% Initialize LSTM deltas
dG = zeros(4*H, N, 'like', X);
dX = zeros(size(X), 'like', X);
dW = zeros(size(W), 'like', X);
dR = zeros(size(R), 'like', X);
db = zeros(size(b), 'like', X);

% If y0 is passed as a vector, expand over batch dimension
y0 = iExpandBatchDimension(y0, N);

if S > 1
    %%% Backpropagate at tt = S
    % Layer output derivative
    dY = dZ(:, :, S);
    
    % Tanh activation of the cell state
    tanhC = nnet.internal.cnngpu.tanhForward( C );
    
    % Determine the gates
    G = W*X(:, :, S) + R*Y(:, :, S - 1) + b;
    G = iNonlinearActivations( G, zInd, ifoInd );
    
    % Cell state
    C = (C - G(zInd, :) .* G(iInd, :)) ./ G(fInd, :);
    
    % Output gate derivative
    dG(oInd, :) = iOutputGateDerivative(dY, tanhC, G(oInd, :));
    
    % Cell derivative
    dC = dY.*G(oInd, :).*( 1 - tanhC.^2 );
    
    % Forget gate derivative
    dG(fInd, :) = iForgetGateDerivative(dC, C, G(fInd, :));
    
    % Input gate derivative
    dG(iInd, :) = iInputGateDerivative(dC, G(zInd, :), G(iInd, :));
    
    % Layer input derivative
    dG(zInd, :) = iLayerInputDerivative(dC, G(iInd, :), G(zInd, :));
    
    % Input data derivative
    dX(:, :, S) = iInputDataDerivative(W(zInd, :), dG(zInd, :), ...
        W(iInd, :), dG(iInd, :), ...
        W(fInd, :), dG(fInd, :), ...
        W(oInd, :), dG(oInd, :));
    
    % Input weights derivative
    dW = iInputWeightsDerivative(dW, dG, X(:, :, S));
    
    % Recurrent weights derivative
    dR = iRecurrentWeightsDerivative(dR, dG, Y(:, :, S - 1));
    
    % Bias derivative
    db = iBiasDerivative(db, dG);
    
    %%% Backpropagation through time over tt = S-1, ..., 2
    for tt = (S-1):-1:2
        % Layer output derivative
        dY = iOutputDerivative(dZ(:, :, tt), R(zInd, :), dG(zInd, :), ...
            R(iInd, :), dG(iInd, :), ...
            R(fInd, :), dG(fInd, :), ...
            R(oInd, :), dG(oInd, :));
        
        % Tanh activation of the cell state
        tanhC = nnet.internal.cnngpu.tanhForward( C );
        
        % Store G(fInd, :, tt + 1)
        Gfttp = G(fInd, :);
        
        % Determine the gates
        G = W*X(:, :, tt) + R*Y(:, :, tt - 1) + b;
        G = iNonlinearActivations( G, zInd, ifoInd );
        
        % Cell state
        C = (C - G(zInd, :) .* G(iInd, :)) ./ G(fInd, :);
        
        % Output gate derivative
        dG(oInd, :) = iOutputGateDerivative(dY, tanhC, G(oInd, :));
        
        % Cell derivative
        dC = iCellDerivative(dY, G(oInd, :), tanhC, dC, Gfttp);
        
        % Forget gate derivative
        dG(fInd, :) = iForgetGateDerivative(dC, C, G(fInd, :));
        
        % Input gate derivative
        dG(iInd, :) = iInputGateDerivative(dC, G(zInd, :), G(iInd, :));
        
        % Layer input derivative
        dG(zInd, :) = iLayerInputDerivative(dC, G(iInd, :), G(zInd, :));
        
        % Input data derivative
        dX(:, :, tt) = iInputDataDerivative(W(zInd, :), dG(zInd, :), ...
            W(iInd, :), dG(iInd, :), ...
            W(fInd, :), dG(fInd, :), ...
            W(oInd, :), dG(oInd, :));
        
        % Input weights derivative
        dW = iInputWeightsDerivative(dW, dG, X(:, :, tt));
        
        % Recurrent weights derivative
        dR = iRecurrentWeightsDerivative(dR, dG, Y(:, :, tt - 1));
        
        % Bias derivative
        db = iBiasDerivative(db, dG);
    end
    
    %%% tt = 1
    % Layer output derivative
    dY = iOutputDerivative(dZ(:, :, 1), R(zInd, :), dG(zInd, :), ...
        R(iInd, :), dG(iInd, :), ...
        R(fInd, :), dG(fInd, :), ...
        R(oInd, :), dG(oInd, :));
    
    % Tanh activation of the cell state
    tanhC = nnet.internal.cnngpu.tanhForward( C );
    
    % Store G(fInd, :, tt + 1)
    Gfttp = G(fInd, :);
    
    % Determine the gates
    G = W*X(:, :, 1) + R*y0 + b;
    G = iNonlinearActivations( G, zInd, ifoInd );
    
    % Output gate derivative
    dG(oInd, :) = iOutputGateDerivative(dY, tanhC, G(oInd, :));
    
    % Cell derivative
    dC = iCellDerivative(dY, G(oInd, :), tanhC, dC, Gfttp);
    
    % Forget gate derivative
    dG(fInd, :) = iForgetGateDerivative(dC, c0, G(fInd, :));
    
    % Input gate derivative
    dG(iInd, :) = iInputGateDerivative(dC, G(zInd, :), G(iInd, :));
    
    % Layer input derivative
    dG(zInd, :) = iLayerInputDerivative(dC, G(iInd, :), G(zInd, :));
    
    % Input data derivative
    dX(:, :, 1) = iInputDataDerivative(W(zInd, :), dG(zInd, :), ...
        W(iInd, :), dG(iInd, :), ...
        W(fInd, :), dG(fInd, :), ...
        W(oInd, :), dG(oInd, :));
    
    % Input weights derivative
    dW = iInputWeightsDerivative(dW, dG, X(:, :, 1));
    
    % Recurrent weights derivative
    dR = iRecurrentWeightsDerivative(dR, dG, y0);
    
    % Bias derivative
    db = iBiasDerivative(db, dG);
    
else
    %%% S = 1
    % Layer output derivative
    dY = dZ;
    
    % Tanh activation of the cell state
    tanhC = nnet.internal.cnngpu.tanhForward( C );
    
    % Determine the gates
    G = W*X(:, :, 1) + R*y0 + b;
    G = iNonlinearActivations( G, zInd, ifoInd );
    
    % Output gate derivative
    dG(oInd, :) = iOutputGateDerivative(dY, tanhC, G(oInd, :));
    
    % Cell derivative
    dC = dY.*G(oInd, :).*( 1 - tanhC.^2 );
    
    % Forget gate derivative
    dG(fInd, :) = iForgetGateDerivative(dC, c0, G(fInd, :));
    
    % Input gate derivative
    dG(iInd, :) = iInputGateDerivative(dC, G(zInd, :), G(iInd, :));
    
    % Layer input derivative
    dG(zInd, :) = iLayerInputDerivative(dC, G(iInd, :), G(zInd, :));
    
    % Input data derivative
    dX(:, :, 1) = iInputDataDerivative(W(zInd, :), dG(zInd, :), ...
        W(iInd, :), dG(iInd, :), ...
        W(fInd, :), dG(fInd, :), ...
        W(oInd, :), dG(oInd, :));
    
    % Input weights derivative
    dW = iInputWeightsDerivative(dW, dG, X(:, :, 1));
    
    % Recurrent weights derivative
    dR = iRecurrentWeightsDerivative(dR, dG, y0);
    
    % Bias derivative
    db = iBiasDerivative(db, dG);
    
end
end

% Expand over batch size
function Y = iExpandBatchDimension(Y, N)
if size(Y, 2) == 1
    Y = repmat(Y, 1, N);
end
end

% Nonlinear activations
function G = iNonlinearActivations( G, zInd, ifoInd )
% Nonlinear gate operations
G(zInd, :) = nnet.internal.cnngpu.tanhForward( G(zInd, :) );
G(ifoInd, :) = nnet.internal.cnngpu.sigmoidForward( G(ifoInd, :) );
end

% Derivatives
function dY = iOutputDerivative(dZ, Rz, dzpp, Ri, dipp, Rf, dfpp, Ro, dopp)
dY = dZ + Rz'*dzpp + Ri'*dipp + Rf'*dfpp + Ro'*dopp;
end

function do = iOutputGateDerivative(dY, tanhC, Go)
do = dY.*tanhC.*Go.*(1 - Go);
end

function dc = iCellDerivative(dY, Go, tanhC, dcpp, Gfpp)
dc = dY.*Go.*( 1 - tanhC.^2 ) + dcpp.*Gfpp;
end

function df = iForgetGateDerivative(dc, Cmm, Gf)
df = dc.*Cmm.*Gf.*(1 - Gf);
end

function di = iInputGateDerivative(dc, Gz, Gi)
di = dc.*Gz.*Gi.*(1 - Gi);
end

function dz = iLayerInputDerivative(dc, Gi, Gz)
dz = dc.*Gi.*(1 - Gz.^2);
end

function dX = iInputDataDerivative(Wz, dz, Wi, di, Wf, df, Wo, do)
dX = Wz'*dz + Wi'*di + Wf'*df + Wo'*do;
end

function dW = iInputWeightsDerivative(dW, dG, X)
dW = dW + dG*X';
end

function dR = iRecurrentWeightsDerivative(dR, dG, Ymm)
dR = dR + dG*Ymm';
end

function db = iBiasDerivative(db, dG)
db = db + sum(dG, 2);
end
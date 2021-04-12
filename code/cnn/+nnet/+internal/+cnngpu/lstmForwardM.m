function [Y, C, G] = lstmForwardM(X, W, R, b, c0, y0)
% lstmForwardM   Propagate Long Short-Term Memory layer forwards
% on the GPU with M-code
%   [Y, C, G] = lstmForwardM(X, W, R, b, c0, y0) computes the forward
%   propagation of the Long Short-Term Memory layer using input data X,
%   input weights W, recurrent weights R, bias term b and initial cell
%   state c0, and initial hidden units y0. Only the final hidden state is
%   returned.
%
%   Definitions:
%   D := Number of dimensions of the input data
%   N := Number of input observations (mini-batch size)
%   S := Sequence length
%   H := Hidden units size
%
%   Inputs:
%   X - Input data            (D)x(N)x(S) array
%   W - Input weights         (4*H)x(D) matrix
%   R - Recurrent weights     (4*H)x(H) matrix
%   b - Bias                  (4*H)x(1) vector
%   c0 - Initial cell state   (H)x(1) vector
%   y0 - Initial hidden units (H)x(1) vector
%
%   Outputs:
%   Y - Output                (H)x(N)x(S) array
%   C - Cell state            (H)x(N) array
%   G - Gates                 (4*H)x(N) array

%   Copyright 2017 The MathWorks, Inc.

% Determine dimensions
[~, N, S] = size(X);
H = size(R, 2);

% Make sure X is on the GPU
X = gpuArray(X);

% Pre-allocate output
Y = zeros(H, N, S, 'like', X);

% Indexing helpers
[zInd, iInd, fInd, oInd] = nnet.internal.cnn.util.gateIndices(H);
ifoInd = [iInd fInd oInd];

C = c0;
% Forward propagate through time
for tt = 1:S
    % Linear gate operations
    if tt == 1
        G = W*X(:, :, tt) + R*y0 + b;
    else
        G = W*X(:, :, tt) + R*Y(:, :, tt - 1) + b;
    end
    
    % Nonlinear gate operations
    G = iNonlinearActivations( G, zInd, ifoInd );
    
    % Cell state update
    C = G(zInd, :) .* G(iInd, :) + G(fInd, :) .* C;
        
    % Layer output
    Y(:, :, tt) = nnet.internal.cnngpu.tanhForward( C ) .* G(oInd, :);
end

end

function G = iNonlinearActivations( G, zInd, ifoInd )
% Nonlinear gate operations
G(zInd, :) = nnet.internal.cnngpu.tanhForward( G(zInd, :) );
G(ifoInd, :) = nnet.internal.cnngpu.sigmoidForward( G(ifoInd, :) );
end

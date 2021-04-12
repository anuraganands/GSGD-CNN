function [Y, C, G] = lstmForward(X, W, R, b, c0, y0)
% lstmForward   Propagate Long Short Term Memory layer forwards on the host
%   [Y, C, G] = lstmForward(X, W, R, b, c0, y0) computes the forward
%   propagation of the Long Short Term Memory layer using input data X,
%   input weights W, recurrent weights R, bias term b and initial cell
%   state c0, and initial hidden units y0.
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
%   C - Cell state            (H)x(N)x(S) array
%   G - Gates                 (4*H)x(N)x(S) array

%   Copyright 2017 The MathWorks, Inc.

% Determine dimensions
[~, N, S] = size(X);
H = size(R, 2);

% Pre-allocate output, gate vectors and cell state
G = zeros(4*H, N, S, 'like', X);
Y = zeros(H, N, S, 'like', X);
C = zeros(H, N, S, 'like', X);

% Indexing helpers
[zInd, iInd, fInd, oInd] = nnet.internal.cnn.util.gateIndices(H);
ifoInd = [iInd fInd oInd];

% Forward propagate through time
for tt = 1:S
    if tt == 1
        % Linear gate operations
        G(:, :, tt) = W*X(:, :, tt) + R*y0 + b;
        
        % Nonlinear gate operations
        G = iNonlinearActivations( G, zInd, ifoInd, tt );
        
        % Cell state update
        C(:, :, tt) = G(zInd, :, tt) .*  G(iInd, :, tt) + ...
            G(fInd, :, tt) .* c0;
       
    else
        % Linear gate operations
        G(:, :, tt) = W*X(:, :, tt) + R*Y(:, :, tt - 1) + b;
        
        % Nonlinear gate operations
        G = iNonlinearActivations( G, zInd, ifoInd, tt );
        
        % Cell state update
        C(:, :, tt) = G(zInd, :, tt) .*  G(iInd, :, tt) + ...
            G(fInd, :, tt) .* C(:, :, tt - 1);
        
    end
    
    % Layer output
    Y(:, :, tt) = nnet.internal.cnnhost.tanhForward( C(:, :, tt) ) .* G(oInd, :, tt);
        
end

end

function G = iNonlinearActivations( G, zInd, ifoInd, tt )
% Nonlinear gate operations
G(zInd, :, tt) = nnet.internal.cnnhost.tanhForward( G(zInd, :, tt) );
G(ifoInd, :, tt) = nnet.internal.cnnhost.sigmoidForward( G(ifoInd, :, tt) );
end

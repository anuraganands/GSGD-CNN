function dLdIn = numericDiff5Point(fcn, in, dLdOut)
% numericDiff5Point   Five-point numerical derivative function. 
% The derivative of function 'fcn' is calculated with respect to the 
% input argument 'in'.
%
% Inputs:
%   fcn    - Function handle that fulfills out = fcn(in).
%            If the derivative should be computed w.r.t. the data variable 
%            'X' then fcn is typically @(X)layer.forward(X). 
%            If the derivative should be computed w.r.t. a learnable 
%            parameter 'W' then fcn needs to set the property 'W' before 
%            forwarding data, e.g. 
%            fcn = @(W) updateWeightsAndForward(layer,W,X);
%            function out = updateWeightsAndForward(layer,W,X)
%               layer.Weights = W;
%               out = layer.forward(X);
%            end
%   in     - Value at which the derivative should be computed 
%

    %   Copyright 2017 The MathWorks, Inc.
    
% Compute the numerical gradient dLdIn using the following epsilon perturbation
epsilon = eps(class(dLdOut))^(1/3);

% Deal with inputs that are all zeros
M = max(abs(in(:)));
if M > 0
    h = M * epsilon; % Step size should match data
else
    h = epsilon;
end

N = numel(in);
dLdIn = zeros( N, 1, 'like', in );
for ii = 1:N
    % Save original value
    x = in(ii);
    
    % Perturb x for finite differences
    xph = x + h;
    xmh = x - h;
    xp2h = x + (2*h);
    xm2h = x - (2*h);
    
    % Correct h (deltas between sample points) based on actual
    % difference. See
    % https://en.wikipedia.org/wiki/Numerical_differentiation#Practical_considerations_using_floating_point_arithmetic
    hp = xph - x;
    hm = x - xmh;
    h2p = xp2h - x;
    h2m = x - xm2h;
    h12 = 8*(hp + hm) - (h2p + h2m);
    
    % 5-point method for numerical derivative
    % x + 2h
    in(ii) = xp2h;
    outputOffset = fcn( in );
    dOutdInII = - outputOffset(:);
    
    % x + h
    in(ii) = xph;
    outputOffset = fcn( in );
    dOutdInII = dOutdInII + 8*outputOffset(:);
    
    % x - h
    in(ii) = xmh;
    outputOffset = fcn( in );
    dOutdInII = dOutdInII - 8*outputOffset(:);
    
    % x - 2h
    in(ii) = xm2h;
    outputOffset = fcn( in );
    dOutdInII = dOutdInII + outputOffset(:);
    
    % sum / 12h
    dOutdInII = dOutdInII ./ h12;
    
    % Reset
    in(ii) = x;

    % Apply chain rule: dLdIn(ii) = dLdOut * dOutdInII = sum(dOutdIn .* dLdOut(:))
    dLdIn(ii) = sum(dOutdInII .* dLdOut(:));
end

dLdIn = reshape( dLdIn, size(in) );
end
function varargout = fevalUserCode(F, varargin)
% fevalUserCode  Wrap calls into user code to trap exceptions and wrap them
% so that the stack is correctly displayed

% Copyright 2017, The MathWorks Inc.

try
    [varargout{1:nargout}] = feval(F, varargin{:});
catch exception
    throw( nnet.internal.cnn.util.CNNException.hBuildUserException( exception ) );
end
end

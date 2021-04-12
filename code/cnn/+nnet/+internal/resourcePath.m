function fullresourcefile = resourcePath(resourceFilename)
% resourcePath   Returns a fullpath to a resource file

%   RESOURCEFILENAME is the name of a resource file in the resources
%   directory for the Convolutional Neural Networks.

%   Copyright 2017 The MathWorks, Inc.

pathstr = fileparts(mfilename('fullpath'));
fullresourcefile = fullfile(pathstr, 'resources', resourceFilename);
end



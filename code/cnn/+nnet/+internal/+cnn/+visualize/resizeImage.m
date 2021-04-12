function B = resizeImage(A, outputSize)
% Resize image A using INTERP2.
%
% Comparable results to 
%
%   B = imresize(A, outputSize,'bilinear','Antialiasing',false);

% interp2 only supports float. 
assert(isfloat(A), 'Input must be double or single');

assert(length(outputSize) == 2, 'output size must be [height width]');

[M,N,~,~] = size(A);

sx = outputSize(2)/ N;
sy = outputSize(1)/ M;

% Map pixel from output space (u,v) to input space (x,y)
u = 1:outputSize(2);
v = 1:outputSize(1);

x = u/sx + 0.5 * (1 - 1/sx);
y = v/sy + 0.5 * (1 - 1/sy);

[x,y] = meshgrid(x,y);

fillValue = 0;

B = nnet.internal.cnn.visualize.interp2d(A, x, y, fillValue);

classdef LaplacianPyramid
    % LaplacianPyramid   Class to compute the Laplacian pyramid and
    % Laplacian pyramid normalization of an image.
    %
    %   An image can be compressed into a Laplacian pyramid, and then the
    %   pyramid can be merged to give back the original image. In Laplacian
    %   pyramid normalization, we normalize each image in the pyramid
    %   before merging.
    
    %   Copyright 2016 The MathWorks, Inc.
    
    methods(Static)
        function L = laplacianPyramid(img, N)
            % This function computes the Laplacian pyramid for N steps.
            % Notation as in Burt--Adelson, with indices starting from 1
            % rather than 0. Let g_1 be the original image.
                   
            % Then define g_i = REDUCE(g_{i-1}) for i = 1, ..., N+1. Define
            % L_i = g_i - EXPAND(g_{i+1}) for i = 1, ..., N. Define L_{N+1}
            % = g_{N+1}.                       
            
            L = cell(1,N);                       
            
            % Initialize the current g. This is g_1.
            currentg = img;
            
            for i = 1:N                               
                % nextg is g_{i+1} = REDUCE(g_i)
                nextg = iReduce(currentg);
                
                % expandednextg is EXPAND(g_{i+1})
                expandednextg = iExpand(nextg);
                
                % Resize expandedg1 to be the same size as g_i -- just to
                % make sure we can subtract currentg and expandednextg.
                expandednextg = iResizeImage(expandednextg, [size(currentg, 1), size(currentg, 2)]);                
                
                % The ith entry in the pyramid
                L{i} = currentg - expandednextg;
                
                % Update g_i
                currentg = nextg;
            end
            
            % The last entry in the pyramid is g_{N+1} = REDUCE(g_N).
            L{N+1} = nextg;
        end
        
        function mergedImg = merge(L)
            % Merge the Laplacian pyramid to recover the original image.
            % This is achieved with: g_{N+1} = L_{N+1}.
            % g_i = L_i + EXPAND(g_{i+1}) for each i. Then g_1 is the
            % original image.
            N = numel(L) - 1;
            
            % img is g_{N+1} = L_{N+1}.
            img = L{N+1};
            
            for i = N:-1:1
                % expandedImg is EXPAND(g_{i+1})
                expandedImg = iExpand(img);
                
                % Resize the image to make sure we can subtract the two images.
                expandedImg = iResizeImage(expandedImg, [size(L{i}, 1), size(L{i}, 2)]);
                
                % img is g_i = L_i + EXPAND(g_{i+1}).
                img = expandedImg + L{i};
            end
            
            % Finally, mergedImg is g_1.
            mergedImg = img;
        end
        
        function normalisedImg = normaliseImage(img)
            % img is a 4D array of 3D images, indexed by the fourth dimension.
            % normalizeImage normalises each image so that its L2 norm is
            % 1.
            size4D = size(img, 4);
            spatialSize = size(img, 1) * size(img, 2) * size(img, 3);
            norms = sqrt(sum(reshape(img, [], size4D).^2) / spatialSize);
            norms = shiftdim(norms, -2);
            normalisedImg = img ./ max(norms, 1e-8);
        end
        
        function laplacianNormalizedImage = laplacianNormalizedImage(img, N)
            % laplacianNormalizedImage takes in an image, and a number of
            % steps to compute the Laplacian pyramid for.
            % We then normalize each entry in the pyramid to have norm 1.
            % This boosts the lower frequencies.
                                 
            L = nnet.internal.cnn.visualize.LaplacianPyramid.laplacianPyramid(img, N);
            for i=1:numel(L)
                L{i} = nnet.internal.cnn.visualize.LaplacianPyramid.normaliseImage(L{i});
            end
            laplacianNormalizedImage = nnet.internal.cnn.visualize.LaplacianPyramid.merge(L);
          
        end
    end
end

%--------------------------------------------------------------------------
function out = iResizeImage(in, sz)
out = nnet.internal.cnn.visualize.resizeImage(in, sz);
end

%--------------------------------------------------------------------------
function h = iFilterCoefficients()
a = 0.375;
h =  [1/4-a/2 1/4 a 1/4 1/4-a/2];
end

%--------------------------------------------------------------------------
function g = iExpand(inImg)
h = iFilterCoefficients();
[M,N,C,numObservations] = size(inImg);

% upsample image
outImg = zeros([2*M 2*N C numObservations],'like', inImg);
outImg(1:2:end,1:2:end, :,:) = inImg; 
outImg = outImg(1:end-1,1:end-1,:,:);

sz = length(h) - floor((length(h)+1)/2);

% allocate output buffer.
g = zeros(size(outImg), 'like', outImg);

% pad and filter image.
outImg = nnet.internal.cnn.visualize.symmetricPad(outImg, [sz sz] );
for j = 1:numObservations
    for i = 1:C
        % Filter image, keep valid portion. b/c of padding, valid output will be
        % same size as input.
        g(:,:,i,j) = 4 * conv2(h,h,outImg(:,:,i,j),'valid');
    end
end

end

%--------------------------------------------------------------------------
function outImg = iReduce(inImg)
h = iFilterCoefficients();
% symmetric padding to ensure same output size
sz = length(h) - floor((length(h)+1)/2);

g = zeros(size(inImg),'like',inImg);

inImg = nnet.internal.cnn.visualize.symmetricPad(inImg, [sz sz] );

[~,~,C,numObservations] = size(inImg);
for j = 1:numObservations
    for i = 1:C
        % Filter image, keep valid portion. b/c of padding, valid output will be
        % same size as input.
        g(:,:,i,j) = conv2(h,h,inImg(:,:,i,j),'valid');
    end
end
% sub-sample by 2
outImg = g(1:2:end,1:2:end,:,:);
end

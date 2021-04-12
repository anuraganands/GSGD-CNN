function im = deepDreamImageLaplacianNorm(iVisualNet, ...
        X, numIterations, numOctaves, ...
        octaveScale, tileSize, stepSize, useLaplacian, verbose)
    % deepDreamImageLaplacianNorm computes deep dream images for
    % the specified layer and channels. X is a 4D array with the fourth
    % dimension indexing the channel that is being optimized.
    
    %   Copyright 2016 The MathWorks, Inc.
    
    % Setup reporters and summary for verbose mode.
    [reporter, summary] = iConfigureReporter(verbose);
    
    height = size(X, 1);
    width = size(X, 2);
    reporter.start();
    for octave=1:numOctaves
        
        if octave > 1
            % Up-scale image for next octave
            height = floor(height * octaveScale);
            width  = floor(width  * octaveScale);
            X = nnet.internal.cnn.visualize.resizeImage(X, [height, width]);
        end
        
        for iter=1:numIterations
            [gradient, activations] = nnet.internal.cnn.visualize.TiledGradients.computeTiledGradient(...
                iVisualNet, X, tileSize);
            if useLaplacian
                gradient = iLaplacianNormalizedImage( gradient );
            else
                gradient = iNormalizeGradient(X,gradient);
            end
            
            % Update step.
            X = X + gradient * stepSize;                        
            
            % Display progress.
            summary.update(octave, iter, activations);
            reporter.reportIteration( summary );
        end
        
    end
    reporter.finish( summary );
    
    im = X;
end

function gradient = iLaplacianNormalizedImage(gradient)
numSteps = 4;

gradient = nnet.internal.cnn.visualize.LaplacianPyramid.laplacianNormalizedImage( gradient, numSteps );
end

function gradient = iNormalizeGradient(X,gradient)
gradient = gradient ./ shiftdim(std(reshape(gradient, [], size(X,4))) + 1e-9, -2);
end

function [reporter, summary] = iConfigureReporter(verbose)
summary = nnet.internal.cnn.util.DeepDreamImageSummary;
reporter = nnet.internal.cnn.util.VectorReporter();

if verbose
    progressDisplayer = nnet.internal.cnn.util.ProgressDisplayer(...
        nnet.internal.cnn.util.DeepDreamImageColumnStrategy);
    
    % Display every iteration. The number of iterations is small.
    progressDisplayer.Frequency = 1; 
    
    reporter.add(progressDisplayer);
end
   
end
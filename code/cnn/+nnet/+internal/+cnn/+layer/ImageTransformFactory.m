classdef ImageTransformFactory
    % ImageTransformFactory   ImageTransform factory class
    %   transform = ImageTransformFactory.create(transformType, inputSize)
    % input:
    %   transformType      - the type of transform to create.
    %                         One of:
    %                        'randcrop'
    %                        'randfliplr'
    %                        'zerocenter'
    %                        'none'
    %   inputSize          - image input size
    % output:
    %   transform          - nnet.internal.cnn.layer.ImageTransform
    
    %   Copyright 2015 The MathWorks, Inc.
    
    methods(Static)
        function transform = create(transformType, inputSize)
            % create     Return a nnet.internal.cnn.layer.ImageTransform object
            transform = iDoCreate(transformType,inputSize);
        end
        
        function transform = deserialize( S )
            transform = iDoCreate( S.Type, S.ImageSize );
            transform = iAssignFields( transform, S );
        end
    end
end

function transform = iDoCreate(type,inputSize)
constructors = iConstructors();
transform = constructors.(type)(inputSize);
end

function types = iConstructors()
% Returns a struct containing the constructors
types = struct( ...
    'randcrop', @nnet.internal.cnn.layer.RandomCropImageTransform, ...
    'randfliplr', @nnet.internal.cnn.layer.RandomFliplrImageTransform, ...
    'zerocenter', @nnet.internal.cnn.layer.ZeroCenterImageTransform, ...
    'none', @(~, ~) nnet.internal.cnn.layer.ImageTransform.empty);
end

function transform = iAssignFields( transform, S )
fieldAssigners = iFieldAssigners();
transform = fieldAssigners.(S.Type)(transform, S);
end

function fieldAssigners = iFieldAssigners()
noFields = @(transform, ~) transform;
fieldAssigners = struct( ...
    'randcrop', noFields , ...
    'randfliplr', noFields , ...
    'zerocenter', @iAssignZeroCenterFields, ...
    'none', noFields );
end

function transform = iAssignZeroCenterFields( transform, S )
transform.AverageImage = S.AverageImage;
end

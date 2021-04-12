classdef MiniBatchable < handle
    %MiniBatchable Declares the interface that adds Neural Network Toolbox 
    %   support for training, inference, and validation to Datastore.
    %
    %   MiniBatchable Properties:
    %
    %   MiniBatchSize   -   The number of observations that will be
    %                       returned in each call of the read function. At
    %                       training/inference time, the MiniBatchSize
    %                       property of the MiniBatchDatastore will be set to the
    %                       MiniBatchSize defined in trainingOptions.
    %
    %   NumObservations -   The number of observations in the overall
    %                       data set. This is the number of observations in
    %                       one training epoch.
    %
    %   MiniBatchDatastore Property Attributes:
    %
    %   MiniBatchSize   -   Public, Abstract
    %   NumObservations -   Protected SetAccess, Public ReadAccess, Abstract
    %
    %   When defining a MiniBatchable Datastore, there are additional
    %   requirements for the read method.
    %
    %   1) The first LHS argument of read, data, must be a table.
    %   2) The table returned by read should have MiniBatchSize number
    %   of rows, returning a MiniBatch of data each time read is called.
    %   3) For the last MiniBatch of data in Datastore, if NumObservations is
    %   not cleanly divisible by MiniBatchSize, then read should return the
    %   remaining observations in the Datastore (a partial batch smaller
    %   than MiniBatchSize).
    %
    %   Example implementation of read() for MiniBatchable Datastore:
    %   -------------------------------------------------------------
    %   function [data,info] = read(ds)
    %       batchSize = ds.MiniBatchSize;
    %       X = cell(batchSize,1);
    %       for idx = 1:batchSize
    %           X{idx} = rand(200,200,3);
    %       end
    %       Y = categorical(round(rand(batchSize,1)));
    %       data = table(X,Y);
    %       info = struct([]);
    %   end
    
    %   Copyright 2017 The MathWorks, Inc.
    
    properties(Access = public, Abstract)
        
        %MiniBatchSize - The MiniBatchSize used in training and prediction
        %
        %    MiniBatchSize is a numeric scalar the specifies the number of
        %    observations returned in each call to read. This property is
        %    is set during training and prediction to be equal to the
        %    'MiniBatchSize' Name/Value in trainingOptions as well as the
        %    same name value in the predict, classify, and activations
        %    methods of SeriesNetwork and DAGNetwork.
        MiniBatchSize
        
    end
    
    properties(SetAccess = protected, Abstract)
        
        %NumObservations - The total number of observations contained
        % within the datastore. This number of observations is the length
        % of one training epoch.
        NumObservations
        
    end
            
end
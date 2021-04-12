classdef BackgroundDispatchable < handle
    %BackgroundDispatchable  Declares the interface that adds background
    %dispatch (pre-fetch read) support to MiniBatchable Datastores.
    %
    %   This abstract class is a mixin for MiniBatchable Datastores that adds 
    %   support pre-fetch reading of samples during training and inference. 
    %   This mixin requires Parallel Computing Toolbox.
    %
    %   BackgroundDispatchable Properties:
    %
    %   DispatchInBackground  - Logical scalar that defines whether or not
    %                           a MiniBatchDatastore will queue observations in
    %                           the background during training and
    %                           inference. If the GPU is being used and the
    %                           read() method of Datastore does a lot of
    %                           computational work, setting this property
    %                           to true can be a meaningful performance
    %                           optimization. 
    %  
    %                           Note: dispatching in background requires 
    %                           Parallel Computing Toolbox.
    %
    %                           Default: True
    %
    %   BackgroundDispatchable Property Attributes:
    %
    %   DispatchInBackground -  Public
    %
    %   BackgroundDispatchable Methods:
    %
    %   readByIndex  - Return table containing requested observations
    %                  specified by index.
    %
    %   BackgroundDispatchable Method Attributes:
    %
    %   readByIndex  -    Public, Abstract
    %
    %   The readByIndex method has to be implemented by subclasses of
    %   BackgroundDispatchable.
    %
    %   See also matlab.io.Datastore, matlab.io.datastore.MiniBatchable,
    %   matlab.io.datastore.PartitionableByIndex
    
    %   Copyright 2017 The MathWorks, Inc.
    
    
    properties (Access = public)
       
        %DispatchInBackground - Whether background dispatch is used in
        %training and inference.
        %
        %    DispatchInBackground is a logical scalar that controls whether
        %    observations from a MiniBatchDatastore will be queued
        %    asynchronously in the background during training and inference.
        DispatchInBackground = true
        
    end
    
    methods (Access = public, Abstract)
        
        %readByIndex Return observations from a MiniBatchDatastore
        %specified by index.
        %
        %   [DATA,INFO] = readByIndex(DS,INDICES) returns observations from a
        %   MiniBatchDatastore specified by INDICES. The readByIndex
        %   method should return DATA in the same form as the READ method,
        %   which is a table in MiniBatchable Datastores.
        %
        %   See also matlab.io.Datastore, matlab.io.datastore.MiniBatchable
        [data,info] = readByIndex(ds,indices)
        
    end
     
end
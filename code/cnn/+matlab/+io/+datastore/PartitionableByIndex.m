classdef PartitionableByIndex < handle
    %PARTITIONABLEBYINDEX   Declares the interface that adds parallel training support
    %   to MiniBatchDatastore.
    %
    %   This abstract class is a mixin for subclasses of
    %   matlab.io.MiniBatchDatastore
    %   that adds NNT/PCT parallel training support to the datastore.
    %
    %   PartitionableByIndex Methods:
    %
    %   partitionByIndex    -   Return a new datastore that represents a single
    %                           partitioned part of the original datastore given
    %                           requested indices of elements (observations) in the
    %                           original datastore.
    %
    %   PartitionableByIndex Method Attributes:
    %
    %   partitionByIndex    -    Public, Abstract
    %
    %   This class implements the partitionByIndex method has to be implemented 
    %   by the subclasses derived from the PartitionableByIndex class.
    %
    %   See also matlab.io.Datastore, matlab.io.datastore.MiniBatchable, matlab.io.datastore.Partitionable
    %
    %   Copyright 2017 The MathWorks, Inc.
    
    methods (Access = public, Abstract)
        
        % PARTITIONBYINDEX Return a partitioned part of the datastore
        % described by indices
        %
        % SUBDS = PARTITIONBYINDEX(DS,INDICES) partitions DS into a new
        % datastore containing the observations that correspond to INDICES
        % in the original datastore.
        subds = partitionByIndex(ds,indices);
                
    end
    
end
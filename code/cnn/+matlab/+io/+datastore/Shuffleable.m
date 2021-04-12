classdef Shuffleable < handle
    %SHUFFLEABLE   Declares the interface that adds shuffling support to a
    %datastore.
    %   This abstract class is a mixin for subclasses of matlab.io.Datastore
    %   that adds support for shuffling samples in the datastore in random
    %   order.
    %
    %   Shuffleable Methods:
    %
    %   shuffle         -    Return a new datastore that represents a shuffled
    %                        version of the original datastore.
    %
    %   Shuffleable Method Attributes:
    %
    %   shuffle         -    Public, Abstract
    %
    %   The shuffle method has to be implemented by the subclasses derived
    %   from the Shuffleable class.
    %
    %   See also matlab.io.Datastore, matlab.io.datastore.MiniBatchable.
    
    %   Copyright 2017 The MathWorks, Inc.
    
    methods (Access = public, Abstract)
        
        %SHUFFLE Return a shuffled version of a datastore.
        %
        %   NEWDS = SHUFFLE(DS) returns a randomly shuffled copy of a
        %   datastore.
        %
        %   See also matlab.io.datastore.Shuffleable.
        newds = shuffle(ds)
        
    end
    
    
end
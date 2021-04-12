classdef ImageDatastoreMiniBatchDatastore < ...
        matlab.io.Datastore &...
        matlab.io.datastore.MiniBatchable &...
        matlab.io.datastore.Shuffleable &...
        matlab.io.datastore.BackgroundDispatchable &...
        matlab.io.datastore.PartitionableByIndex &...
        matlab.io.datastore.internal.FourDArrayReadable
    
    % Required Datastore interface
    methods
        
        function self = ImageDatastoreMiniBatchDatastore(imds,miniBatchSize)
            self.imds = copy(imds);
            self.MiniBatchSize = miniBatchSize;
            self.DispatchInBackground = false;           
        end

        function set.MiniBatchSize(self,batchSize)
               self.imds.ReadSize = batchSize;
        end
        
        function batchSize = get.MiniBatchSize(self)
              batchSize = self.imds.ReadSize;
        end

        function numObs = get.NumObservations(self)
            numObs = length(self.imds.Files);
        end

        function [data,info] = read(self)
            [data,info] = self.imds.read();
            
            if(isfield(info,'Label'))
                Y = info.Label;
            else
                Y = [];
            end
            
            % Underlying imageDatastore returns numeric matrix instead of
            % cell in special case of MiniBatchSize/ReadSize = 1.
            if ~iscell(data)
               data = {data}; 
            end
            
            info.Response = Y;
        end
        
        function [data,info] = readByIndex(self,indices)
           subds = partitionByIndex(self,indices);
           data = subds.readall();
           info.Response = subds.imds.Labels;
        end
        
        function subds = partitionByIndex(ds,idx)
            subimds = copy(ds.imds);
            subimds.Files = subimds.Files(idx);
            if ~isempty(ds.imds.Labels)
                subimds.Labels = ds.imds.Labels(idx);
                iSetCategories(subimds,categories(ds.imds.Labels(1)));
            end
            subds = nnet.internal.cnn.ImageDatastoreMiniBatchDatastore(subimds,ds.MiniBatchSize);
        end
        
        function dsnew = shuffle(self)
            if ~isempty(self)
                dsnew = copy(self);
                dsnew.imds = shuffle(dsnew.imds);
            end    
        end
        
        function reset(self)
            self.imds.reset();
        end
        
        function TF = hasdata(self)
            TF = self.imds.hasdata();
        end
        
    end
    
    methods (Hidden)
       
        function frac = progress(self)
            frac = progress(self.imds);
        end
        
    end
    
    properties (Dependent)
        MiniBatchSize
    end
    
    properties (SetAccess = protected, Dependent)
        NumObservations
    end
    
    properties (Access = private)
        imds
    end
    
end

function iSetCategories( ds, cats )
if ~isempty(ds.Files)
        ds.Labels = setcats( ds.Labels, cats );
end
end

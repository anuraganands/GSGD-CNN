function validateMiniBatchDatastore(ds)
% validateMiniBatchDatastore  Validate a MiniBatchDatastore
%
%   validateMiniBatchDatastore(ds) checks the method and property
%   definitions of concrete implementations of the abstract class
%   MiniBatchDatastore for correctness.

%   Copyright 2017 The MathWorks, Inc.


checkNumObservations(ds)
checkMiniBatchSize(ds)
checkResetMethod(ds);
checkHasData(ds);
checkReadMethod(ds);
checkPartitionByIndex(ds);
checkBackgroundDispatchable(ds);
reset(ds);

end

function checkNumObservations(ds)
    validateattributes(ds.NumObservations,{'numeric'},{'positive','integer','scalar'},class(ds),'NumObservations');
end

function checkMiniBatchSize(ds)
    validateattributes(ds.MiniBatchSize,{'numeric'},{'positive','integer','scalar'},class(ds),'MiniBatchSize');
end

function checkResetMethod(ds)
try
   reset(ds);
catch e
    iThrowMBDSException(e,'reset')
end

end

function checkHasData(ds)
try
   TF = hasdata(ds);
   if ~TF
      error(message('nnet_cnn:internal:cnn:MiniBatchDatastore:HasDataReturnsFalse'));
   end
catch e
    iThrowMBDSException(e,'hasdata')
end

end

function checkReadMethod(ds)

if isa(ds,'matlab.io.datastore.internal.FourDArrayReadable')
    return
end

try
   ds.MiniBatchSize = 1;
   d = read(ds);
   if ~istable(d)
       error(message('nnet_cnn:internal:cnn:MiniBatchDatastore:ReadMustReturnTable'));
   end
   if ~isequal(size(d,1),1)
       error(message('nnet_cnn:internal:cnn:MiniBatchDatastore:ReadMustAgreeWithMiniBatchSize'));
   end
catch e
    iThrowMBDSException(e,'read')
end

end

function checkPartitionByIndex(ds)

% Disabling partitionByIndex checking as well because of a testing strategy
% requires the ability for FourDArray to return a LHS argument of a
% different type than the RHS. Disabling the validation here for a
% different reason than the difference in read syntax that is actually
% described by FourDArrayReadable.

if isa(ds,'matlab.io.datastore.PartitionableByIndex')
    try
       dsnew = partitionByIndex(ds,1);
       if ~(isa(dsnew,class(ds)) || isa(ds,'matlab.io.datastore.internal.FourDArrayReadable'))
          error(message('nnet_cnn:internal:cnn:MiniBatchDatastore:PartitionByIndexWrongClass')); 
       end
       if ~isequal(dsnew.NumObservations,1)
          error(message('nnet_cnn:internal:cnn:MiniBatchDatastore:PartitionByIndexWrongNumObservations'));
       end
    catch e
        iThrowMBDSException(e,'partitionByIndex')
    end
end
end

function checkBackgroundDispatchable(ds)

if isa(ds,'matlab.io.datastore.internal.FourDArrayReadable')
    return
end

if isa(ds,'matlab.io.datastore.BackgroundDispatchable')
   try
       d = readByIndex(ds,1);
       if ~istable(d)
           error(message('nnet_cnn:internal:cnn:MiniBatchDatastore:ReadByIndexMustReturnTable'));
       end
      if ~isequal(size(d,1),1)
          error(message('nnet_cnn:internal:cnn:MiniBatchDatastore:ReadByIndexWrongNumObservations')); 
      end
   catch e
       iThrowMBDSException(e,'readByIndex')
   end
end

end

function iThrowMBDSException(exception, methodName)
   
meMBDS = MException('nnet_cnn:internal:cnn:MiniBatchDatastore:badMBDS',...
getString(message('nnet_cnn:internal:cnn:MiniBatchDatastore:badMBDS',...
methodName,'%s',exception.stack(1).line,exception.message)), ...
exception.stack(1).file);

throwAsCaller(meMBDS);

end

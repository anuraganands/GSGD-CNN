function tf = imdsHasCustomReadFcn( imds )
% imdsHasCustomReadFcn  Robustly check whether the user has customized the
% ReadFcn of an ImageDatastore

%   Copyright 2017 The MathWorks, Inc.

persistent imageDatastoreDefaultReadFcnPath
if isempty(imageDatastoreDefaultReadFcnPath)
    parentDirOfImageDatastore = fileparts( which('matlab.io.datastore.ImageDatastore') );
    imageDatastoreDefaultReadFcnPath = fullfile(parentDirOfImageDatastore, 'private', 'readDatastoreImage.m');
end
if isempty(imds) || isempty(imds.ReadFcn)
    tf = false;
else
    fn = functions(imds.ReadFcn);
    tf = ~isequal(fn.file, imageDatastoreDefaultReadFcnPath);
end

end

function ok = canUsePCT()
% canUsePCT   Check that Parallel Computing Toolbox is installed and licensed

%   Copyright 2016 The MathWorks, Inc.

% Checking for installation is expensive, so only do it once
persistent pctInstalled;
if isempty(pctInstalled)
    pctInstalled = exist('gpuArray', 'file') == 2;
end
 
% Check the license every time as it may have changed
pctLicensed = license('test', 'Distrib_Computing_Toolbox');
 
% Now see if everything is OK with the hardware
ok = pctInstalled && pctLicensed;

end
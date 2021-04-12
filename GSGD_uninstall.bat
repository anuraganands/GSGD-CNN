@echo off
if not "%1"=="am_admin" (powershell start -verb runas '%0' am_admin & exit /b)

setlocal enabledelayedexpansion
echo "Select MATLAB toolbox folder that conatins CNN-GSGD"
set "psCommand="(new-object -COM 'Shell.Application')^
.BrowseForFolder(0,'Please choose a folder.',0,0).self.path""
for /f "usebackq delims=" %%I in (`powershell %psCommand%`) do set "folder=%%I"

set "gsgd=\cnn"
set "cnn_orignal=\cnn_original"
set "gsgdFolder=%folder%%gsgd%"
set "cnnOriginalFolder=%folder%%cnn_orignal%"

IF EXIST !cnnOriginalFolder! (
	rmdir /s/q "%gsgdFolder%"
	rename "%cnnOriginalFolder%" "cnn"
)

echo "exiting"
endlocal
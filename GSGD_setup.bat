@echo off
if not "%1"=="am_admin" (powershell start -verb runas '%0' am_admin & exit /b)

setlocal enabledelayedexpansion
echo "Select Source Folder"
set "psCommand="(new-object -COM 'Shell.Application')^
.BrowseForFolder(0,'Please select the Source Folder.',0,0).self.path""
for /f "usebackq delims=" %%I in (`powershell %psCommand%`) do set "srcfolder=%%I"
set "srcCode=\code"
set "srcCodeFolder=%srcfolder%%srcCode%"

echo "Select Destination Folder"
set "psCommand="(new-object -COM 'Shell.Application')^
.BrowseForFolder(0,'Please Select the Destination MATLAB folder.',0,0).self.path""
for /f "usebackq delims=" %%I in (`powershell %psCommand%`) do set "folder=%%I"
set "gsgdcnn=\cnn"
set "cnn_orignal=\cnn_orignal"
set "folderDuplication=%folder%%gsgdcnn%"
set "avoidfolderDuplication=%folder%%cnn_orignal%"

IF EXIST !folderDuplication! (
	rename "%folderDuplication%" "cnn_original"
)
xcopy "%srcCodeFolder%" "%folder%" /s /i

endlocal
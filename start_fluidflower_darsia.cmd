@echo off
setlocal
set "LAUNCHER=%~dp0launch_fluidflower.ps1"
if not exist "%LAUNCHER%" (
  echo DarSIA launcher not found: "%LAUNCHER%"
  pause
  exit /b 1
)

rem The bypass applies only to this PowerShell process, not to Windows settings.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%LAUNCHER%" -Application native
if errorlevel 1 (
  echo DarSIA did not start. See the error above.
  pause
  exit /b 1
)
endlocal

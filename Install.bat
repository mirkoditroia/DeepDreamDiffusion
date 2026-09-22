@echo off
cd /d "%~dp0"
echo Installing the TouchDeepDream Python environment in this folder.
echo.
echo TouchDesigner must use the same Python version. Current builds use 3.11.
echo In the Textport, run:  sys.version
echo If it is not 3.11, edit the -PythonVersion value in this file.
echo.
powershell -ExecutionPolicy Bypass -File "%~dp0install_dependencies.ps1" -Device cuda -PythonVersion 3.11
echo.
echo When this window finishes without errors, cook the component again.
pause

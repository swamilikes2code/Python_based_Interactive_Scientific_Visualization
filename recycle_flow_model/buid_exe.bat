@echo off
echo ========================================================
echo Rebuilding CHE 031 Evaporative Crystallization EXE...
echo ========================================================

:: Clean up previous build artifacts
if exist "dist" rmdir /s /q "dist"
if exist "build" rmdir /s /q "build"

:: Run PyInstaller using the spec file
pyinstaller --clean combined_app.spec

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================================
    echo Build Successful! Executable is located in dist\
    echo ========================================================
) else (
    echo.
    echo ========================================================
    echo Build Failed! Check the error output above.
    echo ========================================================
)

pause
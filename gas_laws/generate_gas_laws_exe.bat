@echo off
echo ========================================================
echo Rebuilding Gas Laws Single-File Executable via run_app.py...
echo ========================================================

:: Clean up previous build artifacts
if exist "dist" rmdir /s /q "dist"
if exist "build" rmdir /s /q "build"

:: Run PyInstaller using the spec file
pyinstaller --clean Gas_Laws.spec

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================================
    echo Build Successful! Executable is located in the dist\ folder.
    echo ========================================================
) else (
    echo.
    echo ========================================================
    echo Build Failed! Check the error output above.
    echo ========================================================
)

pause
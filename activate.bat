@echo off
REM Emotion Recognition - Environment Activation Script

if exist venv\Scripts\activate.bat (
    echo Activating Emotion Recognition environment...
    call venv\Scripts\activate.bat
    echo Environment activated. To deactivate, type: deactivate
) else (
    echo Error: Virtual environment not found. Please run setup.bat first.
    pause
)

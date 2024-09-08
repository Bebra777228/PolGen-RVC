@echo off
setlocal
title PolGen
set "principal=%cd%"
set "CONDA_ROOT_PREFIX=%UserProfile%\Miniconda3"
set "INSTALL_ENV_DIR=%principal%\env"
set "MINICONDA_DOWNLOAD_URL=https://repo.anaconda.com/miniconda/Miniconda3-py39_23.9.0-0-Windows-x86_64.exe"
set "CONDA_EXECUTABLE=%CONDA_ROOT_PREFIX%\Scripts\conda.exe"
if not exist env (
    if not exist "%CONDA_EXECUTABLE%" (
        echo Miniconda not found. Starting download and installation...
        echo Downloading Miniconda...
        powershell -Command "& {Invoke-WebRequest -Uri '%MINICONDA_DOWNLOAD_URL%' -OutFile 'miniconda.exe'}"
        if not exist "miniconda.exe" (
            echo Download failed. Please check your internet connection and try again.
            goto :error
        )

        echo Installing Miniconda...
        start /wait "" miniconda.exe /InstallationType=JustMe /RegisterPython=0 /S /D=%CONDA_ROOT_PREFIX%
        if errorlevel 1 (
            echo Miniconda installation failed.
            goto :error
        )
        del miniconda.exe
        echo Miniconda installation complete.
    ) else (
        echo Miniconda already installed. Skipping installation.
    )
    echo.

    echo Creating Conda environment...
    call "%CONDA_ROOT_PREFIX%\_conda.exe" create --no-shortcuts -y -k --prefix "%INSTALL_ENV_DIR%" python=3.9
    if errorlevel 1 goto :error
    echo Conda environment created successfully.
    echo.

    if exist "%INSTALL_ENV_DIR%\python.exe" (
        echo Installing specific pip version...
        "%INSTALL_ENV_DIR%\python.exe" -m pip install --no-warn-script-location "pip<24.1"
        if errorlevel 1 goto :error
        echo Pip installation complete.
        echo.
    )

    echo Installing dependencies...
    "%INSTALL_ENV_DIR%\python.exe" -m pip install --no-warn-script-location -r requirements.txt
    if errorlevel 1 goto :error
    cls
    echo Dependencies installed successfully.
    echo.
)

env\python download_models.py

env\python app.py --open
if errorlevel 1 goto :error

goto :eof

:error
echo An error occurred. Exiting...
pause
exit /b 1

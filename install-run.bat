@echo off
setlocal
title PolGen
set "principal=%cd%"
set "CONDA_ROOT_PREFIX=%UserProfile%\Miniconda3"
set "INSTALL_ENV_DIR=%principal%\env"
set "MINICONDA_DOWNLOAD_URL=https://repo.anaconda.com/miniconda/Miniconda3-py39_23.9.0-0-Windows-x86_64.exe"
set "PYTHON_DOWNLOAD_URL=https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe"
set "CONDA_EXECUTABLE=%CONDA_ROOT_PREFIX%\Scripts\conda.exe"
set "PYTHON_VERSION_REQUIRED=3.10.11"

:check_internet_connection
echo Checking internet connection...
ping -n 1 google.com >nul 2>&1
if errorlevel 1 (
    echo No internet connection. Please check your connection and try again.
    goto :error
)

:check_python_version
echo Checking Python version...
for /f "tokens=2 delims==." %%a in ('python -V 2^>^&1 ^| findstr /i "Python"') do set "PYTHON_VERSION=%%a"
if "%PYTHON_VERSION%" geq "3" if "%PYTHON_VERSION%" leq "10" (
    if "%PYTHON_VERSION%" neq "8" if "%PYTHON_VERSION%" neq "9" if "%PYTHON_VERSION%" neq "10" (
        echo Unsupported Python version: %PYTHON_VERSION%
        set /p "REINSTALL=Do you want to reinstall Python 3.10? (y/n): "
        if /i "%REINSTALL%"=="y" (
            echo Uninstalling current Python version...
            rmdir /s /q "%CONDA_ROOT_PREFIX%"
            echo Downloading and installing Python %PYTHON_VERSION_REQUIRED%...
            powershell -Command "& {Invoke-WebRequest -Uri '%PYTHON_DOWNLOAD_URL%' -OutFile 'python_installer.exe'}"
            start /wait "" python_installer.exe /quiet InstallAllUsers=1 PrependPath=1 Include_test=0
            del python_installer.exe
            echo Python %PYTHON_VERSION_REQUIRED% installed successfully.
            goto :check_python_version
        ) else (
            echo Exiting...
            goto :error
        )
    )
) else (
    echo Python not found or version is not supported. Please install Python 3.8, 3.9, or 3.10.
    goto :error
)

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
    "%INSTALL_ENV_DIR%\python.exe" -m pip install --no-warn-script-location --no-deps -r requirements-test.txt
    "%INSTALL_ENV_DIR%\python.exe" -m pip uninstall torch torchvision torchaudio -y
    "%INSTALL_ENV_DIR%\python.exe" -m pip install --no-warn-script-location torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
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

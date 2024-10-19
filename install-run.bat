@echo off
setlocal enabledelayedexpansion
title PolGen
set "principal=%cd%"
set "CONDA_ROOT_PREFIX=%UserProfile%\Miniconda3"
set "INSTALL_ENV_DIR=%principal%\env"
set "MINICONDA_DOWNLOAD_URL=https://repo.anaconda.com/miniconda/Miniconda3-py310_24.7.1-0-Windows-x86_64.exe"
set "CONDA_EXECUTABLE=%CONDA_ROOT_PREFIX%\Scripts\conda.exe"
set "step=1"

:check_internet_connection
echo [~!step!~] - Checking internet connection...
ping -n 1 google.com >nul 2>&1
if errorlevel 1 (
    echo No internet connection detected.
    set "INTERNET_AVAILABLE=0"
) else (
    echo Internet connection is available.
    set "INTERNET_AVAILABLE=1"
)
echo.
set /a step+=1

if not exist env (
    echo [~!step!~] - Checking for Miniconda installation...
    if not exist "%CONDA_EXECUTABLE%" (
        echo Miniconda is not installed. Starting download and installation...
        echo Downloading Miniconda...
        powershell -Command "& {Invoke-WebRequest -Uri '%MINICONDA_DOWNLOAD_URL%' -OutFile 'miniconda.exe'}"
        if not exist "miniconda.exe" (
            echo Miniconda download failed. Please check your internet connection and try again.
            goto :error
        )

        echo Installing Miniconda...
        start /wait "" miniconda.exe /InstallationType=JustMe /RegisterPython=0 /S /D=%CONDA_ROOT_PREFIX%
        if errorlevel 1 (
            echo Miniconda installation failed.
            goto :error
        )
        del miniconda.exe
        echo Miniconda installation completed successfully.
    ) else (
        echo Miniconda is already installed. Skipping installation.
    )
    echo.
    set /a step+=1

    echo [~!step!~] - Creating Conda environment...
    call "%CONDA_ROOT_PREFIX%\_conda.exe" create --no-shortcuts -y -k --prefix "%INSTALL_ENV_DIR%" python=3.10
    if errorlevel 1 goto :error
    echo Conda environment created successfully.
    echo.
    set /a step+=1

    echo [~!step!~] - Installing specific pip version...
    if exist "%INSTALL_ENV_DIR%\python.exe" (
        "%INSTALL_ENV_DIR%\python.exe" -m pip install --no-warn-script-location "pip<24.1"
        if errorlevel 1 goto :error
        echo Pip installed successfully.
        echo.
    )
    set /a step+=1

    echo [!step!] - Installing dependencies...
    "%INSTALL_ENV_DIR%\python.exe" -m pip install --no-warn-script-location -r requirements.txt
    "%INSTALL_ENV_DIR%\python.exe" -m pip uninstall torch torchvision torchaudio -y
    "%INSTALL_ENV_DIR%\python.exe" -m pip install --no-warn-script-location torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
    if errorlevel 1 goto :error
    cls
    echo Dependencies installed successfully.
    echo.
    set /a step+=1
)

echo [~!step!~] - Checking for required models...
set "hubert_base=%principal%\rvc\models\embedders\hubert_base.pt"
set "fcpe=%principal%\rvc\models\predictors\fcpe.pt"
set "rmvpe=%principal%\rvc\models\predictors\rmvpe.pt"

if exist "%hubert_base%" (
    if exist "%fcpe%" (
        if exist "%rmvpe%" (
            echo All required models are installed.
        )
    )
) else (
    echo Required models were not found. Installing models...
    echo.
    env\python download_models.py
    if errorlevel 1 goto :error
)
echo.
set /a step+=1

echo [~!step!~] - Running Interface...
if "%INTERNET_AVAILABLE%"=="1" (
    echo Running app.py...
    env\python app.py --open
) else (
    echo Running app_offline.py...
    env\python app_offline.py --open
)
if errorlevel 1 goto :error
set /a step+=1

goto :eof

:error
echo.
echo An error occurred during the process. Exiting the script...
pause
exit /b 1

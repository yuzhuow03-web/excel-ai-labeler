@echo off
setlocal
cd /d "%~dp0"

if exist "ExcelAI-Desktop\ExcelAI-Desktop.exe" (
  start "" "ExcelAI-Desktop\ExcelAI-Desktop.exe"
  exit /b 0
)

set "PYTHON_CMD=%LocalAppData%\Programs\Python\Python312\python.exe"
set "PYTHON_ARGS="
if not exist "%PYTHON_CMD%" (
  where py >nul 2>&1
  if not errorlevel 1 (
    set "PYTHON_CMD=py"
    set "PYTHON_ARGS=-3.12"
  )
)

if not exist "%LocalAppData%\Programs\Python\Python312\python.exe" if "%PYTHON_CMD%"=="%LocalAppData%\Programs\Python\Python312\python.exe" (
  echo Python 3.12 was not found. Install it from https://www.python.org/downloads/
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo Creating local environment...
  "%PYTHON_CMD%" %PYTHON_ARGS% -m venv .venv
)

if not exist ".venv\Scripts\streamlit.exe" (
  echo Installing dependencies...
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt
  if errorlevel 1 (
    echo Dependency installation failed.
    pause
    exit /b 1
  )
)

echo Fast desktop launcher is missing. Starting browser fallback at http://localhost:8501
".venv\Scripts\python.exe" -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501

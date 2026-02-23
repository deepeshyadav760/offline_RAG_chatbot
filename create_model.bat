@echo off
echo ========================================================
echo   OFFLINE RAG CHATBOT - MODEL SETUP
echo ========================================================
echo.
echo This script will import your local GGUF file into Ollama.
echo Model File: Llama-3.2-1B-Instruct-Q4_K_M.gguf
echo Model Name: custom-llama3.2
echo.

if not exist "Llama-3.2-1B-Instruct-Q4_K_M.gguf" (
    echo [ERROR] GGUF file not found!
    echo Please make sure 'Llama-3.2-1B-Instruct-Q4_K_M.gguf' is in this folder.
    pause
    exit /b
)

echo [1/2] Checking if Ollama is running...
tasklist /FI "IMAGENAME eq ollama.exe" 2>NUL | find /I /N "ollama.exe">NUL
if "%ERRORLEVEL%"=="0" (
    echo Ollama is running.
) else (
    echo Ollama is NOT running. Please start it from your Start Menu first.
    pause
    exit /b
)

echo.
echo [2/2] Creating custom model 'custom-llama3.2'...
echo This might take a moment...
echo.

ollama create custom-llama3.2 -f Modelfile

if "%ERRORLEVEL%"=="0" (
    echo.
    echo [SUCCESS] Model created successfully!
    echo You can now run 'python main.py' to start the chatbot.
) else (
    echo.
    echo [ERROR] Failed to create model.
)

pause

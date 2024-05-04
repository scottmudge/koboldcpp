@echo off
cd /d "%~dp0"
echo "Building KoboldCPP Custom You Beautiful Bitch! :)"
echo "First building The binaries..."
set OLDPATH=%PATH%
set PATH=C:\Development\AI\LLM\source\w64devkit\bin;%PATH%
bash -c "cd C:/Development/AI/LLM/source/koboldcpp && ./make.sh"
bash -c "rm -rf C:/Development/AI/LLM/source/koboldcpp/dist/* && rm -rf C:/Development/AI/LLM/source/koboldcpp/build/koboldcpp.exe/*"
copy .\custom.ico .\venv\Lib\site-packages\customtkinter\assets\icons\CustomTkinter_icon_Windows.ico /Y
copy .\custom.ico .\venv\Lib\site-packages\PyInstaller\bootloader\images\icon-console.ico /Y
copy .\custom.ico .\venv\Lib\site-packages\PyInstaller\bootloader\images\icon-windowed.ico /Y
copy .\build\bin\Release\koboldcpp_cublas.dll .\koboldcpp_cublas.dll /Y
set PATH=%OLDPATH%
call .\venv\Scripts\activate.bat && cd "%~dp0" && make_pyinstaller_cuda12.bat
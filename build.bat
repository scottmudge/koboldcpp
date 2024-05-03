@echo off
echo "Building KoboldCPP Custom You Beautiful Bitch! :)"
copy .\custom.ico .\venv\Lib\site-packages\customtkinter\assets\icons\CustomTkinter_icon_Windows.ico /Y
copy .\custom.ico .\venv\Lib\site-packages\PyInstaller\bootloader\images\icon-console.ico /Y
copy .\custom.ico .\venv\Lib\site-packages\PyInstaller\bootloader\images\icon-windowed.ico /Y
call .\venv\Scripts\activate.bat && cd "%~dp0" && make_pyinstaller_cuda12.bat
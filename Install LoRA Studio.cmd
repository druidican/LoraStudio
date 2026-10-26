@echo off
rem Installs LoRA Studio on Windows (inside WSL2). LoRA Studio version 1.261026r
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install-windows.ps1" %*

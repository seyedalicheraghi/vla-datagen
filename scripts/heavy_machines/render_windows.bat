@echo off
rem Render a heavy-machine scenario on Windows (Isaac Sim pip install in <repo>\.venv).
rem   scripts\heavy_machines\render_windows.bat                                 (adt_approach_hex)
rem   scripts\heavy_machines\render_windows.bat --renderer rtx --cameras front  (extra args go to render_scenario.py)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0render_windows.ps1" %*
exit /b %ERRORLEVEL%

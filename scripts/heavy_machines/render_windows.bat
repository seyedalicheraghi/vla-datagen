@echo off
rem Render a heavy-machine scenario on Windows (Isaac Sim pip install in the repo's .venv folder).
rem   scripts\heavy_machines\render_windows.bat                                 (adt_approach_hex)
rem   scripts\heavy_machines\render_windows.bat --renderer rtx --cameras front  (extra args go to render_scenario.py)
rem The exit stays on the same line: the launcher may git-pull a new copy of this file while it runs,
rem and cmd reads a batch file line by line.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0render_windows.ps1" %* & exit /b

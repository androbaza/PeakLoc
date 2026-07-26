@echo off
rem Make the Metavision Studio / SDK bindings visible to the isolated Pixi interpreter.
if not defined PEAKLOC_METAVISION_ROOT set "PEAKLOC_METAVISION_ROOT=%ProgramFiles%\Prophesee"
if not defined PEAKLOC_OPENEB_SITE_PACKAGES set "PEAKLOC_OPENEB_SITE_PACKAGES=%PEAKLOC_METAVISION_ROOT%\lib\python3\site-packages"
rem Pixi does not add its DLL directory to PATH when a custom activation script runs.
rem Keep it ahead of the SDK's older image-library DLLs.
set "PATH=%CONDA_PREFIX%\Library\bin;%CONDA_PREFIX%;%CONDA_PREFIX%\Scripts;%PATH%"
if exist "%PEAKLOC_METAVISION_ROOT%\bin" set "PATH=%PATH%;%PEAKLOC_METAVISION_ROOT%\bin;%PEAKLOC_METAVISION_ROOT%\third_party\bin"

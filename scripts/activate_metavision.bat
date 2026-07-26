@echo off
rem Make the Metavision Studio / SDK bindings visible to the isolated Pixi interpreter.
if not defined PEAKLOC_METAVISION_ROOT set "PEAKLOC_METAVISION_ROOT=%ProgramFiles%\Prophesee"
if not defined PEAKLOC_OPENEB_SITE_PACKAGES set "PEAKLOC_OPENEB_SITE_PACKAGES=%PEAKLOC_METAVISION_ROOT%\lib\python3\site-packages"
if exist "%PEAKLOC_METAVISION_ROOT%\bin" set "PATH=%PEAKLOC_METAVISION_ROOT%\bin;%PEAKLOC_METAVISION_ROOT%\third_party\bin;%PATH%"

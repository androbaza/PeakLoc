@echo off
rem Make the Metavision Studio / SDK bindings visible to the isolated Pixi interpreter.
if not defined PEAKLOC_METAVISION_ROOT set "PEAKLOC_METAVISION_ROOT=%ProgramFiles%\Prophesee"
if not defined PEAKLOC_OPENEB_SITE_PACKAGES set "PEAKLOC_OPENEB_SITE_PACKAGES=%PEAKLOC_METAVISION_ROOT%\lib\python3\site-packages"
rem Pixi does not add its DLL directory to PATH when a custom activation script runs.
rem Do not add the SDK DLLs here: they shadow h5py's compatible HDF5 DLL at startup.
rem PeakLoc adds them temporarily only after h5py has loaded.
set "PATH=%CONDA_PREFIX%\Library\bin;%CONDA_PREFIX%;%CONDA_PREFIX%\Scripts;%PATH%"

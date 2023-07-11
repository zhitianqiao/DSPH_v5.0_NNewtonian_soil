@echo off
setlocal EnableDelayedExpansion
rem Don't remove the two jump line after than the next line [set NL=^]
set NL=^


rem "name" and "dirout" are named according to the testcase

set name=CaseCollapseSand2d
set dirout=%name%_out
set diroutdata=%dirout%/..

rem "executables" are renamed and called from their directory

set dirbin=../../../bin/windows
set gencase="%dirbin%/GenCase_win64.exe"
set dualsphysicscpu="%dirbin%/DualSPHysics5.0CPU_win64.exe"
set dualsphysicsgpu="%dirbin%/DualSPHysics5.0_win64.exe"
set boundaryvtk="%dirbin%/BoundaryVTK_win64.exe"
set partvtk="%dirbin%/PartVTK_win64.exe"
set partvtkout="%dirbin%/PartVTKOut_win64.exe"
set measuretool="%dirbin%/MeasureTool_win64.exe"
set computeforces="%dirbin%/ComputeForces_win64.exe"
set isosurface="%dirbin%/IsoSurface_win64.exe"
set flowtool="%dirbin%/FlowTool_win64.exe"
set floatinginfo="%dirbin%/FloatingInfo_win64.exe"
set bifileinfo="%dirbin%/BIFileInfo_win64.exe"
set tovtk="%dirbin%/ToVtk_win64.exe"

REM :menu
REM if exist %dirout% ( 
	REM set /p option="The folder "%dirout%" already exists. Choose an option.!NL!  [1]- Delete it and continue.!NL!  [2]- Execute post-processing.!NL!  [3]- Abort and exit.!NL!"
	REM if "!option!" == "1" goto run else (
		REM if "!option!" == "2" goto postprocessing else (
			REM if "!option!" == "3" goto fail else ( 
				REM goto menu
			REM )
		REM )
	REM )
REM )

REM :run
rem "dirout" to store results is removed if it already exists
REM if exist %dirout% rd /s /q %dirout%

rem CODES are executed according the selected parameters of execution in this testcase

rem Executes GenCase to create initial files for simulation.
REM %gencase% %name%_Def %dirout%/%name% -save:all
REM if not "%ERRORLEVEL%" == "0" goto fail

rem Executes DualSPHysics to simulate SPH method.
REM %dualsphysicscpu% %dirout%/%name% %dirout% -dirdataout data -svres
REM if not "%ERRORLEVEL%" == "0" goto fail

:postprocessing
rem Executes PartVTK to create VTK files with particles.
set dirout2=%dirout%\particles
%partvtk% -dirin %diroutdata% -savevtk %dirout2%/PartFluid -onlytype:-all,+fluid -vars:+all
if not "%ERRORLEVEL%" == "0" goto fail

rem Executes PartVTKOut to create VTK files with excluded particles.
%partvtkout% -dirin %diroutdata% -savevtk %dirout2%/PartFluidOut -SaveResume %dirout2%/_ResumeFluidOut
if not "%ERRORLEVEL%" == "0" goto fail

rem Executes BIFileInfo to show information of 
%bifileinfo% Part_0046.bi4 -svarrays
if not "%ERRORLEVEL%" == "0" goto fail

:success
echo All done
goto end

:fail
echo Execution aborted.

:end
pause

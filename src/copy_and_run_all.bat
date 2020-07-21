SET MPATH=C:\school\Thesa\27082019 RUNS
SET MTYPE=greedy
SET MREPEAT=40


FOR /L %%A IN (2,1,%MREPEAT%) DO (
  xcopy "%MPATH%\%MTYPE%_1\*.*" "%MPATH%\%MTYPE%_%%A\" /s/h/e/k/f/c
)
FOR /L %%A IN (1,1,%MREPEAT%) DO (
  start "" python "%MPATH%\%MTYPE%_%%A\src\main_runner.py"
)
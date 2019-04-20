SET MPATH=C:\school\Thesa\20042019 RUNS\omission_poly
SET MTYPE=greedy
SET MREPEAT=7

FOR /L %%A IN (2,1,7) DO (
  xcopy "%MPATH%\%MTYPE%_1\*.*" "%MPATH%\%MTYPE%_%%A\" /s/h/e/k/f/c
)
FOR /L %%A IN (1,1,7) DO (
  start "" python "%MPATH%\%MTYPE%_%%A\src\main_runner.py"
)
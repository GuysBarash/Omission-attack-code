SET MPATH=C:\school\Thesa\omission_poly
SET MTYPE=genetic
FOR /L %%A IN (1,1,7) DO (
  start "" python "%MPATH%\%MTYPE%_%%A\src\main_runner.py"
)
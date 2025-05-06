@echo off
setlocal enabledelayedexpansion

:: List of search methods
set METHODS=BFS DFS GBFS AS CUS1 CUS2

:: Loop through files
for %%F in (Test_*.txt) do (
    echo Running tests on %%F
    for %%M in (%METHODS%) do (
        echo Running method: %%M
        python search.py %%F %%M debug
        echo ----------------------------
    )
)

echo All test cases completed.
pause

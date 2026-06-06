@echo off
cd /d C:\Users\black\OneDrive\Desktop\FCF
set PYTHONIOENCODING=utf-8
echo === Combined Learning Test ===
echo Method: HDBSCAN + Louvain + Sequential Connectedness
echo.
python experiments/test_combined.py
echo.
echo === Done ===
pause

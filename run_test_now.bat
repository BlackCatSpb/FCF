@echo off
title EVA Combined Learning Test
cd /d C:\Users\black\OneDrive\Desktop\FCF
set PYTHONIOENCODING=utf-8
echo ========================================
echo  EVA Combined Learning Test
echo ========================================
echo  L1: HDBSCAN (auto concept clusters)
echo  L2: Louvain (community detection)
echo  Training: sequential connectedness
echo    - Every token type learns position
echo    - Non-matches: full lr (focus)
echo    - Matches: 20%% lr (preserve)
echo    - type-3: prefix-level shift
echo ========================================
echo.
python experiments/test_combined.py
echo.
echo === DONE ===
pause

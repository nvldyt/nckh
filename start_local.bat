@echo off
color 0A
echo ===================================================
echo   DANG DONG BO CODE MOI NHAT TU GITHUB...
echo ===================================================

git pull origin main

echo.
echo ===================================================
echo   KHOI DONG HE THONG STREAMLIT LOCAL...
echo ===================================================

:: Thay 'app.py' bang ten file giao dien chinh cua ban
streamlit run app.py

pause

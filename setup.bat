@echo off
chcp 65001 >nul
echo ========================================
echo   PhotoCleaner 一键部署
echo ========================================

REM 1. 创建虚拟环境
echo.
if not exist "venv\" (
    echo [1/3] 创建虚拟环境...
    python -m venv venv
) else (
    echo [1/3] 虚拟环境已存在，跳过
)

REM 激活
call venv\Scripts\activate.bat

REM 2. 安装依赖
echo.
echo [2/3] 安装 Python 依赖...
pip install -r requirements.txt

REM 3. 下载模型
echo.
echo [3/3] 下载模型文件...
python setup_models.py

echo.
echo ========================================
echo   部署完成
echo   启动: venv\Scripts\activate.bat ^&^& streamlit run app.py
echo ========================================
pause

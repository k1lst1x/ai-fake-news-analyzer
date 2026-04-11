@echo off
setlocal enableextensions

set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"
set "PY=%ROOT%\.venv\Scripts\python.exe"

if not exist "%PY%" (
  echo [ERROR] Python venv not found: %PY%
  echo Create it first: python -m venv .venv
  exit /b 1
)

cd /d "%ROOT%" || exit /b 1

if /I "%SKIP_PIP%"=="1" (
  echo [1/6] Install dependencies... skipped ^(SKIP_PIP=1^)
) else (
  echo [1/6] Install dependencies...
  "%PY%" -m pip install -r requirements.txt || goto :err
)

if not exist "data\expanded\expanded_news_dataset.csv" (
  echo [2/6] Build seed dataset...
  "%PY%" scripts\expand_dataset.py --target-gb 1.0 --out-csv data/expanded/expanded_news_dataset.csv --max-cc-rows 300000 --max-ag-rows 250000 --max-extra-fake-rows 300000 --synth-ratio 0.65 || goto :err
) else (
  echo [2/6] Seed dataset exists, skip.
)

if not exist "data\expanded\expanded_news_dataset_15gb.csv" (
  echo [3/6] Scale dataset to 15.2 GB...
  "%PY%" scripts\scale_dataset_to_target.py --source-csv data/expanded/expanded_news_dataset.csv --out-csv data/expanded/expanded_news_dataset_15gb.csv --target-gb 15.2 || goto :err
) else (
  echo [3/6] 15GB dataset exists, skip.
)

if not exist "models\rf_meta_model.joblib" (
  echo [4/6] Train Random Forest...
  "%PY%" scripts\train_random_forest.py --data-csv data/expanded/expanded_news_dataset_15gb.csv --sample-size 500000 --chunksize 150000 || goto :err
) else (
  echo [4/6] RF model exists, skip.
)

if not exist "models\fake_news_model_ft\config.json" (
  echo [5/6] Fine-tune transformer...
  "%PY%" scripts\fine_tune_transformer.py --data-csv data/expanded/expanded_news_dataset_15gb.csv --base-model-dir models/fake_news_model --output-dir models/fake_news_model_ft --max-rows 140000 --chunksize 180000 --epochs 0.35 --batch-size 2 --grad-accum 8 --lr 2e-5 || goto :err
) else (
  echo [5/6] Fine-tuned model exists, skip.
)

echo [6/6] Start API and UI in new PowerShell windows...
start "FakeNews API" powershell -NoExit -Command "cd '%ROOT%'; & '%PY%' -m uvicorn api:app --host 0.0.0.0 --port 8000"
start "FakeNews UI" powershell -NoExit -Command "cd '%ROOT%'; & '%PY%' -m streamlit run app.py"

echo.
echo Done.
echo API: http://127.0.0.1:8000/docs
echo UI : http://localhost:8501
exit /b 0

:err
echo.
echo [ERROR] Command failed with exit code %ERRORLEVEL%.
exit /b %ERRORLEVEL%

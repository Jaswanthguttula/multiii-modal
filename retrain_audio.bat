@echo off
echo ========================================
echo   AUDIO MODEL RETRAINING SCRIPT
echo ========================================
echo.
echo This will retrain the audio model with:
echo - Correct feature extraction (40 MFCC means)
echo - Proper StandardScaler normalization
echo - Saved scaler for inference
echo.
echo Files to be regenerated:
echo   - audio_isolation_forest.pkl
echo   - audio_scaler.pkl
echo.
pause
echo.
echo Starting training...
echo.

python train_audio.py

echo.
echo ========================================
if %ERRORLEVEL% EQU 0 (
    echo Training completed successfully!
    echo.
    echo Next steps:
    echo 1. Restart the dashboard: python web_dashboard.py
    echo 2. Test audio scoring - it should now vary with audio content
    echo 3. Audio scores should no longer be stuck at ~0.55
) else (
    echo Training FAILED! Error code: %ERRORLEVEL%
    echo.
    echo Possible issues:
    echo - No audio files in DATASET/dataset.audio/
    echo - Missing libraries: pip install librosa scikit-learn
    echo - Python not in PATH
)
echo ========================================
echo.
pause

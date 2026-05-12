# =============================================================================
#  setup.ps1  —  Installazione automatica dell'ambiente ASL Real-time
# =============================================================================
#  Esegui una sola volta prima di usare avvia.ps1
#  Requisiti: Windows 10/11 a 64 bit, connessione Internet
#
#  Versioni installate (testate e compatibili):
#    Python      3.11.9
#    TensorFlow  2.17.0
#    MediaPipe   0.10.14
#    OpenCV      ultima stabile
#    NumPy       compatibile con TF 2.17
# =============================================================================

$ErrorActionPreference = "Stop"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$PYTHON_VERSION       = "3.11.9"
$PYTHON_INSTALLER_URL = "https://www.python.org/ftp/python/$PYTHON_VERSION/python-$PYTHON_VERSION-amd64.exe"
$PYTHON_INSTALLER     = "$env:TEMP\python-$PYTHON_VERSION-amd64.exe"
$PYTHON_EXE           = "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe"
$VENV_DIR             = "C:\asl_env"

Write-Host ""
Write-Host "======================================" -ForegroundColor Cyan
Write-Host "  Setup ASL Real-time Recognition" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""

# -----------------------------------------------------------------------------
#  1. Installa Python 3.11 se non presente
# -----------------------------------------------------------------------------
if (-not (Test-Path $PYTHON_EXE)) {
    Write-Host "[1/4] Python 3.11 non trovato. Download in corso..." -ForegroundColor Yellow
    Invoke-WebRequest -Uri $PYTHON_INSTALLER_URL -OutFile $PYTHON_INSTALLER -UseBasicParsing
    Write-Host "[1/4] Installazione Python 3.11..." -ForegroundColor Yellow
    & $PYTHON_INSTALLER /quiet InstallAllUsers=0 PrependPath=1 Include_test=0
    Start-Sleep -Seconds 10
    if (-not (Test-Path $PYTHON_EXE)) {
        Write-Host "ERRORE: Installazione Python fallita. Installa manualmente da https://www.python.org" -ForegroundColor Red
        exit 1
    }
    Write-Host "[1/4] Python 3.11 installato." -ForegroundColor Green
} else {
    Write-Host "[1/4] Python 3.11 gia' presente." -ForegroundColor Green
}

# -----------------------------------------------------------------------------
#  2. Crea il virtual environment in C:\asl_env
#     (percorso senza caratteri speciali per evitare problemi di encoding)
# -----------------------------------------------------------------------------
if (-not (Test-Path "$VENV_DIR\Scripts\python.exe")) {
    Write-Host "[2/4] Creazione virtual environment in $VENV_DIR ..." -ForegroundColor Yellow
    & $PYTHON_EXE -m venv $VENV_DIR
    Write-Host "[2/4] Virtual environment creato." -ForegroundColor Green
} else {
    Write-Host "[2/4] Virtual environment gia' presente." -ForegroundColor Green
}

$VENV_PYTHON = "$VENV_DIR\Scripts\python.exe"

# -----------------------------------------------------------------------------
#  3. Aggiorna pip
# -----------------------------------------------------------------------------
Write-Host "[3/4] Aggiornamento pip..." -ForegroundColor Yellow
& $VENV_PYTHON -m pip install --upgrade pip --quiet

# -----------------------------------------------------------------------------
#  4. Installa le dipendenze con versioni specifiche testate
#
#  ATTENZIONE: non cambiare le versioni di tensorflow e mediapipe.
#  tensorflow==2.17.0 e mediapipe==0.10.14 sono la combinazione minima
#  che risolve i conflitti su protobuf, jax e ml_dtypes.
#  Versioni piu' recenti di mediapipe (>= 0.10.20 circa) hanno rimosso
#  mp.solutions.holistic, richiesto da questo script.
# -----------------------------------------------------------------------------
Write-Host "[4/4] Installazione librerie..." -ForegroundColor Yellow
Write-Host "      Questa operazione puo' richiedere 5-10 minuti (~500 MB)." -ForegroundColor Gray
& $VENV_PYTHON -m pip install `
    "tensorflow==2.17.0" `
    "mediapipe==0.10.14" `
    "opencv-python" `
    "numpy"

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERRORE: Installazione dipendenze fallita." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "======================================" -ForegroundColor Green
Write-Host "  Setup completato con successo!" -ForegroundColor Green
Write-Host "======================================" -ForegroundColor Green
Write-Host ""
Write-Host "Per avviare il progetto esegui:" -ForegroundColor Cyan
Write-Host "  .\avvia.ps1" -ForegroundColor White
Write-Host ""

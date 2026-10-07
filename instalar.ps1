# Crea los cuatro entornos del #19 (Windows, PowerShell, Python 3.11).
# Cada modelo va en su propio entorno porque sus dependencias chocan.
# Uso, desde la carpeta 19:  powershell -ExecutionPolicy Bypass -File instalar.ps1

$ErrorActionPreference = "Stop"

function Ejecutar {
    # Los ejecutables externos no paran el script al fallar: hay que mirar
    # el código de salida a mano.
    & $args[0] $args[1..($args.Count - 1)]
    if ($LASTEXITCODE -ne 0) { throw "Falló: $($args -join ' ')" }
}

# Varias dependencias (Chatterbox, spacy-pkuseg) no tienen ruedas para
# Python 3.13 en Windows, así que usamos 3.11 en paralelo al 3.13 que ya hay.
$ErrorActionPreference = "Continue"
& py -3.11 --version *> $null
$hay311 = ($LASTEXITCODE -eq 0)
$ErrorActionPreference = "Stop"
if (-not $hay311) {
    Write-Host "== Instalando Python 3.11 con winget"
    Ejecutar winget install -e --id Python.Python.3.11 --scope user --accept-package-agreements --accept-source-agreements
}

function Nuevo-Entorno($nombre, $torch, $indice, $reqs) {
    Write-Host "== $nombre"
    if (-not (Test-Path "$nombre\Scripts\python.exe")) { Ejecutar py -3.11 -m venv $nombre }
    $py = "$nombre\Scripts\python.exe"
    Ejecutar $py -m pip install --upgrade pip
    Ejecutar $py -m pip install @torch --index-url $indice
    Ejecutar $py -m pip install -r $reqs
}

# torch 2.8 mantiene torchaudio.load sin torchcodec, que en Windows da guerra.
$t28 = @("torch==2.8.0", "torchaudio==2.8.0")
$cu128 = "https://download.pytorch.org/whl/cu128"

Nuevo-Entorno ".venv-xtts"       $t28 $cu128 "requirements\xtts.txt"
Nuevo-Entorno ".venv-f5"         $t28 $cu128 "requirements\f5.txt"
Nuevo-Entorno ".venv-metricas"   $t28 $cu128 "requirements\metricas.txt"
# torchcodec llega como dependencia, pero en Windows necesita las DLL de FFmpeg
# y rompe la transcripción de transformers. Nadie lo usa: fuera.
foreach ($e in ".venv-f5", ".venv-metricas") { & "$e\Scripts\python.exe" -m pip uninstall -y torchcodec }
# Chatterbox fija torch 2.6.0.
Nuevo-Entorno ".venv-chatterbox" @("torch==2.6.0", "torchaudio==2.6.0") "https://download.pytorch.org/whl/cu124" "requirements\chatterbox.txt"

Write-Host "Listo."

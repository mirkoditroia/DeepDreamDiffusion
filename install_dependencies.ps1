param(
    [ValidateSet("cuda", "cpu")]
    [string]$Device = "cuda",

    [string]$PythonVersion = "3.11"
)

$ErrorActionPreference = "Stop"

$PluginDir = $PSScriptRoot
$VenvDir = Join-Path $PSScriptRoot "venv_td"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$Marker = Join-Path $VenvDir ".deepdream_td_version"
$TorchIndex = "https://download.pytorch.org/whl/cu128"

Write-Host "DeepDream TouchDesigner - dependency installer" -ForegroundColor Cyan
Write-Host "Plugin directory: $PluginDir"
Write-Host "Required TouchDesigner Python version: $PythonVersion"
Write-Host "Device: $Device"
Write-Host ""

& py "-$PythonVersion" -c "import sys; print(sys.executable)"
if ($LASTEXITCODE -ne 0) {
    throw "Python $PythonVersion was not found. Install it manually and try again."
}

if (-not (Test-Path $VenvPython)) {
    Write-Host "Creating the dedicated environment: $VenvDir" -ForegroundColor Yellow
    & py "-$PythonVersion" -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) {
        throw "Could not create the virtual environment."
    }
}

& $VenvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "Could not update pip." }

if ($Device -eq "cuda") {
    Write-Host "Installing CUDA-enabled PyTorch in the environment..." -ForegroundColor Yellow
    & $VenvPython -m pip install `
        "torch==2.11.0+cu128" `
        "torchvision==0.26.0+cu128" `
        --index-url $TorchIndex
} else {
    Write-Host "Installing CPU-only PyTorch in the environment..." -ForegroundColor Yellow
    & $VenvPython -m pip install torch torchvision
}
if ($LASTEXITCODE -ne 0) { throw "Could not install PyTorch." }

Write-Host "Installing the remaining DeepDream dependencies..." -ForegroundColor Yellow
& $VenvPython -m pip install `
    opencv-contrib-python `
    "numpy<2" `
    "pillow>=10" `
    "open-clip-torch>=2.24"
if ($LASTEXITCODE -ne 0) { throw "Could not install the DeepDream dependencies." }

& $VenvPython -c "import torch, cv2, open_clip; print('torch', torch.__version__, 'CUDA', torch.cuda.is_available()); print('opencv', cv2.__version__)"
if ($LASTEXITCODE -ne 0) { throw "Dependency import check failed." }

Set-Content -Path $Marker -Value $PythonVersion -NoNewline

Write-Host ""
Write-Host "Installation complete." -ForegroundColor Green
Write-Host "Return to TouchDesigner and press Check Dependencies."

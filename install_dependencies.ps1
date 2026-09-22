param(
    [ValidateSet("cuda", "cpu")]
    [string]$Device = "cuda",

    [string]$PythonVersion = ""
)

$ErrorActionPreference = "Stop"

$PluginDir = $PSScriptRoot
$VenvDir = Join-Path $PluginDir "venv_td"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$Marker = Join-Path $VenvDir ".deepdream_td_version"
$Lock = Join-Path $PluginDir "install.lock"
$Log = Join-Path $PluginDir "install_log.txt"
$TorchIndex = "https://download.pytorch.org/whl/cu128"

function Write-Log([string]$Message) {
    $line = "{0} {1}" -f (Get-Date -Format "HH:mm:ss"), $Message
    Add-Content -Path $Log -Value $line
    Write-Host $Message
}

function Get-PyVersion([string]$PythonExe) {
    $out = & $PythonExe -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
    if ($LASTEXITCODE -ne 0 -or -not $out) {
        throw "Could not read the Python version from $PythonExe"
    }
    return ($out | Select-Object -Last 1).Trim()
}

function Find-TouchDesignerPythons {
    $roots = New-Object System.Collections.Generic.List[string]
    foreach ($base in @($env:ProgramFiles, ${env:ProgramFiles(x86)}, $env:LOCALAPPDATA)) {
        if (-not $base) { continue }
        $deriv = Join-Path $base "Derivative"
        if (Test-Path $deriv) { $roots.Add($deriv) }
    }
    foreach ($key in @(
        "HKLM:\SOFTWARE\Derivative",
        "HKLM:\SOFTWARE\WOW6432Node\Derivative",
        "HKCU:\SOFTWARE\Derivative"
    )) {
        if (-not (Test-Path $key)) { continue }
        Get-ChildItem $key -ErrorAction SilentlyContinue | ForEach-Object {
            $props = Get-ItemProperty $_.PSPath -ErrorAction SilentlyContinue
            foreach ($name in @("InstallDir", "InstallLocation", "Path")) {
                $value = $props.$name
                if ($value -and (Test-Path $value)) { $roots.Add([string]$value) }
            }
        }
    }
    $found = @()
    foreach ($root in ($roots | Select-Object -Unique)) {
        $direct = Join-Path $root "bin\python.exe"
        if (Test-Path $direct) { $found += Get-Item $direct }
        Get-ChildItem $root -Directory -ErrorAction SilentlyContinue | ForEach-Object {
            $nested = Join-Path $_.FullName "bin\python.exe"
            if (Test-Path $nested) { $found += Get-Item $nested }
        }
    }
    $found | Sort-Object FullName -Unique | Sort-Object LastWriteTime -Descending
}

function Install-OfficialPython([string]$Version) {
    $versions = @{
        "3.11" = "3.11.9"
        "3.12" = "3.12.10"
    }
    $full = $versions[$Version]
    if (-not $full) { $full = "$Version.0" }
    $url = "https://www.python.org/ftp/python/$full/python-$full-amd64.exe"
    $dest = Join-Path $env:TEMP "TouchDeepDream-Python-$full.exe"
    Write-Log "Downloading Python $full"
    Invoke-WebRequest -Uri $url -OutFile $dest
    Write-Log "Installing Python $full for the current user"
    $installer = Start-Process -FilePath $dest -ArgumentList @(
        "/quiet", "InstallAllUsers=0", "PrependPath=1",
        "Include_launcher=1", "Include_pip=1", "Include_test=0"
    ) -Wait -PassThru
    if ($installer.ExitCode -ne 0) {
        throw "The Python installer exited with code $($installer.ExitCode)."
    }
    $tag = $Version.Replace(".", "")
    $local = Join-Path $env:LOCALAPPDATA "Programs\Python\Python$tag\python.exe"
    if (Test-Path $local) { return $local }
    $fromPy = & py "-$Version" -c "import sys; print(sys.executable)"
    if ($LASTEXITCODE -eq 0 -and $fromPy) { return ($fromPy | Select-Object -Last 1).Trim() }
    throw "Python $Version was installed, but python.exe was not found."
}

if (Test-Path $Lock) {
    $age = (Get-Date) - (Get-Item $Lock).LastWriteTime
    $owner = (Get-Content $Lock -Raw -ErrorAction SilentlyContinue)
    if ($owner) { $owner = $owner.Trim() }
    if ($age.TotalHours -lt 3 -and $owner -match '^\d+$') {
        Write-Log "An installation is already running."
        exit 0
    }
}
Set-Content -Path $Lock -Value $PID -Encoding utf8
Set-Content -Path $Log -Value "TouchDeepDream install started" -Encoding utf8

try {
    $candidates = @(Find-TouchDesignerPythons)
    $basePython = $null
    foreach ($item in $candidates) {
        $ver = Get-PyVersion $item.FullName
        if (-not $PythonVersion -or $ver -eq $PythonVersion) {
            $basePython = $item.FullName
            $PythonVersion = $ver
            break
        }
    }
    if (-not $basePython) {
        if (-not $PythonVersion) { $PythonVersion = "3.11" }
        Write-Log "TouchDesigner Python was not found. Installing Python $PythonVersion."
        $basePython = Install-OfficialPython $PythonVersion
        $PythonVersion = Get-PyVersion $basePython
    }
    Write-Log "Using $basePython ($PythonVersion)"

    if (-not (Test-Path $VenvPython)) {
        Write-Log "Creating the environment in $VenvDir"
        & $basePython -m venv $VenvDir
        if ($LASTEXITCODE -ne 0) { throw "Could not create the virtual environment." }
    }

    foreach ($name in @(
        "PIP_INDEX_URL", "PIP_EXTRA_INDEX_URL", "PIP_FIND_LINKS",
        "PIP_TRUSTED_HOST", "PIP_CONFIG_FILE"
    )) {
        Remove-Item "Env:$name" -ErrorAction SilentlyContinue
    }

    Write-Log "Updating pip"
    & $VenvPython -m pip install --isolated --disable-pip-version-check --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw "Could not update pip." }

    if ($Device -eq "cuda") {
        Write-Log "Installing PyTorch with CUDA. This is the long step."
        & $VenvPython -m pip install --isolated --disable-pip-version-check `
            "torch==2.11.0+cu128" `
            "torchvision==0.26.0+cu128" `
            --index-url $TorchIndex
    } else {
        Write-Log "Installing the CPU build of PyTorch."
        & $VenvPython -m pip install --isolated --disable-pip-version-check torch torchvision
    }
    if ($LASTEXITCODE -ne 0) { throw "Could not install PyTorch." }

    Write-Log "Installing OpenCV, NumPy, and Pillow"
    & $VenvPython -m pip install --isolated --disable-pip-version-check `
        opencv-contrib-python "numpy<2" "pillow>=10"
    if ($LASTEXITCODE -ne 0) { throw "Could not install the remaining libraries." }

    & $VenvPython -c "import torch, cv2; print('torch', torch.__version__, 'CUDA', torch.cuda.is_available()); print('opencv', cv2.__version__)"
    if ($LASTEXITCODE -ne 0) { throw "The installed libraries could not be imported." }

    Set-Content -Path $Marker -Value $PythonVersion -NoNewline -Encoding ascii
    Write-Log "Installation complete."
    Write-Host ""
    Write-Host "Installation complete. In TouchDesigner, pulse Check Dependencies and turn Active on." -ForegroundColor Green
}
catch {
    Write-Log "Installation failed: $($_.Exception.Message)"
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}
finally {
    Remove-Item $Lock -ErrorAction SilentlyContinue
    Write-Host ""
    Read-Host "Press Enter to close"
}

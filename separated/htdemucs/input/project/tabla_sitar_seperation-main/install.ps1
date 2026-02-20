Write-Host "🎵 Tabla Sitar Separation Installer" -ForegroundColor Cyan

$projectName = "tabla_sitar_seperation"
$zipUrl = "https://github.com/charudatta10/tabla_sitar_seperation/archive/refs/heads/main.zip"

# -------------------------------
# Install Python if missing
# -------------------------------
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {

    Write-Host "🐍 Python not found. Installing Python..." -ForegroundColor Yellow

    $pythonUrl = "https://www.python.org/ftp/python/3.12.2/python-3.12.2-amd64.exe"
    $installer = "$env:TEMP\python-installer.exe"

    Invoke-WebRequest $pythonUrl -OutFile $installer

    Start-Process -Wait -FilePath $installer -ArgumentList `
        "/quiet InstallAllUsers=1 PrependPath=1 Include_test=0"

    Remove-Item $installer

    $env:Path += ";C:\Program Files\Python312\;C:\Program Files\Python312\Scripts\"

    Write-Host "✅ Python installed."
}

# -------------------------------
# Install uv if missing
# -------------------------------
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {

    Write-Host "📦 Installing uv..." -ForegroundColor Yellow
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
}

# -------------------------------
# Download project
# -------------------------------
Write-Host "📥 Downloading project..."
Invoke-WebRequest $zipUrl -OutFile "project.zip"
Expand-Archive project.zip -Force
Remove-Item project.zip

Rename-Item "tabla_sitar_seperation-main" $projectName -ErrorAction SilentlyContinue
cd $projectName

# -------------------------------
# Install dependencies
# -------------------------------
Write-Host "🔧 Installing dependencies via uv..."
uv sync

Write-Host ""
Write-Host "✅ Installation complete!"
Write-Host ""
Write-Host "🚀 Launch with:"
Write-Host "uv run streamlit run sitar_tabla_separator.py"

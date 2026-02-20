Write-Host "📦 GitHub Project Installer" -ForegroundColor Cyan

$repoZipUrl = "https://github.com/charudatta10/tabla_sitar_seperation/archive/refs/heads/main.zip"
$projectName = "tabla_sitar_seperation"

# ---------------------------------------
# 1️⃣ Check & Install Python (if missing)
# ---------------------------------------
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {

    Write-Host "🐍 Python not found. Installing..." -ForegroundColor Yellow

    $pythonUrl = "https://www.python.org/ftp/python/3.12.2/python-3.12.2-amd64.exe"
    $installer = "$env:TEMP\python-installer.exe"

    Invoke-WebRequest $pythonUrl -OutFile $installer

    Start-Process -Wait -FilePath $installer -ArgumentList `
        "/quiet InstallAllUsers=1 PrependPath=1 Include_test=0"

    Remove-Item $installer

    $env:Path += ";C:\Program Files\Python312\;C:\Program Files\Python312\Scripts\"

    Write-Host "✅ Python installed."
}
else {
    Write-Host "✅ Python already installed."
}

# ---------------------------------------
# 2️⃣ Check & Install uv (if missing)
# ---------------------------------------
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {

    Write-Host "⚡ uv not found. Installing..." -ForegroundColor Yellow

    powershell -ExecutionPolicy Bypass -c "irm https://astral.sh/uv/install.ps1 | iex"

    Write-Host "✅ uv installed."
}
else {
    Write-Host "✅ uv already installed."
}

# ---------------------------------------
# 3️⃣ Download GitHub Repository
# ---------------------------------------
Write-Host "📥 Downloading repository..."

Invoke-WebRequest $repoZipUrl -OutFile "project.zip"
Expand-Archive project.zip -Force
Remove-Item project.zip

# Rename folder if needed
if (Test-Path "$projectName-main") {
    Rename-Item "$projectName-main" $projectName -Force
}

Set-Location $projectName

# ---------------------------------------
# 4️⃣ Install Dependencies
# ---------------------------------------
Write-Host "🔧 Installing project dependencies..."
uv sync

Write-Host ""
Write-Host "🎉 Setup complete!"
Write-Host "Run with:"
Write-Host "uv run streamlit run sitar_tabla_separator.py"

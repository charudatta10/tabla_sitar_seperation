# use PowerShell instead of sh:
set shell := ["powershell.exe", "-c"]

hello:
  Write-Host "Hello, world!"

install:
    uv sync

run:
    uv run streamlit run sitar_tabla_separtor.py

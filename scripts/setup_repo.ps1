# scripts/setup_repo.ps1
# 一键完成今天的收尾：跑测试 -> 生成 requirements.txt -> 初始化 Git 并提交。
#
# 用法（在项目根目录）：
#   .\scripts\setup_repo.ps1
#   .\scripts\setup_repo.ps1 -WithLLM      # 额外跑真实大模型回归（会产生费用）
#
# 如果被 ExecutionPolicy 拦住：
#   powershell -ExecutionPolicy Bypass -File .\scripts\setup_repo.ps1
param(
    [switch]$WithLLM
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
Write-Host "Project: $Root" -ForegroundColor Cyan

# ---------- 0. 环境检查 ----------
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "python not found. Activate the conda env first (conda activate agent)."
}
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "git not found. Install Git for Windows first."
}

# ---------- 1. 不花钱的测试 ----------
Write-Host "`n[1/4] Running tool / loop / FastAPI tests ..." -ForegroundColor Yellow
python -m pytest tests/test_tools.py tests/test_tool_agent.py tests/test_api.py -v
if ($LASTEXITCODE -ne 0) {
    throw "Tests failed. Fix them before committing."
}

# ---------- 2. 真实大模型回归（可选） ----------
if ($WithLLM) {
    Write-Host "`n[2/4] Running real-LLM regression tests (costs tokens) ..." -ForegroundColor Yellow
    python -m pytest tests/test_regression.py -v -s
    if ($LASTEXITCODE -ne 0) {
        throw "Regression tests failed. Fix them before committing."
    }
} else {
    Write-Host "`n[2/4] Skipped real-LLM regression. Re-run with -WithLLM to include it." -ForegroundColor DarkGray
}

# ---------- 3. 生成 requirements.txt ----------
Write-Host "`n[3/4] Freezing dependencies ..." -ForegroundColor Yellow
$freeze = pip freeze
if ($LASTEXITCODE -ne 0) {
    throw "pip freeze failed."
}
$pattern = "^(fastapi|uvicorn|python-dotenv|langchain|langgraph|chromadb|rank-bm25|jieba|pypdf|pytest|httpx|langfuse)"
$lines = @($freeze | Select-String -Pattern $pattern | ForEach-Object { $_.Line })
if ($lines.Count -eq 0) {
    throw "No dependency matched. Check the active Python environment."
}
$requirementsPath = Join-Path $Root "requirements.txt"
[System.IO.File]::WriteAllLines($requirementsPath, $lines)
Write-Host "requirements.txt written:" -ForegroundColor Green
Get-Content $requirementsPath

# ---------- 4. Git 初始化 + 提交 ----------
Write-Host "`n[4/4] Initializing Git and committing ..." -ForegroundColor Yellow
if (-not (Test-Path ".git")) {
    git init | Out-Null
    Write-Host "git repository initialized." -ForegroundColor Green
}

# 关键保护：.env 必须被忽略，绝对不能提交
git check-ignore -q .env
if ($LASTEXITCODE -ne 0) {
    throw ".env is NOT ignored by .gitignore. Aborting before any commit."
}

if (-not (git config user.name) -or -not (git config user.email)) {
    throw "Git identity missing. Run: git config --global user.name ""Your Name""; git config --global user.email ""you@example.com"""
}

git add .
$staged = @(git diff --cached --name-only)
if ($staged -contains ".env") {
    throw ".env is staged. Aborting commit."
}
git status --short

git commit -m "feat: LangGraph education agent (RAG + tool calling + HITL)"
if ($LASTEXITCODE -ne 0) {
    Write-Host "Commit failed or nothing to commit. Check the output above." -ForegroundColor DarkYellow
    exit 1
}

Write-Host "`nDone: tests passed, requirements.txt generated, code committed." -ForegroundColor Green
Write-Host "Reminder: rotate OPENAI_API_KEY / LANGFUSE keys if .env was ever shared." -ForegroundColor Red

# Renewal Shield - deploy the web app to Azure Container Apps (PowerShell, run from the project folder).
#
# One-time setup:
#   1. Activate Azure for Students: https://azure.microsoft.com/free/students  (HWU email, no card)
#   2. Install the Azure CLI:  winget install -e --id Microsoft.AzureCLI   (then open a NEW terminal)
#   3. az login
#   4. python law_index.py --build   (if law/articles.json doesn't exist yet)
#
# Deploy, and re-deploy after every change:
#   .\deploy_azure.ps1
# If Azure says the region isn't allowed for your subscription:
#   .\deploy_azure.ps1 -Location westeurope
#
# The OpenAI key is read from .env and stored as an Azure *secret*; it never appears in the image,
# the command line or the logs. .dockerignore keeps .env, photos and data out of the upload.

param(
    [string]$Location = "uaenorth",          # UAE North (Dubai/Abu Dhabi)
    [string]$Group    = "renewal-shield-rg",
    [string]$App      = "renewal-shield"
)
$ErrorActionPreference = "Stop"

foreach ($f in @("law/articles.json", "renewal_shield_clause_bank_v0.3.xlsx", "Dockerfile", ".dockerignore")) {
    if (-not (Test-Path $f)) { throw "Missing $f - run this from the project folder (and python law_index.py --build)." }
}
$line = Select-String -Path .env -Pattern '^\s*OPENAI_API_KEY\s*=' | Select-Object -First 1
if (-not $line) { throw "OPENAI_API_KEY not found in .env" }
$key = $line.Line.Split("=", 2)[1].Trim().Trim('"').Trim("'")

Write-Host "== One-time Azure setup (safe to repeat) =="
az extension add --name containerapp --upgrade --only-show-errors
az provider register --namespace Microsoft.App --wait
az provider register --namespace Microsoft.OperationalInsights --wait
az group create --name $Group --location $Location --only-show-errors | Out-Null

Write-Host "== Building the image in Azure and deploying (first time ~10 min) =="
az containerapp up --name $App --resource-group $Group --location $Location `
    --source . --ingress external --target-port 7860

Write-Host "== Secret, size and scaling =="
az containerapp secret set --name $App --resource-group $Group --secrets "openai-key=$key" --only-show-errors | Out-Null
az containerapp update --name $App --resource-group $Group `
    --cpu 1 --memory 2Gi --min-replicas 0 --max-replicas 1 `
    --set-env-vars "OPENAI_API_KEY=secretref:openai-key" "RS_DAILY_LIMIT=150" "RS_PHOTOS_PER_HOUR=5" `
    --only-show-errors | Out-Null

$url = az containerapp show --name $App --resource-group $Group --query properties.configuration.ingress.fqdn -o tsv
Write-Host ""
Write-Host "Live at: https://$url"
Write-Host "Note: after ~10 idle minutes it sleeps; the first visit then takes ~30-60 s to wake up."

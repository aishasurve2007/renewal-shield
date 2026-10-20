# Renewal Shield - run the web app on Azure Container Apps (PowerShell, run from the project folder).
#
# How it works:
#   git push  ->  GitHub Actions builds the Docker image  ->  ghcr.io/<you>/<repo>:sha-<commit>
#   this script  ->  tells Azure to run that image (Azure for Students can't build images itself)
#
# One-time setup:
#   1. Azure for Students activated, Azure CLI installed, az login
#   2. Push the repo with .github/workflows/build-image.yml; wait for the green tick in the Actions tab
#   3. GitHub -> your profile -> Packages -> renewal-shield -> Package settings -> Change visibility -> Public
#      (Azure pulls the image without a password; the image holds code only, never .env)
#
# Deploy, and re-deploy after every change (push first, wait for the Actions tick):
#   .\deploy_azure.ps1
#
# The OpenAI key is read from .env and stored as an Azure *secret*; it never appears in the image,
# the command line or the logs.

param(
    [string]$Location = "germanywestcentral",  # UAE North isn't allowed on Azure for Students; Frankfurt is
    [string]$Group    = "renewal-shield-rg",
    [string]$AppEnv   = "renewal-shield-env",
    [string]$App      = "renewal-shield",
    [string]$Tag      = ""                     # default: the commit you last pushed
)
$ErrorActionPreference = "Stop"

# az is a separate program: PowerShell doesn't stop on its errors by itself, so check after each call
function Check($step) { if ($LASTEXITCODE -ne 0) { throw "Stopped: '$step' failed (see the error above)." } }

# --- which image: ghcr.io/<owner>/<repo>:sha-<commit>, from the git remote and the last pushed commit
$remote = git remote get-url origin; Check "read git remote"
if ($remote -notmatch "github\.com[:/](.+?)(\.git)?$") { throw "origin is not a GitHub repo: $remote" }
$repo = $Matches[1].ToLower()
if (-not $Tag) {
    git fetch --quiet origin; Check "git fetch"
    $sha = git rev-parse "@{u}"; Check "find the pushed commit (have you pushed?)"
    $Tag = "sha-$sha"
}
$image = "ghcr.io/${repo}:$Tag"
Write-Host "Image: $image"

$line = Select-String -Path .env -Pattern '^\s*OPENAI_API_KEY\s*=' | Select-Object -First 1
if (-not $line) { throw "OPENAI_API_KEY not found in .env" }
$key = $line.Line.Split("=", 2)[1].Trim().Trim('"').Trim("'")

Write-Host "== One-time Azure setup (safe to repeat) =="
az extension add --name containerapp --upgrade --only-show-errors; Check "install containerapp extension"
az provider register --namespace Microsoft.App --wait; Check "register Microsoft.App"
az provider register --namespace Microsoft.OperationalInsights --wait; Check "register Microsoft.OperationalInsights"
if ((az group exists --name $Group) -eq "false") {
    az group create --name $Group --location $Location --only-show-errors | Out-Null; Check "create resource group"
}
$envState = az containerapp env list --resource-group $Group --query "[?name=='$AppEnv'].properties.provisioningState" -o tsv
if ($envState -and $envState -ne "Succeeded") {
    # left half-created by an interrupted earlier attempt: delete it and start clean (~3-5 min)
    Write-Host "Environment is in state '$envState' - deleting and recreating it"
    az containerapp env delete --name $AppEnv --resource-group $Group --yes --only-show-errors
    Check "delete the broken environment"
    $envState = $null
}
if (-not $envState) {
    Write-Host "Creating the Container Apps environment in $Location (~3-5 min)"
    az containerapp env create --name $AppEnv --resource-group $Group --location $Location --only-show-errors | Out-Null
    Check "create Container Apps environment"
}
# The earlier attempt created an Azure Container Registry we can't use (no builds on student plans).
# It costs ~$5/month, so remove it.
foreach ($acr in (az acr list --resource-group $Group --query "[].name" -o tsv)) {
    if ($acr) { Write-Host "Removing unused registry $acr"; az acr delete --name $acr --resource-group $Group --yes --only-show-errors }
}

$appExists = az containerapp list --resource-group $Group --query "[?name=='$App'].name" -o tsv
if (-not $appExists) {
    Write-Host "== Creating the app (first time ~3-5 min) =="
    az containerapp create --name $App --resource-group $Group --environment $AppEnv `
        --image $image --ingress external --target-port 7860 `
        --cpu 1 --memory 2Gi --min-replicas 0 --max-replicas 1 `
        --secrets "openai-key=$key" `
        --env-vars "OPENAI_API_KEY=secretref:openai-key" "RS_DAILY_LIMIT=150" "RS_PHOTOS_PER_HOUR=5" `
        --only-show-errors | Out-Null
    Check "create the app (if 'image not found': is the Actions build finished and the package Public?)"
} else {
    Write-Host "== Updating the app to the new image =="
    az containerapp secret set --name $App --resource-group $Group --secrets "openai-key=$key" --only-show-errors | Out-Null
    Check "update the OpenAI key secret"
    az containerapp update --name $App --resource-group $Group --image $image --only-show-errors | Out-Null
    Check "update the app (if 'image not found': is the Actions build for this commit finished?)"
}

$url = az containerapp show --name $App --resource-group $Group --query properties.configuration.ingress.fqdn -o tsv
Write-Host ""
Write-Host "Live at: https://$url"
Write-Host "Note: after ~10 idle minutes it sleeps; the first visit then takes ~30-60 s to wake up."

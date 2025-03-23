Write-Host "Setting up Fractalic..."

# 1. Clone repositories if they don't exist
git clone https://github.com/fractalic-ai/fractalic.git
git clone https://github.com/fractalic-ai/fractalic-ui.git

# 2. Copy Docker files
Set-Location -Path ".\fractalic"
Copy-Item -Path ".\docker\*" -Destination "..\" -Recurse -Force
Set-Location -Path "..\"

# 3. Build Docker image
Write-Host "Building Docker image..."
docker build -t fractalic-app .

# 4. Check if container already exists and remove it
$existingContainer = docker ps -qa -f name=fractalic-app
if ($existingContainer) {
    Write-Host "Removing existing container..."
    docker stop fractalic-app > $null 2>&1
    docker rm fractalic-app > $null 2>&1
}

# 5. Run the container
Write-Host "Starting container..."
docker run -d `
  -p 8000:8000 `
  -p 3000:3000 `
  --name fractalic-app `
  fractalic-app

# 6. Wait for services to be ready
Write-Host "Waiting for services to start (this may take a moment)..."
Start-Sleep -Seconds 10

# 7. Check if services are running
$running = docker ps | Select-String "fractalic-app"
if ($running) {
    Write-Host "✅ Container is running"
    
    # Check if service is responding
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8000" -UseBasicParsing -TimeoutSec 5
        if ($response.StatusCode -eq 200) {
            Write-Host "✅ Backend is available at: http://localhost:8000"
        }
    }
    catch {
        Write-Host "⚠️ Backend may still be starting at: http://localhost:8000"
    }
    
    Write-Host "✅ UI should be available at: http://localhost:3000"
    Write-Host "Setup complete! Container is ready."
}
else {
    Write-Host "❌ Container failed to start. Check logs with: docker logs fractalic-app"
}

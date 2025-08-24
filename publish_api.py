#!/usr/bin/env python3
"""
Docker Publish API

FastAPI endpoint that provides Docker publishing functionality for the Fractalic UI.
This can be integrated into the main UI server or run as a separate service.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, Dict, Any
import asyncio
import subprocess
import json
import os
from pathlib import Path
import time
import uuid

# Import the publisher class
from publish_docker import FractalicDockerPublisher

app = FastAPI(title="Fractalic Docker Publisher API", version="1.0.0")

# Store for tracking publish operations
publish_operations = {}

class PublishRequest(BaseModel):
    container_name: Optional[str] = "fractalic-published"
    port_offset: Optional[int] = 0
    keep_temp: Optional[bool] = False

class PublishStatus(BaseModel):
    operation_id: str
    status: str  # "pending", "running", "completed", "failed"
    message: str
    container_name: Optional[str] = None
    ports: Optional[Dict[str, int]] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None

async def run_publish_operation(operation_id: str, request: PublishRequest):
    """Run the publish operation in background"""
    try:
        # Update status to running
        publish_operations[operation_id]["status"] = "running"
        publish_operations[operation_id]["message"] = "Publishing container..."
        
        # Create publisher instance
        publisher = FractalicDockerPublisher(
            container_name=request.container_name,
            port_offset=request.port_offset
        )
        
        if request.keep_temp:
            publisher.cleanup = lambda: None
        
        # Run the publish process
        success = publisher.publish()
        
        if success:
            publish_operations[operation_id]["status"] = "completed"
            publish_operations[operation_id]["message"] = "Container published successfully"
            publish_operations[operation_id]["container_name"] = request.container_name
            publish_operations[operation_id]["ports"] = publisher.host_ports  # Use host_ports instead of ports
            publish_operations[operation_id]["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        else:
            publish_operations[operation_id]["status"] = "failed"
            publish_operations[operation_id]["message"] = "Publication failed"
            publish_operations[operation_id]["error"] = "See logs for details"
            publish_operations[operation_id]["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            
    except Exception as e:
        publish_operations[operation_id]["status"] = "failed"
        publish_operations[operation_id]["message"] = "Publication failed with error"
        publish_operations[operation_id]["error"] = str(e)
        publish_operations[operation_id]["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

@app.post("/publish", response_model=Dict[str, str])
async def publish_container(request: PublishRequest, background_tasks: BackgroundTasks):
    """
    Start a Docker container publish operation
    
    This endpoint starts the publish process in the background and returns
    an operation ID that can be used to track progress.
    """
    # Generate unique operation ID
    operation_id = str(uuid.uuid4())
    
    # Initialize operation tracking
    publish_operations[operation_id] = {
        "operation_id": operation_id,
        "status": "pending",
        "message": "Preparing to publish container...",
        "container_name": request.container_name,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "ports": None,
        "completed_at": None,
        "error": None
    }
    
    # Start background task
    background_tasks.add_task(run_publish_operation, operation_id, request)
    
    return {
        "operation_id": operation_id,
        "status": "pending",
        "message": "Publish operation started"
    }

@app.get("/publish/status/{operation_id}", response_model=PublishStatus)
async def get_publish_status(operation_id: str):
    """
    Get the status of a publish operation
    """
    if operation_id not in publish_operations:
        raise HTTPException(status_code=404, detail="Operation not found")
    
    return PublishStatus(**publish_operations[operation_id])

@app.get("/publish/operations", response_model=Dict[str, PublishStatus])
async def list_publish_operations():
    """
    List all publish operations
    """
    return {
        op_id: PublishStatus(**op_data) 
        for op_id, op_data in publish_operations.items()
    }

@app.delete("/publish/operations/{operation_id}")
async def delete_publish_operation(operation_id: str):
    """
    Delete a publish operation from tracking
    """
    if operation_id not in publish_operations:
        raise HTTPException(status_code=404, detail="Operation not found")
    
    del publish_operations[operation_id]
    return {"message": "Operation deleted"}

@app.post("/containers/{container_name}/stop")
async def stop_container(container_name: str):
    """
    Stop a running Docker container
    """
    try:
        # Stop the container
        result = subprocess.run(
            ["docker", "stop", container_name],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            return {"message": f"Container {container_name} stopped successfully"}
        else:
            raise HTTPException(
                status_code=400, 
                detail=f"Failed to stop container: {result.stderr}"
            )
            
    except subprocess.TimeoutExpired:
        raise HTTPException(
            status_code=408, 
            detail="Container stop operation timed out"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error stopping container: {str(e)}"
        )

@app.post("/containers/{container_name}/remove")
async def remove_container(container_name: str):
    """
    Remove a Docker container
    """
    try:
        # Remove the container
        result = subprocess.run(
            ["docker", "rm", container_name],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            return {"message": f"Container {container_name} removed successfully"}
        else:
            raise HTTPException(
                status_code=400, 
                detail=f"Failed to remove container: {result.stderr}"
            )
            
    except subprocess.TimeoutExpired:
        raise HTTPException(
            status_code=408, 
            detail="Container removal operation timed out"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error removing container: {str(e)}"
        )

@app.get("/containers", response_model=Dict[str, Any])
async def list_containers():
    """
    List Docker containers (running and stopped)
    """
    try:
        # Get all containers
        result = subprocess.run(
            ["docker", "ps", "-a", "--format", "json"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to list containers: {result.stderr}"
            )
        
        containers = []
        for line in result.stdout.strip().split('\n'):
            if line:
                try:
                    containers.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        
        return {"containers": containers}
        
    except subprocess.TimeoutExpired:
        raise HTTPException(
            status_code=408, 
            detail="Container listing operation timed out"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error listing containers: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy", "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)

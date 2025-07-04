#!/usr/bin/env python3
"""
Lightweight standalone Fractalic execution server.
Runs on a separate port from the existing UI server.
Executes Fractalic scripts via the fractalic module API.
"""

import os
import sys
import tempfile
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Add fractalic to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from fractalic import run_fractalic


class ExecuteRequest(BaseModel):
    filename: str
    parameter_text: Optional[str] = None


app = FastAPI(title="Fractalic AI Server", version="1.0.0")


@app.post("/execute")
async def execute_script(request: ExecuteRequest):
    """
    Execute a Fractalic script.
    
    Args:
        filename: Path to the Fractalic script to execute
        parameter_text: Optional parameter text to inject into the script
    
    Returns:
        dict: Execution result with success status, output, branch_name, explicit_return, and return_content
    """
    try:
        # Validate file exists
        if not os.path.exists(request.filename):
            return {
                'success': False,
                'error': f"File not found: {request.filename}",
                'output': '',
                'explicit_return': False,
                'return_content': None,
                'branch_name': None,
                'ctx_file': None
            }
        
        # Set working directory to the script's directory
        script_dir = os.path.dirname(os.path.abspath(request.filename))
        original_cwd = os.getcwd()
        
        try:
            # Change to fractalic directory to ensure settings.toml is accessible
            fractalic_dir = os.path.dirname(current_dir)  # parent_dir is the fractalic directory
            os.chdir(fractalic_dir)
            
            # Prepare parameters
            task_file = None
            param_input_user_request = None
            
            # Handle parameter injection
            if request.parameter_text:
                # Create temporary parameter file
                temp_dir = tempfile.mkdtemp()
                task_file = os.path.join(temp_dir, "parameters.md")
                
                with open(task_file, 'w', encoding='utf-8') as f:
                    f.write("# Input Parameters {id=input-parameters}\n\n")
                    f.write(request.parameter_text)
                
                param_input_user_request = 'input-parameters'
            
            # Execute Fractalic using the module API
            result = run_fractalic(
                input_file=request.filename,
                task_file=task_file,
                param_input_user_request=param_input_user_request,
                capture_output=True
            )
            
            # Cleanup temp file if created
            if task_file and os.path.exists(task_file):
                os.unlink(task_file)
                os.rmdir(temp_dir)
            
            # Debug: log what we got from run_fractalic
            # print(f"[DEBUG] run_fractalic returned: {result}")
            
            # Ensure we always return the expected format
            if not isinstance(result, dict):
                result = {'success': False, 'error': 'Invalid result format'}
            
            # Handle the different return formats from run_fractalic
            if 'success' in result:
                # run_fractalic returned error format
                if not result['success']:
                    return {
                        'success': False,
                        'error': result.get('error', 'Unknown error'),
                        'output': result.get('output', ''),
                        'explicit_return': False,
                        'return_content': None,
                        'branch_name': None,
                        'ctx_file': None
                    }
            
            # Normal successful execution - format the response
            return {
                'success': True,
                'explicit_return': result.get('explicit_return', False),
                'return_content': result.get('return_content', None),
                'branch_name': result.get('branch_name', None),
                'ctx_file': None,  # Add this field for compatibility
                'output': result.get('output', '')
            }
            
        finally:
            os.chdir(original_cwd)
            
    except Exception as e:
        return {
            'success': False,
            'error': f"Execution failed: {str(e)}",
            'output': '',
            'explicit_return': False,
            'return_content': None,
            'branch_name': None,
            'ctx_file': None
        }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


def main():
    """Start the Fractalic AI Server."""
    import uvicorn
    
    # Try different ports if 8001 is in use
    ports_to_try = [8001, 8002, 8003, 8004]
    
    for port in ports_to_try:
        try:
            print(f"🚀 Starting Fractalic AI Server on port {port}...")
            print(f"📍 Server will be available at: http://localhost:{port}")
            print(f"📚 API docs: http://localhost:{port}/docs")
            
            uvicorn.run(app, host="0.0.0.0", port=port)
            break
        except OSError as e:
            if "Address already in use" in str(e):
                print(f"❌ Port {port} is already in use, trying next port...")
                continue
            else:
                raise


if __name__ == "__main__":
    main()
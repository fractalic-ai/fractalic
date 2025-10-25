# server.py
import asyncio
import sys
import time
import threading
import uuid
import aiohttp
import shutil
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import PlainTextResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import git
from pathlib import Path
import json
import os
import toml
from fastapi.responses import FileResponse, Response
import logging
import subprocess
import signal
import aiohttp
import time
from typing import Optional, Dict, Any, List
from collections import deque
import threading
from datetime import datetime
import pathlib

from core.events.types import EventType

# --- Robust import for ToolRegistry regardless of working directory ---
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import our centralized path management system
from core.paths import (
    get_fractalic_root, 
    get_session_root, 
    get_session_cwd,
    get_tools_directory,
    get_logs_directory
)

from core.plugins.tool_registry import ToolRegistry

logging.getLogger("git").setLevel(logging.CRITICAL)
logging.getLogger("git.cmd").setLevel(logging.CRITICAL)

app = FastAPI()

# Dev mode check - disable caching for static files during development
DEV_MODE = os.environ.get("FRACTALIC_DEV_MODE", "0") not in ("0", "false", "False", "")
if DEV_MODE:
    print("⚡ DEV MODE: Static file caching disabled")

current_repo_path = ""

# MCP Manager process management
mcp_manager_process: Optional[subprocess.Popen] = None
# Trace log (ring buffer) capturing lifecycle, stdout/stderr lines, API interactions
TRACE_MAX_ENTRIES = 10000
_trace_buffer = deque(maxlen=TRACE_MAX_ENTRIES)
_trace_lock = threading.Lock()
_trace_log_path: Optional[Path] = None
_trace_log_max_bytes = 2_000_000  # 2 MB
_trace_log_backups = 3
_stdout_thread: Optional[threading.Thread] = None
_stderr_thread: Optional[threading.Thread] = None
_process_monitor_task: Optional[asyncio.Task] = None
_mcp_state: Dict[str, Any] = {"phase": "idle", "last_exit_code": None, "last_error": None, "start_time": None}

# Trace enable/disable flag (default disabled). Set MCP_TRACE_ENABLED=1 to enable.
_TRACE_ENABLED = os.environ.get("MCP_TRACE_ENABLED", "0") not in ("0", "false", "False", "")

mcp_manager_port = 5859
mcp_manager_url = f"http://localhost:{mcp_manager_port}"

# Set BASE_DIR to the root directory to allow navigation to parent directories
BASE_DIR = Path('/').resolve()

# Mount static files (HTML, CSS, JS)
# app.mount("/static", StaticFiles(directory="static"), name="static")

# Enable CORS for static files
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins since everything is local
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware to disable caching in dev mode
@app.middleware("http")
async def disable_caching_in_dev_mode(request: Request, call_next):
    response = await call_next(request)
    if DEV_MODE and (request.url.path.startswith("/chat/") or request.url.path == "/chat"):
        # Disable caching for all static files in dev mode
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response

# Mount static files for modular web structure OR fallback to legacy file
root = get_fractalic_root()
web_dir = Path(root) / "web"

if web_dir.exists():
    try:
        # Mount /chat to serve the entire web directory with index.html as default
        app.mount("/chat", StaticFiles(directory=str(web_dir), html=True), name="chat")
        print(f"✅ Mounted /chat → {web_dir}")
    except Exception as e:
        print(f"Warning: Could not mount /chat static files: {e}")
else:
    # Fallback: serve legacy fractalic_chat.html at /chat if web/ doesn't exist
    print(f"ℹ️  web/ folder not found, using fallback endpoint for fractalic_chat.html")

    @app.get("/chat")
    async def serve_chat_fallback():
        try:
            root = get_fractalic_root()
            candidate_names = ["fractalic_chat.html", "fractalic_chat_fixed.html"]
            for name in candidate_names:
                p = Path(root) / name
                if p.exists():
                    print(f"📄 Serving {name} from {p}")
                    return FileResponse(str(p), media_type="text/html")
            raise HTTPException(status_code=404, detail="Chat client html not found")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load chat html: {e}")

# Define the settings file path using centralized path management
def get_current_settings_path():
    """Get the settings file path for UI server - always use global settings"""
    # UI server always works with global settings from fractalic_root
    return os.path.join(get_fractalic_root(), 'settings.toml')

def set_repo_path(path: str):
    """Set the current repository path globally"""
    global current_repo_path
    current_repo_path = path

# MCP Manager process management functions
async def _cleanup_port_conflicts(port: int):
    """Find and terminate processes using the specified port"""
    try:
        # Use lsof to find processes using the port
        result = subprocess.run(
            ['lsof', '-ti', f':{port}'],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0 and result.stdout.strip():
            pids = [pid.strip() for pid in result.stdout.strip().split('\n') if pid.strip()]
            logging.info(f"Found {len(pids)} process(es) using port {port}: {pids}")
            
            for pid in pids:
                try:
                    # Try graceful termination first
                    subprocess.run(['kill', pid], timeout=5)
                    await asyncio.sleep(1)
                    
                    # Check if still running, force kill if needed
                    check_result = subprocess.run(['kill', '-0', pid], capture_output=True, timeout=2)
                    if check_result.returncode == 0:
                        subprocess.run(['kill', '-9', pid], timeout=5)
                        logging.info(f"Force killed process {pid} on port {port}")
                    else:
                        logging.info(f"Gracefully terminated process {pid} on port {port}")
                        
                except subprocess.TimeoutExpired:
                    logging.warning(f"Timeout while trying to kill process {pid}")
                except Exception as e:
                    logging.warning(f"Failed to kill process {pid}: {e}")
            
            # Wait a moment for port to be freed
            await asyncio.sleep(2)
            
    except subprocess.TimeoutExpired:
        logging.warning(f"Timeout while checking for port conflicts on {port}")
    except FileNotFoundError:
        # lsof not available, skip port cleanup
        logging.debug("lsof command not available, skipping port cleanup")
    except Exception as e:
        logging.warning(f"Error during port cleanup for {port}: {e}")

# ---------------- Trace / Monitoring Utilities (top-level) ----------------
def _record_trace(event_type: str, message: str = "", **fields):
    if not _TRACE_ENABLED:
        return
    record = {
        "ts": datetime.utcnow().isoformat(timespec='milliseconds') + 'Z',
        "type": event_type,
        "message": message,
        **fields
    }
    with _trace_lock:
        _trace_buffer.append(record)
        if _trace_log_path:
            try:
                _write_trace_line(record)
            except Exception:
                pass

def _init_trace_log(base_dir: Optional[str] = None):
    """Initialize on-disk trace log location (idempotent)."""
    if not _TRACE_ENABLED:
        return
    global _trace_log_path
    if _trace_log_path is not None:
        return
    
    # Use centralized path management for logs directory
    try:
        logs_dir = get_logs_directory()
    except:
        # Fallback to fractalic_root/logs if no session is set
        logs_dir = Path(get_fractalic_root()) / 'logs'
    
    logs_dir.mkdir(parents=True, exist_ok=True)
    _trace_log_path = logs_dir / 'mcp_trace.log'
    _record_trace('trace', message='trace file initialized', path=str(_trace_log_path))

def _write_trace_line(record: Dict[str, Any]):
    if not _trace_log_path:
        return
    path = _trace_log_path
    line = json.dumps(record, ensure_ascii=False)
    rotate = False
    try:
        if path.exists() and path.stat().st_size + len(line) > _trace_log_max_bytes:
            rotate = True
    except Exception:
        rotate = False
    if rotate:
        # Rotate backups
        for i in range(_trace_log_backups, 0, -1):
            src = path.with_suffix(path.suffix + f'.{i}')
            if src.exists():
                if i == _trace_log_backups:
                    try:
                        src.unlink()
                    except Exception:
                        pass
                else:
                    try:
                        src.rename(path.with_suffix(path.suffix + f'.{i+1}'))
                    except Exception:
                        pass
        try:
            path.rename(path.with_suffix(path.suffix + '.1'))
        except Exception:
            pass
    try:
        with path.open('a', encoding='utf-8') as f:
            f.write(line + '\n')
    except Exception:
        pass

def _stream_pipe_to_trace(pipe, stream_name: str, pid: int):
    try:
        for line in iter(pipe.readline, ''):
            if not line:
                break
            _record_trace(stream_name, line=line.rstrip('\n'), pid=pid)
    except Exception as e:
        _record_trace("error", message=f"stream reader failed ({stream_name})", error=str(e))
    finally:
        try:
            pipe.close()
        except Exception:
            pass

async def _monitor_mcp_process():
    global mcp_manager_process
    try:
        while mcp_manager_process and mcp_manager_process.poll() is None:
            await asyncio.sleep(1)
        if mcp_manager_process:
            exit_code = mcp_manager_process.poll()
            _mcp_state["last_exit_code"] = exit_code
            _mcp_state["phase"] = "exited"
            _record_trace("lifecycle", message="process exited", exit_code=exit_code)
    except asyncio.CancelledError:
        _record_trace("monitor", message="process monitor cancelled")
    except Exception as e:
        _record_trace("error", message="process monitor error", error=str(e))

@app.get("/mcp/trace")
async def get_mcp_trace(tail: int = Query(200, ge=1, le=5000)):
    if not _TRACE_ENABLED:
        return {"disabled": True, "reason": "Tracing disabled. Set MCP_TRACE_ENABLED=1 to enable."}
    with _trace_lock:
        data = list(_trace_buffer)[-tail:]
    return {"tail": tail, "size": len(_trace_buffer), "entries": data}

@app.post("/mcp/trace/clear")
async def clear_mcp_trace():
    if not _TRACE_ENABLED:
        return {"disabled": True, "status": "noop"}
    with _trace_lock:
        _trace_buffer.clear()
    _record_trace("trace", message="trace cleared")
    return {"status": "cleared"}

async def _verify_mcp_api_ready(timeout: int = 10) -> bool:
    """Fast readiness probe.
    1. Prefer /health (constant‑time, no tool listing)
    2. Fallback to /status only if /health not yet available
    This prevents slow/blocked service (e.g. notion streamable-http init) from delaying api_ready.
    """
    start_time = time.time()
    attempt = 0
    while (time.time() - start_time) < timeout:
        attempt += 1
        try:
            async with aiohttp.ClientSession() as session:
                # First: lightweight /health
                try:
                    async with session.get(f"{mcp_manager_url}/health", timeout=2) as r_health:
                        if r_health.status == 200:
                            h = await r_health.json()
                            if h.get("status") == "healthy":
                                _record_trace("health", message=f"health ok (fast) attempt={attempt}")
                                return True
                        else:
                            _record_trace("health", message=f"health non-200={r_health.status}")
                except asyncio.TimeoutError:
                    _record_trace("health", message="health timeout")
                except Exception as e_h:
                    _record_trace("health", message=f"health error: {str(e_h)[:80]}")
                # Second: fallback /status (heavier)
                try:
                    async with session.get(f"{mcp_manager_url}/status", timeout=5) as r_status:
                        if r_status.status == 200:
                            data = await r_status.json()
                            if 'services' in data and 'total_services' in data:
                                _record_trace("health", message=f"status ok services={data.get('total_services')}")
                                return True
                            else:
                                _record_trace("health", message="status missing keys")
                        else:
                            _record_trace("health", message=f"status non-200={r_status.status}")
                except asyncio.TimeoutError:
                    _record_trace("health", message="status timeout")
                except Exception as e_s:
                    _record_trace("health", message=f"status error: {str(e_s)[:80]}")
        except Exception as outer:
            logging.debug(f"Readiness outer failure: {outer}")
            _record_trace("health", message=f"outer failure: {str(outer)[:80]}")
        await asyncio.sleep(1)
    return False

async def start_mcp_manager():
    """Start the MCP manager process with proper port management"""
    global mcp_manager_process
    global _stdout_thread, _stderr_thread, _process_monitor_task
    
    if mcp_manager_process and mcp_manager_process.poll() is None:
        _record_trace("lifecycle", message="start requested while already running", pid=mcp_manager_process.pid)
        return {"status": "already_running", "pid": mcp_manager_process.pid}
    try:
        _mcp_state.update({"phase": "starting", "start_time": time.time(), "last_exit_code": None, "last_error": None})
        _init_trace_log()
        _record_trace("lifecycle", message="starting mcp manager")
        
        # Use fractalic_root for MCP manager script location
        fractalic_root = get_fractalic_root()
        mcp_manager_script = Path(fractalic_root) / "fractalic_mcp_manager.py"
        if not mcp_manager_script.exists():
            raise HTTPException(status_code=500, detail=f"MCP manager script not found at {mcp_manager_script}")
        
        await _cleanup_port_conflicts(mcp_manager_port)
        env = os.environ.copy()
        env.update({'PYTHONIOENCODING': 'utf-8','LC_ALL': 'en_US.UTF-8','LANG': 'en_US.UTF-8','PYTHONUNBUFFERED': '1'})
        
        # Use Python from virtual environment if available
        venv_python = Path(fractalic_root) / ".venv" / "bin" / "python"
        python_executable = str(venv_python) if venv_python.exists() else sys.executable
        
        mcp_manager_process = subprocess.Popen(
            [python_executable, str(mcp_manager_script), "serve", "--port", str(mcp_manager_port), "--host", "localhost"],
            cwd=fractalic_root,  # Always run MCP manager from fractalic_root
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env
        )
        _record_trace("lifecycle", message="process spawned", pid=mcp_manager_process.pid)
        if mcp_manager_process.stdout:
            _stdout_thread = threading.Thread(target=_stream_pipe_to_trace, args=(mcp_manager_process.stdout, "stdout", mcp_manager_process.pid), daemon=True)
            _stdout_thread.start()
        if mcp_manager_process.stderr:
            _stderr_thread = threading.Thread(target=_stream_pipe_to_trace, args=(mcp_manager_process.stderr, "stderr", mcp_manager_process.pid), daemon=True)
            _stderr_thread.start()
        loop = asyncio.get_running_loop()
        _process_monitor_task = loop.create_task(_monitor_mcp_process())
        await asyncio.sleep(3)
        if mcp_manager_process.poll() is not None:
            stdout, stderr = mcp_manager_process.communicate()
            if "address already in use" in stderr:
                error_msg = f"Port {mcp_manager_port} is still in use. Failed to clean up existing processes."
            else:
                error_msg = f"MCP manager failed to start. stderr: {stderr[:500]}..."
            _record_trace("lifecycle", message="startup failed", stderr_tail=stderr[-500:], exit_code=mcp_manager_process.poll())
            mcp_manager_process = None
            raise HTTPException(status_code=500, detail=error_msg)
        api_ready = await _verify_mcp_api_ready(timeout=15)
        if not api_ready:
            logging.warning("MCP manager process started but API is not responding yet")
            _record_trace("health", message="api not ready within initial timeout")
        else:
            _record_trace("health", message="api ready")
            _mcp_state["phase"] = "running"
            # Additional check after short delay to ensure stability
            await asyncio.sleep(2)
            stable_check = await _verify_mcp_api_ready(timeout=3)
            if not stable_check:
                logging.warning("MCP manager API became unresponsive after initial readiness")
                _record_trace("health", message="api became unstable after readiness check")
                api_ready = False
        return {"status": "started", "pid": mcp_manager_process.pid, "port": mcp_manager_port, "api_ready": api_ready}
    except Exception as e:
        mcp_manager_process = None
        _mcp_state.update({"phase": "error", "last_error": str(e)})
        _record_trace("error", message="exception during start", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to start MCP manager: {str(e)}")

async def stop_mcp_manager():
    """Stop the MCP manager process"""
    global mcp_manager_process
    global _process_monitor_task
    
    if not mcp_manager_process or mcp_manager_process.poll() is not None:
        return {"status": "not_running"}
    
    try:
        # First, try to gracefully shutdown via API
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{mcp_manager_url}/kill") as response:
                    if response.status == 200:
                        # Wait for process to terminate
                        for _ in range(10):  # Wait up to 10 seconds
                            if mcp_manager_process.poll() is not None:
                                break
                            await asyncio.sleep(1)
        except:
            pass  # Graceful shutdown failed, continue with forceful termination
        
        # If still running, force terminate
        if mcp_manager_process.poll() is None:
            mcp_manager_process.terminate()
            
            # Wait for termination
            for _ in range(5):  # Wait up to 5 seconds
                if mcp_manager_process.poll() is not None:
                    break
                await asyncio.sleep(1)
            
            # If still running, kill forcefully
            if mcp_manager_process.poll() is None:
                mcp_manager_process.kill()
                mcp_manager_process.wait()
        
        _record_trace("lifecycle", message="process stopped", pid=mcp_manager_process.pid)
        pid = mcp_manager_process.pid
        mcp_manager_process = None
        _mcp_state["phase"] = "stopped"
        if _process_monitor_task:
            _process_monitor_task.cancel()
            _process_monitor_task = None
        return {"status": "stopped", "pid": pid}
        
    except Exception as e:
        _record_trace("error", message="stop failed", error=str(e))
        return {"status": "error", "error": str(e)}

async def get_mcp_manager_status():
    """Get the status of the MCP manager process"""
    global mcp_manager_process
    
    # First, try to connect to the MCP manager API regardless of how it was started
    api_status = None
    mcp_data = None
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{mcp_manager_url}/status", timeout=30) as response:
                if response.status == 200:
                    mcp_data = await response.json()
                    api_status = "responsive"
                else:
                    api_status = f"not_responsive_http_{response.status}"
    except Exception as e:
        api_status = f"not_responsive: {str(e)}"
    _mcp_state["api_status"] = api_status
    
    # Check if we have a process we started
    if mcp_manager_process:
        poll_result = mcp_manager_process.poll()
        if poll_result is not None:
            # Our process has terminated
            pid = mcp_manager_process.pid if mcp_manager_process else None
            mcp_manager_process = None
            
            # But check if API is still responsive (maybe manually started)
            if api_status == "responsive":
                return {
                    "status": "running_external",
                    "running": True,
                    "api_responsive": True,
                    "port": mcp_manager_port,
                    "managed_process": False,
                    "note": "MCP manager is running but not managed by UI server",
                    "servers": mcp_data,
                    "last_managed_pid": pid,
                    "exit_code": poll_result
                }
            else:
                return {
                    "status": "terminated",
                    "running": False,
                    "api_responsive": False,
                    "exit_code": poll_result,
                    "last_pid": pid
                }
        else:
            # Our process is still running
            if api_status == "responsive":
                return {
                    "status": "running",
                    "running": True,
                    "pid": mcp_manager_process.pid,
                    "port": mcp_manager_port,
                    "api_responsive": True,
                    "managed_process": True,
                    "servers": mcp_data
                }
            else:
                return {
                    "status": "running_not_responsive",
                    "running": True,
                    "pid": mcp_manager_process.pid,
                    "port": mcp_manager_port,
                    "api_responsive": False,
                    "managed_process": True,
                    "api_error": api_status
                }
    else:
        # No managed process, but check if API is responsive
        if api_status == "responsive":
            return {
                "status": "running_external",
                "running": True,
                "api_responsive": True,
                "port": mcp_manager_port,
                "managed_process": False,
                "note": "MCP manager is running but not managed by UI server",
                "servers": mcp_data
            }
        else:
            return {
                "status": "not_started",
                "running": False,
                "api_responsive": False,
                "managed_process": False,
                "api_error": api_status
            }

# Cleanup function for graceful shutdown
async def cleanup_mcp_manager():
    """Cleanup MCP manager process on shutdown"""
    if mcp_manager_process and mcp_manager_process.poll() is None:
        try:
            await stop_mcp_manager()
        except:
            pass

# Helper functions
def get_repo(repo_path: str):
    """Get Git repository object (old compatibility, rarely used now)."""
    resolved_path = Path(repo_path).resolve()
    if not resolved_path.is_dir():
        raise HTTPException(status_code=400, detail="Invalid repository path")
    try:
        repo = git.Repo(resolved_path)
        set_repo_path(resolved_path)
        return repo
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error initializing git repository: {e}")

def get_file_content(repo, commit_hash, filepath):
    try:
        file_content = repo.git.show(f'{commit_hash}:{filepath}')
        return file_content
    except Exception as e:
        print(f"Error fetching '{filepath}' at commit '{commit_hash}': {e}")
        return None

def ensure_empty_lines_before_symbols(content: str) -> str:
    """Ensure there's a blank line before 'def' or 'class' if missing for readability.
    This is a lightweight formatter to avoid NameError where previous helper was referenced."""
    lines = content.splitlines()
    out = []
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if (stripped.startswith('def ') or stripped.startswith('class ')) and out:
            if out[-1].strip() != '':
                out.append('')
        out.append(line)
    return '\n'.join(out)

def enrich_call_tree(node, repo):
    operation_src_list = node.get('operation_src', [])
    if operation_src_list and isinstance(operation_src_list[0], str):
        operation_src = operation_src_list[0]
    else:
        operation_src = "No Operation Source"

    filename = node.get('filename')
    ctx_file = node.get('ctx_file')
    trc_file = node.get('trc_file')  # Add this line to get trc_file
    md_commit_hash = node.get('md_commit_hash')
    ctx_commit_hash = node.get('ctx_commit_hash')
    trc_commit_hash = node.get('trc_commit_hash')  # Add this line to get trc_commit_hash

    source_content = get_file_content(repo, md_commit_hash, filename)
    target_content = get_file_content(repo, ctx_commit_hash, ctx_file)
    trace_content = get_file_content(repo, trc_commit_hash, trc_file)
      
    node['source_content'] = ensure_empty_lines_before_symbols(source_content)
    node['target_content'] = ensure_empty_lines_before_symbols(target_content)
    node['trace_content'] = trace_content  # Add trace_content to node
    node['operation_src'] = operation_src

    children = node.get('children', [])
    enriched_children = []
    for child in children:
        enriched_child = enrich_call_tree(child, repo)
        enriched_children.append(enriched_child)
    node['children'] = enriched_children

    return node

@app.get("/serve_image/")
async def serve_image(path: str = Query(...)):
    """
    Serves an image file from a path relative to the current repository path
    that was set during get_repo call.
    """
    if not current_repo_path:
        raise HTTPException(
            status_code=400, 
            detail="Repository path not set. Call get_repo first."
        )
    
    image_path = os.path.join(current_repo_path, path)
    
    if not os.path.isfile(image_path):
        raise HTTPException(status_code=404, detail="Image not found")

    response = FileResponse(image_path)
    response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/list_directory/")
async def list_directory(path: str = Query("")):
    resolved_path = Path(path).resolve()
    if not resolved_path.is_dir():
        raise HTTPException(status_code=400, detail="Invalid directory path")

    items = []
    for item in resolved_path.iterdir():
        is_dir = item.is_dir()
        is_git_repo = False
        # Check for .fractalic directory (storage-based sessions) instead of .git
        if is_dir and (item / '.fractalic').is_dir():
            is_git_repo = True
        items.append({
            'name': item.name,
            'path': str(item.resolve()),
            'is_dir': is_dir,
            'is_git_repo': is_git_repo
        })

    # Sort items: directories first, then files
    items.sort(key=lambda x: (not x['is_dir'], x['name'].lower()))

    if resolved_path != resolved_path.root:
        parent_path = str(resolved_path.parent.resolve())
        items.insert(0, {
            'name': '..',
            'path': parent_path,
            'is_dir': True,
            'is_git_repo': False
        })

    return items

@app.get("/branches_and_commits/")
async def get_branches_and_commits(repo_path: str = Query(...)):
    """List storage sessions as 'branches' for frontend."""
    from core.ui_server.storage_git_bridge import list_storage_sessions_as_branches

    try:
        # Return ONLY storage sessions (execution_id as "branch name")
        return list_storage_sessions_as_branches(repo_path=repo_path)
    except Exception as e:
        print(f"Error listing storage sessions: {str(e)}")
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"detail": f"Internal Server Error: {str(e)}"})

@app.get("/get_file_content/")
async def get_file_content_endpoint(
    repo_path: str = Query(...),
    file_path: str = Query(...),
    commit_hash: str = Query(...)
):
    """Return file content for the storage-backed session system.

    Frontend queries the same endpoint twice when building a diff:
    1) once for the original markdown (`file_path` relative to project root);
    2) once for the produced context/trace (`file_path` under `artifacts/nodes/...`).

    This resolver inspects the recorded sessions and deterministically selects
    the execution_id that owns the requested file.
    """
    from core.ui_server.storage_git_bridge import get_artifact_from_storage

    try:
        from pathlib import Path
        import json
        import hashlib

        sessions_dir = Path(repo_path) / '.fractalic' / 'sessions'
        if not sessions_dir.exists():
            raise HTTPException(status_code=404, detail="No sessions found")

        sessions = [p for p in sessions_dir.iterdir() if p.is_dir()]
        if not sessions:
            raise HTTPException(status_code=404, detail="No sessions found")

        # Prefer newest sessions first (mtime descending) for faster resolution in practice
        sessions.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        execution_id = None
        is_artifact_request = file_path.startswith('artifacts/')

        # 1) Direct path resolution for artifacts (unique per session)
        if is_artifact_request:
            for session_dir in sessions:
                candidate = session_dir / file_path
                if candidate.exists():
                    execution_id = session_dir.name
                    break

        # 2) Inspect call_tree metadata for md/ctx/trc entries
        if execution_id is None:
            def node_matches(tree: dict) -> bool:
                stack = [tree]
                while stack:
                    node = stack.pop()
                    if (
                        node.get('ctx_file') == file_path
                        and node.get('ctx_commit_hash') == commit_hash
                    ):
                        return True
                    if (
                        node.get('trc_file') == file_path
                        and node.get('trc_commit_hash') == commit_hash
                    ):
                        return True
                    if (
                        node.get('filename') == file_path
                        and node.get('md_commit_hash') == commit_hash
                    ):
                        return True
                    stack.extend(node.get('children', []) or [])
                return False

            for session_dir in sessions:
                meta_file = session_dir / 'metadata' / 'call_tree.json'
                if not meta_file.exists():
                    continue
                try:
                    with meta_file.open('r', encoding='utf-8') as f:
                        tree = json.load(f)
                except Exception:
                    continue
                if tree and node_matches(tree):
                    execution_id = session_dir.name
                    break

        # 3) Fallback: hash workspace file and compare with commit (for legacy sessions)
        if execution_id is None and not is_artifact_request:
            for session_dir in sessions:
                workspace_file = session_dir / 'workspace' / file_path
                if not workspace_file.exists():
                    continue
                try:
                    with workspace_file.open('r', encoding='utf-8') as wf:
                        content = wf.read()
                except Exception:
                    continue
                computed_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()[:40]
                if computed_hash == commit_hash:
                    execution_id = session_dir.name
                    break

        if execution_id is None:
            raise HTTPException(status_code=404, detail=f"Content mapping not found for {file_path} @ {commit_hash}")

        content = get_artifact_from_storage(execution_id, commit_hash, file_path, repo_path)
        if content is None:
            raise HTTPException(status_code=404, detail=f"File not found in session {execution_id}: {file_path}")
        return PlainTextResponse(content)
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error fetching file content: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.get("/get_enriched_call_tree/")
async def get_enriched_call_tree(repo_path: str = Query(...), branch: str = Query(...)):
    """Get enriched call tree using Git adapter for storage."""
    from core.ui_server.storage_git_bridge import is_storage_session, load_call_tree_from_storage
    from core.storage.git_api_adapter import get_repo_adapter

    try:
        # 'branch' is actually execution_id
        if not is_storage_session(branch):
            raise HTTPException(status_code=404, detail=f"Session not found: {branch}")

        # Load call tree from storage
        call_tree = load_call_tree_from_storage(branch)
        if not call_tree:
            raise HTTPException(status_code=404, detail=f"Call tree not found for session {branch}")

        # Create Git adapter for this session
        repo = get_repo_adapter(repo_path, execution_id=branch)

        # Enrich with content from storage
        enriched_call_tree = enrich_call_tree(call_tree, repo)

        return JSONResponse(content=enriched_call_tree)
    except Exception as e:
        print(f"Error fetching enriched call tree: {str(e)}")
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"detail": f"Internal Server Error: {str(e)}"})

@app.get("/get_file_content_disk/")
async def get_file_content_disk(path: str = Query(...)):
    try:
        if not os.path.isfile(path):
            return JSONResponse(status_code=404, content={"detail": "File not found."})
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        return Response(content=content, media_type="text/plain")
    except Exception as e:
        print(f"Error fetching file content from disk: {str(e)}")
        return JSONResponse(status_code=500, content={"detail": f"Internal Server Error: {str(e)}"})


@app.post("/save_settings/")
async def save_settings(request: Request):
    try:
        settings_data = await request.json()
        settings_path = get_current_settings_path()
        with open(settings_path, 'w') as f:
            toml.dump(settings_data, f)
        return JSONResponse(content={"detail": "Settings saved successfully"})
    except Exception as e:
        print(f"Error saving settings: {str(e)}")
        return JSONResponse(status_code=500, content={"detail": f"Internal Server Error: {str(e)}"})

# Endpoint to load settings
@app.get("/load_settings/")
async def load_settings():
    try:
        settings_path = get_current_settings_path()
        if not os.path.exists(settings_path):
            return JSONResponse(content={"settings": None})
        with open(settings_path, 'r') as f:
            settings_data = toml.load(f)
        return JSONResponse(content={"settings": settings_data})
    except Exception as e:
        print(f"Error loading settings: {str(e)}")
        return JSONResponse(status_code=500, content={"detail": f"Internal Server Error: {str(e)}"})

# Updated run_command endpoint to stream output with UTF-8 support
@app.post("/ws/run_command")
async def run_command(request: Request):
    data = await request.json()
    print("Data: ", data)
    command = data.get("command")
    path = data.get("path")

    if not command or not path:
        raise HTTPException(status_code=400, detail="Command and path are required")

    try:
        async def stream_command():
            # Set up UTF-8 environment for the subprocess
            env = os.environ.copy()
            env.update({
                'PYTHONIOENCODING': 'utf-8',
                'LC_ALL': 'en_US.UTF-8',
                'LANG': 'en_US.UTF-8'
            })
            
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=path,
                env=env
            )
            
            # Stream stdout with UTF-8-safe chunking to preserve Rich ANSI and encoding
            buffer = b''
            while True:
                chunk = await read_utf8_safe_chunk(process.stdout, 1024)
                if not chunk:
                    break
                
                # Add to buffer and yield complete chunk
                buffer += chunk
                try:
                    decoded_chunk = buffer.decode('utf-8', errors='strict')
                    yield decoded_chunk
                    buffer = b''  # Clear buffer after successful decode
                except UnicodeDecodeError:
                    # If we still can't decode, yield with error replacement
                    decoded_chunk = buffer.decode('utf-8', errors='replace')
                    yield decoded_chunk
                    buffer = b''

            # Yield any remaining buffer content
            if buffer:
                yield buffer.decode('utf-8', errors='replace')

            # Stream any stderr after stdout
            stderr_buffer = b''
            while True:
                chunk = await read_utf8_safe_chunk(process.stderr, 1024)
                if not chunk:
                    break
                
                stderr_buffer += chunk
                try:
                    decoded_chunk = stderr_buffer.decode('utf-8', errors='strict')
                    yield decoded_chunk
                    stderr_buffer = b''
                except UnicodeDecodeError:
                    decoded_chunk = stderr_buffer.decode('utf-8', errors='replace')
                    yield decoded_chunk
                    stderr_buffer = b''

            # Yield any remaining stderr buffer
            if stderr_buffer:
                yield stderr_buffer.decode('utf-8', errors='replace')

            # Wait for process to complete
            await process.wait()

        return StreamingResponse(stream_command(), media_type="text/plain")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to execute command: {str(e)}")

@app.post("/create_file/")
async def create_file_endpoint(path: str = Query(...), name: str = Query(...)):
    try:
        full_path = os.path.join(path, name)
        with open(full_path, 'w') as f:
            pass  # Creates an empty file
        return JSONResponse(status_code=200, content={"detail": "File created successfully"})
    except Exception as e:
        print(f"Error creating file: {str(e)}")
        return JSONResponse(status_code=500, content={"detail": f"Internal Server Error: {str(e)}"})

@app.post("/create_folder/")
async def create_folder_endpoint(path: str = Query(...), name: str = Query(...)):
    try:
        full_path = os.path.join(path, name)
        os.makedirs(full_path, exist_ok=True)
        return JSONResponse(status_code=200, content={"detail": "Folder created successfully"})
    except Exception as e:
        print(f"Error creating folder: {str(e)}")
        return JSONResponse(status_code=500, content={"detail": f"Internal Server Error: {str(e)}"})

@app.post("/save_file")
async def save_file(request: Request):
    try:
        # Get JSON payload
        data = await request.json()
        
        # Validate required fields
        if not all(key in data for key in ['path', 'content']):
            raise HTTPException(status_code=400, detail="Missing required fields")
            
        file_path = data['path']
        content = data['content']
        
        # Ensure path is within BASE_DIR
        full_path = (Path(BASE_DIR) / file_path).resolve()
        if not str(full_path).startswith(str(BASE_DIR)):
            raise HTTPException(status_code=403, detail="Access denied: Path outside base directory")
            
        # Create parent directories if they don't exist
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write content to file
        full_path.write_text(content)
        
        return JSONResponse(
            content={"message": "File saved successfully"},
            status_code=200
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def read_utf8_safe_chunk(stream, chunk_size=1024):
    """Read a chunk that doesn't split UTF-8 characters"""
    chunk = await stream.read(chunk_size)
    if not chunk:
        return chunk
    
    # If chunk ends with incomplete UTF-8 sequence, read more bytes
    while True:
        try:
            chunk.decode('utf-8')
            break  # Valid UTF-8, safe to return
        except UnicodeDecodeError as e:
            if e.start < len(chunk) - 4:  # Error not at the end, return what we have
                break
            # Read one more byte and try again
            next_byte = await stream.read(1)
            if not next_byte:
                break  # End of stream
            chunk += next_byte
            if len(chunk) > chunk_size + 16:  # Prevent infinite loop
                break
    return chunk

@app.post("/ws/run_fractalic")
async def run_fractalic(request: Request):
    data = await request.json()
    file_path = data.get("file_path")
    if not file_path:
        raise HTTPException(status_code=400, detail="file_path is required")

    # Generate execution ID for this run
    execution_id = str(uuid.uuid4())

    # Build command using centralized path management
    fractalic_root = get_fractalic_root()
    fractalic_path = Path(fractalic_root) / "fractalic.py"

    # Using current Python (from venv)
    python_exe = sys.executable 
    command = f'"{python_exe}" "{fractalic_path}" "{file_path}"'

    async def stream_fractalic():
        process = None
        try:
            # Set up UTF-8 environment for the subprocess
            env = os.environ.copy()
            env.update({
                'PYTHONIOENCODING': 'utf-8',
                'LC_ALL': 'en_US.UTF-8',
                'LANG': 'en_US.UTF-8',
                'FRACTALIC_EXECUTION_ID': execution_id
            })
            
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=fractalic_root,  # Use fractalic_root instead of root_dir
                env=env
            )

            # Function to check if process is still alive
            def is_process_alive():
                return process and process.returncode is None

            # Stream stdout with UTF-8-safe chunking to preserve Rich ANSI and encoding
            buffer = b''
            while is_process_alive():
                try:
                    # Use a timeout to avoid blocking forever
                    chunk = await asyncio.wait_for(
                        read_utf8_safe_chunk(process.stdout, 1024), 
                        timeout=1.0
                    )
                    if not chunk:
                        break
                    
                    # Add to buffer and yield complete chunk
                    buffer += chunk
                    try:
                        decoded_chunk = buffer.decode('utf-8', errors='strict')
                        
                        # НЕ фильтруем [Event] строки - они нужны фронтенду
                        # События теперь дублируются: идут и через HTTP для chat, и через terminal для совместимости
                        yield decoded_chunk
                        buffer = b''  # Clear buffer after successful decode
                    except UnicodeDecodeError:
                        # If we still can't decode, yield with error replacement
                        # This handles edge cases where the chunk boundary still splits characters
                        decoded_chunk = buffer.decode('utf-8', errors='replace')
                        yield decoded_chunk
                        buffer = b''
                        
                except asyncio.TimeoutError:
                    # Check if process is still running, if not break
                    if not is_process_alive():
                        break
                    continue
                except Exception as e:
                    # Process might have terminated unexpectedly
                    if is_process_alive():
                        yield f"\n[Error reading stdout: {str(e)}]\n"
                    break

            # Yield any remaining buffer content
            if buffer:
                yield buffer.decode('utf-8', errors='replace')

            # Stream any stderr after stdout
            stderr_buffer = b''
            while is_process_alive():
                try:
                    # Use a timeout for stderr as well
                    chunk = await asyncio.wait_for(
                        read_utf8_safe_chunk(process.stderr, 1024), 
                        timeout=1.0
                    )
                    if not chunk:
                        break
                    
                    stderr_buffer += chunk
                    try:
                        decoded_chunk = stderr_buffer.decode('utf-8', errors='strict')
                        yield decoded_chunk
                        stderr_buffer = b''
                    except UnicodeDecodeError:
                        decoded_chunk = stderr_buffer.decode('utf-8', errors='replace')
                        yield decoded_chunk
                        stderr_buffer = b''
                        
                except asyncio.TimeoutError:
                    if not is_process_alive():
                        break
                    continue
                except Exception as e:
                    if is_process_alive():
                        yield f"\n[Error reading stderr: {str(e)}]\n"
                    break

            # Yield any remaining stderr buffer
            if stderr_buffer:
                yield stderr_buffer.decode('utf-8', errors='replace')

            # Wait for process to complete and get exit code
            if process:
                exit_code = await process.wait()
                if exit_code != 0:
                    yield f"\n[Process exited with code {exit_code}]\n"
                else:
                    yield f"\n[Process completed successfully]\n"

        except Exception as e:
            yield f"\n[Fatal error in fractalic execution: {str(e)}]\n"
        finally:
            # Cleanup: terminate process if it's still running
            if process and process.returncode is None:
                try:
                    process.terminate()
                    await asyncio.wait_for(process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    # Force kill if terminate doesn't work
                    try:
                        process.kill()
                        await process.wait()
                    except:
                        pass
                except:
                    pass

    return StreamingResponse(
        stream_fractalic(), 
        media_type="text/plain",
        headers={"X-Execution-Id": execution_id}
    )

# Cache for tools schema to avoid recreating ToolRegistry repeatedly
_tools_schema_cache = {}
_SCHEMA_CACHE_DURATION = 30  # Cache for 30 seconds

@app.get("/tools_schema/")
async def tools_schema(tools_dir: str = Query("tools", description="Path to the tools directory")):
    """
    Autodiscover tools from the specified tools_dir and return their schema in OpenAI/MCP-compatible JSON format.
    Preserves original frontend compatibility while using improved ToolRegistry.
    """
    try:
        current_time = time.time()
        
        # Check if we have a cached schema for this tools_dir
        cache_key = tools_dir
        if cache_key in _tools_schema_cache:
            cached_schema, timestamp = _tools_schema_cache[cache_key]
            if current_time - timestamp < _SCHEMA_CACHE_DURATION:
                return JSONResponse(content=cached_schema)
        
        # Use the original simple logic - frontend knows where tools are
        # ToolRegistry will handle path resolution internally
        registry = ToolRegistry(tools_dir=tools_dir)
        schema = registry.generate_schema()
        _tools_schema_cache[cache_key] = (schema, current_time)
        return JSONResponse(content=schema)
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": f"Internal Server Error: {str(e)}"})

@app.delete("/delete_item/")
async def delete_item(path: str = Query(...)):
    """
    Delete a file or directory from the filesystem.
    """
    try:
        # Resolve the path
        item_path = Path(path).resolve()
        
        # Ensure the path exists
        if not item_path.exists():
            raise HTTPException(status_code=404, detail="File or directory not found")
        
        # Check if it's a file or directory and delete accordingly
        if item_path.is_file():
            item_path.unlink()
            return JSONResponse(status_code=200, content={"detail": f"File '{item_path.name}' deleted successfully"})
        elif item_path.is_dir():
            import shutil
            shutil.rmtree(item_path)
            return JSONResponse(status_code=200, content={"detail": f"Directory '{item_path.name}' deleted successfully"})
        else:
            raise HTTPException(status_code=400, detail="Invalid file or directory")
            
    except PermissionError:
        raise HTTPException(status_code=403, detail="Permission denied: Cannot delete file or directory")
    except Exception as e:
        print(f"Error deleting item: {str(e)}")
        return JSONResponse(status_code=500, content={"detail": f"Internal Server Error: {str(e)}"})

@app.post("/rename_item/")
async def rename_item(old_path: str = Query(...), new_name: str = Query(...)):
    """
    Rename a file or directory.
    """
    try:
        # Resolve the old path
        old_item_path = Path(old_path).resolve()
        
        # Ensure the old path exists
        if not old_item_path.exists():
            raise HTTPException(status_code=404, detail="File or directory not found")
        
        # Validate new name (basic validation)
        if not new_name or new_name.strip() == "":
            raise HTTPException(status_code=400, detail="New name cannot be empty")
        
        # Check for invalid characters in the new name
        invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
        if any(char in new_name for char in invalid_chars):
            raise HTTPException(status_code=400, detail="New name contains invalid characters")
        
        # Create the new path (same parent directory, new name)
        new_item_path = old_item_path.parent / new_name
        
        # Check if target already exists
        if new_item_path.exists():
            raise HTTPException(status_code=409, detail="A file or directory with that name already exists")
        
        # Rename the item
        old_item_path.rename(new_item_path)
        
        item_type = "directory" if new_item_path.is_dir() else "file"
        return JSONResponse(status_code=200, content={
            "detail": f"{item_type.capitalize()} renamed successfully",
            "old_name": old_item_path.name,
            "new_name": new_name,
            "new_path": str(new_item_path)
        })
        
    except PermissionError:
        raise HTTPException(status_code=403, detail="Permission denied: Cannot rename file or directory")
    except Exception as e:
        print(f"Error renaming item: {str(e)}")
        return JSONResponse(status_code=500, content={"detail": f"Internal Server Error: {str(e)}"})

# ============================================================================
# MCP Manager Control Routes
# ============================================================================

@app.post("/mcp/start")
async def start_mcp_manager_route():
    """Start the MCP manager process"""
    return await start_mcp_manager()

@app.post("/mcp/stop")
async def stop_mcp_manager_route():
    """Stop the MCP manager process gracefully.
    
    First attempts graceful shutdown via API, then terminate, finally kill if needed.
    """
    try:
        _record_trace("api", message="/mcp/stop called")
        result = await stop_mcp_manager()
        return result
    except Exception as e:
        _record_trace("error", message="/mcp/stop failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to stop MCP manager: {str(e)}")

@app.get("/mcp/status")
async def get_mcp_manager_status_route():
    """Get the status of the MCP manager and its servers"""
    return await get_mcp_manager_status()

# ============================================================================
# Health Check and Info Routes
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    mcp_status = await get_mcp_manager_status()
    return {
        "status": "healthy",
        "ui_server": "running",
        "mcp_manager": mcp_status
    }

@app.get("/info")
async def get_info():
    """Get information about the application"""
    mcp_status = await get_mcp_manager_status()
    return {
        "application": "Fractalic UI Server",
        "version": "1.0.0",
        "features": {
            "git_operations": True,
            "file_management": True,
            "mcp_manager_control": True,
            "mcp_server_proxy": True
        },
        "mcp_manager": mcp_status
    }

# ============================================================================
# Docker Registry Deployment API
# ============================================================================

# Store active deployments (in production, use a database)
active_deployments: Dict[str, Dict[str, Any]] = {}

# Store for deployment progress streams
deployment_streams: Dict[str, List[Dict[str, Any]]] = {}

def validate_docker_registry_request(data: Dict[str, Any]) -> Dict[str, str]:
    """Validate Docker registry deployment request"""
    errors = []
    
    # Check required fields based on plugin expectations
    if not data.get("script_name", "").strip():
        errors.append("script_name is required")
    
    if not data.get("script_folder", "").strip():
        errors.append("script_folder is required") 
    
    # Note: image_name is hardcoded in backend, no validation needed
    
    return errors

@app.post("/api/deploy/docker-registry")
async def deploy_docker_registry(request: Request):
    """Deploy using Docker registry (non-streaming version)"""
    import uuid
    from datetime import datetime
    
    try:
        data = await request.json()
        
        # Validate input
        validation_errors = validate_docker_registry_request(data)
        if validation_errors:
            raise HTTPException(
                status_code=400, 
                detail=f"Validation failed: {', '.join(validation_errors)}"
            )
        
        deployment_id = str(uuid.uuid4())
        
        # Import the plugin manager
        from publisher.plugin_manager import PluginManager
        from publisher.models import PublishRequest
        
        # Initialize plugin manager and get Docker registry plugin
        plugin_manager = PluginManager()
        docker_plugin = plugin_manager.get_plugin("docker-registry")
        
        if not docker_plugin:
            raise HTTPException(status_code=500, detail="Docker registry plugin not available")
        
        # Create deployment config from request data
        from publisher.models import DeploymentConfig
        
        # Convert frontend request to DeploymentConfig
        config = DeploymentConfig(
            plugin_name="docker-registry",
            script_name=data.get("script_name", ""),
            script_folder=data.get("script_folder", ""),
            container_name=data.get("container_name", f"fractalic-{deployment_id[:8]}"),
            environment_vars=data.get("env_vars", {}),
            port_mapping={},
            custom_domain=None,
            plugin_specific={
                "image_name": "ghcr.io/fractalic-ai/fractalic:latest-production",  # Always use production image
                "config_files": data.get("config_files", []),
                "mount_paths": data.get("mount_paths", {})
            }
        )
        
        # Define progress callback for real-time updates
        def progress_callback(message: str, progress: int):
            print(f"[PROGRESS] {progress}% - {message}")
        
        # Run deployment (blocking) with correct interface
        response = docker_plugin.publish(
            source_path=data.get("script_folder", ""),
            config=config,
            progress_callback=progress_callback
        )
        
        # Store deployment info if successful
        if response.success and response.deployment_id:
            active_deployments[response.deployment_id] = {
                "deployment_id": response.deployment_id,
                "script_name": data.get("script_name", "unknown"),
                "status": "running",
                "created_at": time.time(),
                "metadata": response.metadata or {}
            }
        
        return {
            "success": response.success,
            "message": response.message,
            "deployment_id": response.deployment_id,
            "endpoint_url": response.url,  # Use 'url' field from PublishResult
            "metadata": response.metadata
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Deployment failed: {str(e)}")

@app.post("/api/deploy/docker-registry/stream")
async def deploy_docker_registry_with_progress(request: Request):
    """Deploy using Docker registry with real-time progress streaming via SSE"""
    import uuid
    import asyncio
    import json
    from datetime import datetime
    
    try:
        data = await request.json()
        
        # Validate input upfront - return HTTP error instead of SSE error for validation failures
        validation_errors = validate_docker_registry_request(data)
        if validation_errors:
            raise HTTPException(
                status_code=400,
                detail=f"Validation failed: {', '.join(validation_errors)}"
            )
        
        deployment_id = str(uuid.uuid4())
        
        # Initialize progress tracking
        deployment_streams[deployment_id] = []
        
        def progress_callback(message: str, progress: int):
            """Callback to track deployment progress"""
            progress_data = {
                "deployment_id": deployment_id,
                "timestamp": datetime.now().isoformat(),
                "message": message,
                "stage": "deploying",  # Default stage for now
                "progress": progress
            }
            deployment_streams[deployment_id].append(progress_data)
        
        async def stream_deployment():
            """Stream deployment progress to client"""
            try:
                # Import the plugin manager
                from publisher.plugin_manager import PluginManager
                from publisher.models import PublishRequest, DeploymentConfig
                
                # Initialize plugin manager and get Docker registry plugin
                plugin_manager = PluginManager()
                docker_plugin = plugin_manager.get_plugin("docker-registry")
                
                if not docker_plugin:
                    yield f"data: {json.dumps({'error': 'Docker registry plugin not available'})}\n\n"
                    return
                # Create deployment config (mirror non-streaming endpoint)
                config = DeploymentConfig(
                    plugin_name="docker-registry",
                    script_name=data.get("script_name", ""),
                    script_folder=data.get("script_folder", ""),
                    container_name=data.get("container_name", f"fractalic-{deployment_id[:8]}"),
                    environment_vars=data.get("env_vars", {}),
                    port_mapping={},
                    custom_domain=None,
                    plugin_specific={
                        "image_name": "ghcr.io/fractalic-ai/fractalic:latest-production",
                        "config_files": data.get("config_files", []),
                        "mount_paths": data.get("mount_paths", {})
                    }
                )
                # Start deployment in background
                loop = asyncio.get_event_loop()
                
                def run_deployment():
                    return docker_plugin.publish(
                        source_path=data.get("script_folder", ""),
                        config=config,
                        progress_callback=progress_callback
                    )
                
                # Run deployment in thread pool to avoid blocking
                deployment_future = loop.run_in_executor(None, run_deployment)
                
                # Stream progress updates
                last_sent_count = 0
                while not deployment_future.done():
                    # Send new progress updates
                    current_progress = deployment_streams.get(deployment_id, [])
                    for progress_data in current_progress[last_sent_count:]:
                        yield f"data: {json.dumps(progress_data)}\n\n"
                    last_sent_count = len(current_progress)
                    
                    await asyncio.sleep(0.5)  # Check for updates every 500ms
                
                # Get final result
                response = await deployment_future
                
                # Send any remaining progress updates
                current_progress = deployment_streams.get(deployment_id, [])
                for progress_data in current_progress[last_sent_count:]:
                    yield f"data: {json.dumps(progress_data)}\n\n"
                
                # Send final result
                final_result = {
                    "deployment_id": deployment_id,
                    "timestamp": datetime.now().isoformat(),
                    "message": "Deployment completed",
                    "stage": "completed",
                    "progress": 100,
                    "result": {
                        "success": response.success,
                        "message": response.message,
                        "deployment_id": response.deployment_id,
                        "endpoint_url": response.url,  # Use 'url' field from PublishResult
                        "metadata": response.metadata
                    }
                }
                
                yield f"data: {json.dumps(final_result)}\n\n"
                
                # Store deployment info if successful
                if response.success and response.deployment_id:
                    active_deployments[response.deployment_id] = {
                        "deployment_id": response.deployment_id,
                        "script_name": data.get("script_name", "unknown"),
                        "status": "running",
                        "created_at": time.time(),
                        "metadata": response.metadata or {}
                    }
                
            except Exception as e:
                error_data = {
                    "deployment_id": deployment_id,
                    "timestamp": datetime.now().isoformat(),
                    "message": f"Deployment failed: {str(e)}",
                    "stage": "error",
                    "progress": 100,
                    "error": str(e)
                }
                yield f"data: {json.dumps(error_data)}\n\n"
            finally:
                # Cleanup progress tracking
                if deployment_id in deployment_streams:
                    del deployment_streams[deployment_id]
        
        return StreamingResponse(
            stream_deployment(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"  # Disable nginx buffering
            }
        )
        
    except HTTPException:
        raise  # Re-raise HTTPExceptions to preserve status codes
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start deployment: {str(e)}")

# Event streaming system (HTTP-based instead of EventBus)
from collections import deque, defaultdict
import threading
import json
from datetime import datetime

# Single FIFO queue for all events (maintains chronological order)
# Previously used separate queues per execution_id which broke timestamp ordering
events_queue = deque(maxlen=10000)
events_lock = threading.Lock()

# Global sequence counter for event ordering verification
events_seq_counter = 0

# Store running processes per execution_id
running_processes = {}
processes_lock = threading.Lock()

# Track parent relationships between execution IDs
execution_parent_map = {}

# Track per-execution event sequence numbers
event_sequence_counters = defaultdict(int)

# Terminal stream ownership tracking
terminal_owner_stacks = defaultdict(list)
terminal_state_lock = threading.Lock()

# Track which execution_ids have active capture tasks (prevent duplicates)
active_capture_tasks = set()
active_capture_tasks_lock = threading.Lock()

# Track background monitor tasks that wait for subprocess completion.
process_monitor_tasks = {}
process_monitor_tasks_lock = threading.Lock()


def resolve_root_execution_id(execution_id: str) -> str:
    if not execution_id:
        return execution_id
    current = execution_id
    visited = set()
    while True:
        parent = execution_parent_map.get(current)
        if not parent or parent in visited:
            return current
        visited.add(current)
        current = parent


def ensure_root_stack(root_execution_id: str):
    stack = terminal_owner_stacks[root_execution_id]
    if not stack or stack[0] != root_execution_id:
        stack.clear()
        stack.append(root_execution_id)
    return stack

@app.post("/api/events/receive")
async def receive_event(event: dict):
    """Receive event from fractalic process."""
    execution_id = event.get('execution_id')
    if not execution_id:
        raise HTTPException(status_code=400, detail='Missing execution_id')

    print(f"[DEBUG /api/events/receive] Received event: {event.get('type')} for {execution_id}")

    # Log nested_execution_id if present
    if 'nested_execution_id' in event:
        print(f"[DEBUG /api/events/receive]   nested_execution_id: {event.get('nested_execution_id')}")
    if 'target' in event:
        print(f"[DEBUG /api/events/receive]   target: {event.get('target')}")
    if 'node_id' in event:
        print(f"[DEBUG /api/events/receive]   node_id: {event.get('node_id')}")

    if event.get('type') == 'chat_message':
        content_preview = event.get('content', '')[:100] if event.get('content') else 'no content'
        print(f"[DEBUG /api/events/receive] CHAT_MESSAGE content preview: {content_preview}")
    elif event.get('type') == 'execution':
        print(f"[DEBUG /api/events/receive] EXECUTION event - phase: {event.get('phase')}, full event: {event}")

    parent_execution_id = event.get('parent_execution_id')

    # Update parent mapping & terminal ownership stack
    with terminal_state_lock:
        if parent_execution_id:
            execution_parent_map[execution_id] = parent_execution_id
        else:
            execution_parent_map.setdefault(execution_id, None)

        root_execution_id = resolve_root_execution_id(execution_id)
        stack = ensure_root_stack(root_execution_id)

        event_type = event.get('type')
        if event_type == EventType.WORKFLOW_START.value or event_type == 'workflow_start':
            if execution_id not in stack:
                stack.append(execution_id)
        elif event_type in (EventType.WORKFLOW_COMPLETE.value, EventType.WORKFLOW_ERROR.value, 'workflow_complete', 'workflow_error'):
            if execution_id in stack:
                while len(stack) > 1 and stack[-1] != execution_id:
                    stack.pop()
                if len(stack) > 1 and stack[-1] == execution_id:
                    stack.pop()

        event['root_execution_id'] = root_execution_id

    # Add event to single global FIFO queue with global sequence counter
    # This maintains chronological order across all execution IDs
    with events_lock:
        global events_seq_counter
        events_seq_counter += 1
        event['seq'] = events_seq_counter

        # Also track per-execution sequence for compatibility
        event_sequence_counters[execution_id] += 1
        event['execution_seq'] = event_sequence_counters[execution_id]

        events_queue.append(event)

    return {"status": "received"}

async def start_fractalic_process(file_path: str, execution_id: str, user_request: str = ""):
    """Start fractalic process with execution_id and user request (already formatted by frontend with history)."""
    try:
        fractalic_root = get_fractalic_root()
        fractalic_path = Path(fractalic_root) / "fractalic.py"
        python_exe = sys.executable

        # Формируем команду
        command = f'"{python_exe}" "{fractalic_path}" "{file_path}"'

        # Frontend sends fully formatted markdown with history - just pass it through
        if user_request.strip():
            # Экранируем кавычки
            escaped_request = user_request.replace('"', '\\"')
            command += f' --param_input_user_request "UserRequest" --param_input_user_request_value "{escaped_request}"'
            print(f"[DEBUG] Starting fractalic with user_request (length: {len(user_request)} chars)")
            print(f"[DEBUG] User request preview: {user_request[:300]}...")
            print(f"[DEBUG] Command: {command}")
        else:
            print(f"[DEBUG] Starting fractalic without user_request")
            print(f"[DEBUG] Command: {command}")

        env = os.environ.copy()
        env.update({
            'PYTHONIOENCODING': 'utf-8',
            'LC_ALL': 'en_US.UTF-8',
            'LANG': 'en_US.UTF-8',
            'FRACTALIC_EXECUTION_ID': execution_id
        })

        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=fractalic_root,
            env=env
        )

        # Store process for terminal streaming
        with processes_lock:
            running_processes[execution_id] = process

        with terminal_state_lock:
            stack = terminal_owner_stacks[execution_id]
            stack.clear()
            stack.append(execution_id)
        event_sequence_counters[execution_id] = 0
        execution_parent_map.setdefault(execution_id, None)

        # Start background task to capture terminal output ALWAYS
        # Only start if not already capturing for this execution_id
        with active_capture_tasks_lock:
            if execution_id not in active_capture_tasks:
                active_capture_tasks.add(execution_id)
                asyncio.create_task(capture_terminal_output(process, execution_id))
                print(f"[DEBUG] Started terminal capture task for {execution_id}")
            else:
                print(f"[DEBUG] Skipped duplicate terminal capture task for {execution_id}")

        monitor_task = asyncio.create_task(monitor_process_completion(process, execution_id))
        with process_monitor_tasks_lock:
            process_monitor_tasks[execution_id] = monitor_task
        print(f"[DEBUG] Started process monitor task for {execution_id} (pid={getattr(process, 'pid', 'unknown')})")

        # Don't wait for completion - let events drive the response
        return process
    except Exception as e:
        print(f"[ERROR] Failed to start process: {str(e)}")
        return None

async def capture_terminal_output(process, execution_id: str):
    """Background task that captures terminal output and emits as events."""

    def emit_terminal_chunk(data: str, is_stderr: bool):
        """Emit terminal output as event to events_queue."""
        with terminal_state_lock:
            stack = ensure_root_stack(execution_id)
            owner_id = stack[-1] if stack else execution_id

        # Add event directly to events_queue
        with events_lock:
            global events_seq_counter
            events_seq_counter += 1

            data_len = len(data)

            # Warn about very large chunks
            if data_len > 10000:
                print(f"[WARNING] Large terminal chunk: {data_len} bytes - may cause delays")

            event = {
                'type': EventType.TERMINAL_OUTPUT.value,
                'execution_id': owner_id,  # Route to current owner in stack
                'root_execution_id': execution_id,  # Track root process
                'data': data,
                'is_stderr': is_stderr,
                'timestamp': time.time(),
                'seq': events_seq_counter
            }
            events_queue.append(event)
            print(f"[DEBUG] Emitted terminal_output event for {owner_id} (root: {execution_id}), data length: {data_len} bytes")

    try:
        while True:
            if process.returncode is not None:
                print(f"[DEBUG] Capture task detected process completion for {execution_id} (returncode={process.returncode})")
                # Process finished, read remaining data
                remaining_stdout = await process.stdout.read()
                remaining_stderr = await process.stderr.read()

                if remaining_stdout:
                    decoded = remaining_stdout.decode('utf-8', errors='replace')
                    emit_terminal_chunk(decoded, is_stderr=False)

                if remaining_stderr:
                    decoded = remaining_stderr.decode('utf-8', errors='replace')
                    emit_terminal_chunk(decoded, is_stderr=True)
                break

            # Read chunks periodically with larger buffer for faster capture
            try:
                chunk = await asyncio.wait_for(process.stdout.read(8192), timeout=0.1)
                if chunk:
                    decoded = chunk.decode('utf-8', errors='replace')
                    emit_terminal_chunk(decoded, is_stderr=False)
            except asyncio.TimeoutError:
                pass

            try:
                err_chunk = await asyncio.wait_for(process.stderr.read(8192), timeout=0.1)
                if err_chunk:
                    decoded = err_chunk.decode('utf-8', errors='replace')
                    emit_terminal_chunk(decoded, is_stderr=True)
            except asyncio.TimeoutError:
                pass

            await asyncio.sleep(0.05)  # Small delay to avoid busy loop

    except Exception as e:
        print(f"[ERROR] Terminal capture failed for {execution_id}: {str(e)}")
    finally:
        # Clean up
        with processes_lock:
            running_processes.pop(execution_id, None)
        with active_capture_tasks_lock:
            active_capture_tasks.discard(execution_id)
        print(f"[DEBUG] Terminal capture completed for {execution_id}")

async def monitor_process_completion(process, execution_id: str):
    """Background diagnostic task that waits for subprocess completion."""
    pid = getattr(process, 'pid', 'unknown')
    start_time = time.time()
    print(f"[DEBUG] Process monitor started for {execution_id} (pid={pid})")
    try:
        returncode = await process.wait()
        duration = time.time() - start_time
        print(f"[DEBUG] Process monitor detected completion for {execution_id} (pid={pid}, returncode={returncode}, runtime={duration:.2f}s)")
    except Exception as e:
        print(f"[ERROR] Process monitor failed for {execution_id}: {e}")
    finally:
        with process_monitor_tasks_lock:
            process_monitor_tasks.pop(execution_id, None)

@app.post('/api/chat/stream')
async def stream_chat_events(request: Request):
    """Streamable HTTP endpoint for structured chat events (from fractalic HTTP events)."""
    
    data = await request.json()
    file_path = data.get("file_path")
    user_message = data.get("message", "")
    # Frontend sends fully formatted markdown with history embedded
    if not file_path:
        raise HTTPException(status_code=400, detail="file_path is required")

    # Generate execution ID for correlation
    execution_id = str(uuid.uuid4())

    import asyncio, json

    async def event_stream():
        try:
            # Start fractalic process in background (history already in user_message)
            process = await start_fractalic_process(file_path, execution_id, user_message)
            if not process:
                yield f"{json.dumps({'type': 'error', 'message': 'Failed to start process'}, ensure_ascii=False)}\n"
                return
            
            # Stream events from single FIFO queue as they arrive (maintains chronological order)
            # Server is AGNOSTIC to event types and execution_id structure - just proxy everything
            completed = False
            timeout_count = 0
            max_timeouts = 120  # 60 seconds total (0.5s * 120)
            process_finished_handled = False  # Flag to handle process completion only once
            last_process_state_log = 0
            process_state_log_interval = 5.0

            while not completed and timeout_count < max_timeouts:
                try:
                    # Stream events from single global queue in chronological order
                    events_found = False
                    with events_lock:
                        # Stream all events from FIFO queue
                        while events_queue:
                            event = events_queue.popleft()
                            event_type = event.get('type')
                            exec_id = event.get('execution_id', 'N/A')

                            # Log data size for terminal_output events
                            if event_type == 'terminal_output' and 'data' in event:
                                data_len = len(event.get('data', ''))
                                print(f"[DEBUG /api/chat/stream] Streaming event #{event.get('seq')}: {event_type} for {exec_id} (data: {data_len} bytes)")
                            else:
                                print(f"[DEBUG /api/chat/stream] Streaming event #{event.get('seq')}: {event_type} for {exec_id}")

                            # Safe JSON serialization with error handling
                            try:
                                json_str = json.dumps(event, ensure_ascii=False)
                                yield f"{json_str}\n"
                                events_found = True
                            except Exception as json_err:
                                print(f"[ERROR] Failed to serialize event {event.get('seq')}: {json_err}")
                                # Send error event instead
                                error_event = {
                                    'type': 'error',
                                    'message': f'Failed to serialize event: {str(json_err)}',
                                    'execution_id': exec_id
                                }
                                yield f"{json.dumps(error_event, ensure_ascii=False)}\n"
                                events_found = True

                    if events_found:
                        timeout_count = 0  # Reset timeout when we get events
                    else:
                        # No events available, wait a bit and increment timeout
                        await asyncio.sleep(0.5)
                        timeout_count += 1

                    now = time.time()
                    if now - last_process_state_log >= process_state_log_interval:
                        rc = getattr(process, 'returncode', 'N/A') if process else 'N/A'
                        stdout_eof = process.stdout.at_eof() if process and getattr(process, 'stdout', None) else 'N/A'
                        stderr_eof = process.stderr.at_eof() if process and getattr(process, 'stderr', None) else 'N/A'
                        with events_lock:
                            queue_size_snapshot = len(events_queue)
                        with active_capture_tasks_lock:
                            capture_active = execution_id in active_capture_tasks
                        with process_monitor_tasks_lock:
                            monitor_task = process_monitor_tasks.get(execution_id)
                        monitor_status = None
                        if monitor_task:
                            monitor_status = "done" if monitor_task.done() else "pending"
                        else:
                            monitor_status = "missing"
                        print(
                            f"[TRACE /api/chat/stream] Process state exec={execution_id}: "
                            f"returncode={rc}, stdout_eof={stdout_eof}, stderr_eof={stderr_eof}, "
                            f"queue_size={queue_size_snapshot}, capture_active={capture_active}, "
                            f"monitor={monitor_status}, timeout_count={timeout_count}"
                        )
                        last_process_state_log = now

                    # Check if process is still running - ALWAYS check, not just when no events!
                    # This must be outside the else block to catch process completion even when events are flowing
                    # Use flag to handle completion only once
                    if not process_finished_handled and process and hasattr(process, 'returncode') and process.returncode is not None:
                            process_finished_handled = True  # Set flag immediately to prevent re-entry
                            # Process finished, wait for terminal capture to complete AND events queue to drain
                            print(f"[DEBUG /api/chat/stream] Process finished (returncode={process.returncode}), waiting for terminal capture completion")

                            # Wait for capture task to finish (up to 10 seconds)
                            wait_time = 0
                            capture_finished = False
                            while wait_time < 10:
                                with active_capture_tasks_lock:
                                    if execution_id not in active_capture_tasks:
                                        capture_finished = True
                                        print(f"[DEBUG /api/chat/stream] Terminal capture task completed")
                                        break
                                await asyncio.sleep(0.5)
                                wait_time += 0.5

                            if not capture_finished:
                                print(f"[WARNING /api/chat/stream] Terminal capture task timeout after {wait_time}s")

                            # Now wait for events queue to drain (up to 5 seconds)
                            print(f"[DEBUG /api/chat/stream] Waiting for events queue to drain...")
                            drain_wait = 0
                            last_queue_size = -1
                            stuck_iterations = 0
                            while drain_wait < 5:
                                with events_lock:
                                    queue_size = len(events_queue)
                                    print(f"[DEBUG /api/chat/stream] Events queue size: {queue_size}")
                                    if queue_size == 0:
                                        print(f"[DEBUG /api/chat/stream] Events queue drained")
                                        break
                                    # If queue is not shrinking, detect no reader scenario
                                    if queue_size == last_queue_size:
                                        stuck_iterations += 1
                                        print(f"[DEBUG /api/chat/stream] Events queue stuck at {queue_size} events (iteration {stuck_iterations})")
                                        # If stuck for 3 iterations (1.5s), assume no reader and exit early
                                        if stuck_iterations >= 3:
                                            print(f"[INFO /api/chat/stream] No active consumer detected, exiting drain wait")
                                            break
                                    else:
                                        stuck_iterations = 0  # Reset if queue is changing
                                    last_queue_size = queue_size
                                await asyncio.sleep(0.5)
                                drain_wait += 0.5

                            # Extra delay for HTTP events to arrive
                            await asyncio.sleep(0.5)

                            # Drain all remaining events from single queue
                            final_event_count = 0
                            with events_lock:
                                # Stream ALL remaining events from FIFO queue
                                while events_queue:
                                    event = events_queue.popleft()
                                    event_type = event.get('type')

                                    # Log data size for terminal_output events
                                    if event_type == 'terminal_output' and 'data' in event:
                                        data_len = len(event.get('data', ''))
                                        print(f"[DEBUG /api/chat/stream] Streaming FINAL event #{event.get('seq')}: {event_type} for {event.get('execution_id')} (data: {data_len} bytes)")
                                    else:
                                        print(f"[DEBUG /api/chat/stream] Streaming FINAL event #{event.get('seq')}: {event_type} for {event.get('execution_id')}")

                                    # Safe JSON serialization with error handling
                                    try:
                                        json_str = json.dumps(event, ensure_ascii=False)
                                        yield f"{json_str}\n"
                                        final_event_count += 1
                                    except Exception as json_err:
                                        print(f"[ERROR] Failed to serialize FINAL event {event.get('seq')}: {json_err}")
                                        # Send error event instead
                                        error_event = {
                                            'type': 'error',
                                            'message': f'Failed to serialize final event: {str(json_err)}',
                                            'execution_id': event.get('execution_id', 'N/A')
                                        }
                                        yield f"{json.dumps(error_event, ensure_ascii=False)}\n"
                                        final_event_count += 1

                            print(f"[DEBUG /api/chat/stream] Streamed {final_event_count} final events")
                            completed = True
                            break
                        
                except Exception as e:
                    yield f"{json.dumps({'type': 'error', 'message': f'Stream error: {str(e)}'}, ensure_ascii=False)}\n"
                    break
            
            # Final completion message
            if not completed:
                rc = getattr(process, 'returncode', 'N/A') if process else 'N/A'
                print(f"[WARNING /api/chat/stream] Stream exiting without completion for {execution_id} (timeout_count={timeout_count}, returncode={rc})")
                yield f"{json.dumps({'type': 'error', 'message': 'Process timeout - no completion event received'}, ensure_ascii=False)}\n"
                
        except Exception as e:
            yield f"{json.dumps({'type': 'error', 'message': f'Fatal error: {str(e)}'}, ensure_ascii=False)}\n"

    return StreamingResponse(
        event_stream(), 
        media_type='text/plain',
        headers={
            'Cache-Control': 'no-cache', 
            'X-Execution-Id': execution_id
        }
    )

# ============================================================================
# Storage Mode API - Session Management
# ============================================================================

@app.get("/api/storage/sessions")
async def list_storage_sessions():
    """
    List all storage sessions.

    Returns session information similar to /branches_and_commits/
    but for storage mode.
    """
    try:
        from core.storage import get_sessions_dir
        sessions_dir = get_sessions_dir()

        if not sessions_dir.exists():
            return []

        sessions_data = []

        for session_dir in sorted(sessions_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if not session_dir.is_dir():
                continue

            execution_id = session_dir.name

            # Read session metadata
            metadata_file = session_dir / 'metadata' / 'session.json'
            session_metadata = {}
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    session_metadata = json.load(f)

            # Find call_tree in artifacts
            artifacts_dir = session_dir / 'artifacts' / 'nodes'
            call_tree = None

            if artifacts_dir.exists():
                for node_dir in artifacts_dir.iterdir():
                    if not node_dir.is_dir():
                        continue

                    call_tree_file = node_dir / 'call_tree.dat'
                    if call_tree_file.exists():
                        with open(call_tree_file, 'r') as f:
                            call_tree = json.load(f)
                        break

            if not call_tree:
                # No call tree found, skip this session
                continue

            # Build session node similar to branch node structure
            session_node = {
                'id': execution_id,
                'text': f"Session {execution_id[:8]}... ({session_metadata.get('created_at', 'unknown')})",
                'state': {'opened': True},
                'children': [],
                'execution_id': execution_id,
                'type': 'storage_session'
            }

            # Build tree from call_tree (same structure as Git mode)
            def build_tree(node):
                node_id = f"{node.get('ctx_file', 'unknown')}_{node.get('ctx_commit_hash', 'unknown')}"
                tree_node = {
                    'id': node_id,
                    'text': node.get('ctx_file', 'unknown'),
                    'ctx_file': node.get('ctx_file'),
                    'filename': node.get('filename'),
                    'md_file': node.get('filename'),
                    'md_commit_hash': node.get('md_commit_hash'),
                    'ctx_commit_hash': node.get('ctx_commit_hash'),
                    'trc_file': node.get('trc_file', ''),
                    'trc_commit_hash': node.get('trc_commit_hash', ''),
                    'execution_id': execution_id,  # Add execution_id for storage mode
                    'children': []
                }
                for child in node.get('children', []):
                    child_node = build_tree(child)
                    tree_node['children'].append(child_node)
                return tree_node

            root_node = build_tree(call_tree)
            session_node['children'].append(root_node)
            sessions_data.append(session_node)

        return sessions_data

    except Exception as e:
        print(f"Error listing storage sessions: {str(e)}")
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"detail": f"Internal Server Error: {str(e)}"})


@app.get("/api/storage/sessions/{execution_id}")
async def get_storage_session(execution_id: str):
    """Get detailed information about a specific storage session."""
    try:
        from core.storage import get_sessions_dir
        sessions_dir = get_sessions_dir()
        session_dir = sessions_dir / execution_id

        if not session_dir.exists():
            raise HTTPException(status_code=404, detail=f"Session {execution_id} not found")

        # Read metadata
        metadata_file = session_dir / 'metadata' / 'session.json'
        metadata = {}
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

        # List all nodes
        artifacts_dir = session_dir / 'artifacts' / 'nodes'
        nodes = []

        if artifacts_dir.exists():
            for node_dir in artifacts_dir.iterdir():
                if not node_dir.is_dir():
                    continue

                node_info = {
                    'node_id': node_dir.name,
                    'artifacts': [f.name for f in node_dir.iterdir() if f.is_file()]
                }
                nodes.append(node_info)

        return {
            'execution_id': execution_id,
            'metadata': metadata,
            'nodes': nodes,
            'workspace_dir': str(session_dir / 'workspace'),
            'artifacts_dir': str(artifacts_dir)
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting storage session: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


# Helper to emit execution lifecycle - now just logs
async def emit_execution_start(file_path: str, execution_id: str):
    print(f"[INFO] Execution started: {execution_id} for {file_path}")
    # Events will come via HTTP from fractalic.py, not here

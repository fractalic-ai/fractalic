# server.py
import asyncio
import sys
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import PlainTextResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import git
from pathlib import Path
import json
import os
import toml
from fastapi.responses import FileResponse

app = FastAPI()

current_repo_path = ""

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

# Define the settings file path
SETTINGS_FILE_PATH = '../../settings.toml'

def set_repo_path(path: str):
    """Set the current repository path globally"""
    global current_repo_path
    current_repo_path = path

# Helper functions
def get_repo(repo_path: str):
    resolved_path = Path(repo_path).resolve()
    if not resolved_path.is_dir():
        raise HTTPException(status_code=400, detail="Invalid repository path")
    try:
        repo = git.Repo(resolved_path)
        set_repo_path(resolved_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error initializing git repository: {e}")
    return repo

def get_file_content(repo, commit_hash, filepath):
    try:
        file_content = repo.git.show(f'{commit_hash}:{filepath}')
        return file_content
    except Exception as e:
        print(f"Error fetching '{filepath}' at commit '{commit_hash}': {e}")
        return ""

def ensure_empty_lines_before_symbols(text):
    lines = text.split('\n')
    new_lines = []
    for i, line in enumerate(lines):
        if line.startswith('#') or line.startswith('@'):
            if i > 0 and lines[i - 1].strip() != '':
                new_lines.append('')
        new_lines.append(line)
    return '\n'.join(new_lines)

def enrich_call_tree(node, repo):
    operation_src_list = node.get('operation_src', [])
    if operation_src_list and isinstance(operation_src_list[0], str):
        operation_src = operation_src_list[0]
    else:
        operation_src = "No Operation Source"

    filename = node.get('filename')
    ctx_file = node.get('ctx_file')
    md_commit_hash = node.get('md_commit_hash')
    ctx_commit_hash = node.get('ctx_commit_hash')

    source_content = get_file_content(repo, md_commit_hash, filename)
    target_content = get_file_content(repo, ctx_commit_hash, ctx_file)

    node['source_content'] = ensure_empty_lines_before_symbols(source_content)
    node['target_content'] = ensure_empty_lines_before_symbols(target_content)
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

    return FileResponse(image_path)


@app.get("/list_directory/")
async def list_directory(path: str = Query("")):
    resolved_path = Path(path).resolve()
    if not resolved_path.is_dir():
        raise HTTPException(status_code=400, detail="Invalid directory path")

    items = []
    for item in resolved_path.iterdir():
        is_dir = item.is_dir()
        is_git_repo = False
        if is_dir and (item / '.git').is_dir():
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
    try:
        repo = get_repo(repo_path)
        branches_data = []
        # Filter out main branch, only get custom branches
        branches = [b for b in repo.branches if b.name != 'main']
        branches = sorted(branches, key=lambda b: b.commit.committed_datetime, reverse=True)

        for branch in branches:
            try:
                call_tree_file_content = repo.git.show(f'{branch.name}:call_tree.json')
                call_tree = json.loads(call_tree_file_content)
            except Exception as e:
                print(f"Error reading call_tree.json from branch {branch.name}: {str(e)}")
                continue

            branch_node = {
                'id': branch.name,
                'text': branch.name,
                'state': {'opened': True},
                'children': []
            }

            def build_tree(node):
                node_id = f"{node['ctx_file']}_{node['ctx_commit_hash']}"
                tree_node = {
                    'id': node_id,
                    'text': node['ctx_file'],
                    'ctx_file': node['ctx_file'],
                    'filename': node['filename'],
                    'md_file': node['filename'],
                    'md_commit_hash': node['md_commit_hash'],
                    'ctx_commit_hash': node['ctx_commit_hash'],
                    'branch': branch.name,
                    'children': []
                }
                for child in node.get('children', []):
                    child_node = build_tree(child)
                    tree_node['children'].append(child_node)
                return tree_node

            root_node = build_tree(call_tree)
            branch_node['children'].append(root_node)
            branches_data.append(branch_node)

        return branches_data
    except Exception as e:
        print(f"Error processing branches and commits: {str(e)}")
        return JSONResponse(status_code=500, content={"detail": f"Internal Server Error: {str(e)}"})

@app.get("/get_file_content/")
async def get_file_content_endpoint(repo_path: str = Query(...), file_path: str = Query(...), commit_hash: str = Query(...)):
    try:
        repo = get_repo(repo_path)
        try:
            file_content = repo.git.show(f'{commit_hash}:{file_path}')
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"File not found in commit {commit_hash}: {str(e)}")
        return PlainTextResponse(file_content)
    except Exception as e:
        print(f"Error fetching file content: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.get("/get_enriched_call_tree/")
async def get_enriched_call_tree(repo_path: str = Query(...), branch: str = Query(...)):
    try:
        repo = get_repo(repo_path)
        call_tree_file_content = repo.git.show(f'{branch}:call_tree.json')
        call_tree = json.loads(call_tree_file_content)

        enriched_call_tree = enrich_call_tree(call_tree, repo)

        return JSONResponse(content=enriched_call_tree)
    except Exception as e:
        print(f"Error fetching enriched call tree: {str(e)}")
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


# Endpoint to save settings
@app.post("/save_settings/")
async def save_settings(request: Request):
    try:
        settings_data = await request.json()
        with open(SETTINGS_FILE_PATH, 'w') as f:
            toml.dump(settings_data, f)
        return JSONResponse(content={"detail": "Settings saved successfully"})
    except Exception as e:
        print(f"Error saving settings: {str(e)}")
        return JSONResponse(status_code=500, content={"detail": f"Internal Server Error: {str(e)}"})

# Endpoint to load settings
@app.get("/load_settings/")
async def load_settings():
    try:
        if not os.path.exists(SETTINGS_FILE_PATH):
            return JSONResponse(content={"settings": None})
        with open(SETTINGS_FILE_PATH, 'r') as f:
            settings_data = toml.load(f)
        return JSONResponse(content={"settings": settings_data})
    except Exception as e:
        print(f"Error loading settings: {str(e)}")
        return JSONResponse(status_code=500, content={"detail": f"Internal Server Error: {str(e)}"})

# Updated run_command endpoint to stream output
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
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=path
            )
            # Stream stdout in real-time
            while True:
                line = await process.stdout.readline()
                if not line:
                    break
                yield line.decode()

            # Stream any stderr after stdout is complete
            async for line in process.stderr:
                yield line.decode()

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


# Updated run_fractalic endpoint to stream output
@app.post("/ws/run_fractalic")
async def run_fractalic(request: Request):
    data = await request.json()
    file_path = data.get("file_path")
    if not file_path:
        raise HTTPException(status_code=400, detail="file_path is required")

    # Build command
    server_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(server_dir)
    root_dir = os.path.dirname(parent_dir)
    fractalic_path = os.path.join(root_dir, "fractalic.py")

    # Using current Python (from venv)
    python_exe = sys.executable 
    command = f'"{python_exe}" "{fractalic_path}" "{file_path}"'

    async def stream_fractalic():
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=root_dir
        )

        # Stream stdout in small chunks instead of entire lines
        while True:
            chunk = await process.stdout.read(16)  # read 16 bytes at a time
            if not chunk:
                break
            yield chunk.decode(errors="replace")

        # Stream any stderr after stdout
        while True:
            chunk = await process.stderr.read(16)
            if not chunk:
                break
            yield chunk.decode(errors="replace")

        await process.wait()

    return StreamingResponse(stream_fractalic(), media_type="text/plain")

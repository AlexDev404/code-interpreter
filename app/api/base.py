from fastapi import APIRouter, UploadFile, File, HTTPException, Path, Form
from fastapi.params import Body
from fastapi.responses import StreamingResponse
from typing import Annotated, List, Optional
import hashlib
import base64
import sys
from loguru import logger
from io import BytesIO

from app.api.exceptions import BadLanguageException

from ..models.base import (
    ExecuteResponse as CodeExecutionResponse,
    FileMetadata,
    UploadResponse,
    FileObject,
    Error as ErrorResponse,  # Aliased for backward compatibility
    SuccessResponse,
    PathParams,
    FileRef,
    CodeExecutionRequest,
)
from ..services.docker_executor import docker_executor
from ..services.file_manager import FileManager
from ..shared.config import get_settings
from app.utils.generate_id import generate_id

settings = get_settings()
router = APIRouter(prefix=settings.API_PREFIX)

# Initialize services
file_manager = FileManager()

SUPPORTED_LANGUAGES = {"py", "r", "bash", "js", "ts"}  # Python, R, Bash, JavaScript (Node.js) and TypeScript
MAX_RETRIES = 3


def _derive_session_id(user_id: Optional[str], entity_id: Optional[str]) -> str:
    """Derive a deterministic session_id from user_id and/or entity_id.

    When the client does not provide a session_id, we generate a stable one
    based on available identifiers so that subsequent requests from the same
    user/entity reuse the same sandbox container instead of creating a new one
    every time.

    When neither identifier is provided, a fixed default key is used so that
    anonymous requests still reuse a single container. In multi-user
    deployments, callers should supply at least user_id for proper isolation.
    """
    key = f"{user_id or 'default'}:{entity_id or 'default'}"
    raw = hashlib.sha256(key.encode()).digest()
    # base64url encoding produces A-Za-z0-9_- characters; strip padding to
    # ensure all 21 characters match the session_id pattern.
    encoded = base64.urlsafe_b64encode(raw).decode().rstrip("=")
    # Take exactly 21 characters to match the session_id pattern ^[A-Za-z0-9_-]{21}$
    return encoded[:21]


@router.post(
    "/execute",
    response_model=CodeExecutionResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    description="Execute code in the specified language",
    summary="Execute code",
    response_description="Returns the execution results",
    tags=["execution"],
)
async def execute_code(
    request: Annotated[
        CodeExecutionRequest,
        Body(
            openapi_examples={
                "Hello World (Python)": {
                    "summary": "Hello World in Python",
                    "value": {"code": "print('Hello, world!')", "lang": "py"},
                },
                "Random Number (Python)": {
                    "summary": "Random Number in Python",
                    "value": {"code": "import random; print(random.randint(1, 100))", "lang": "py"},
                },
                "Sleep (Python)": {
                    "summary": "Sleep",
                    "value": {"code": "import time; time.sleep(10); print('Done sleeping')", "lang": "py"},
                },
                "Hello World (R)": {
                    "summary": "Hello World in R",
                    "value": {"code": "cat('Hello, world!')", "lang": "r"},
                },
                "Random Number (R)": {
                    "summary": "Random Number in R",
                    "value": {"code": "cat(sample(1:100, 1))", "lang": "r"},
                },
                "Hello World (Node.js)": {
                    "summary": "Hello World in JavaScript",
                    "value": {"code": "console.log('Hello, world!')", "lang": "js"},
                },
                "Hello World (TypeScript)": {
                    "summary": "Hello World in TypeScript",
                    "value": {"code": "const msg: string = 'Hello, world!'; console.log(msg)", "lang": "ts"},
                },
            }
        ),
    ],
):
    """Execute code in the specified language."""
    logger.info(f"Executing code: {request.model_dump_json()}")

    if request.lang not in SUPPORTED_LANGUAGES:
        raise BadLanguageException(  # noqa: F821
            message=f"Language '{request.lang}' is not supported. Please use Python ('py'), R ('r'), Bash ('bash'), JavaScript ('js') or TypeScript ('ts')."
        )

    try:
        # Execution sessions are distinct from storage sessions: each request
        # gets a fresh sandbox session unless one is explicitly continued.
        # Referenced input files are staged into it from their own storage
        # sessions, which may differ per file.
        # When session_id is not provided, derive a deterministic one from
        # user_id/entity_id so the same user/entity reuses the same container.
        session_id = request.session_id or _derive_session_id(request.user_id, request.entity_id)
        logger.info(f"Using session ID: {session_id}")

        # Stage referenced input files into the execution session directory
        files = []
        if request.files:
            files = await file_manager.stage_files(session_id, request.files)

        # Execute code in Docker container
        result = await docker_executor.execute(code=request.code, session_id=session_id, lang=request.lang, files=files)

        # Add a language-specific error message if the stdout is empty
        if not result.get("stdout"):
            if request.lang == "py":
                result["stdout"] = "Empty. Make sure to explicitly print the results in Python"
            elif request.lang == "r":
                result["stdout"] = "Empty. Make sure to use print() or cat() to display results in R"
            elif request.lang == "bash":
                result["stdout"] = "Empty. Make sure the command writes its results to stdout (e.g. echo, cat)"
            elif request.lang == "js":
                result["stdout"] = "Empty. Make sure to explicitly console.log() the results in JavaScript"
            elif request.lang == "ts":
                result["stdout"] = "Empty. Make sure to explicitly console.log() the results in TypeScript"
            else:
                result["stdout"] = "Empty. Make sure to explicitly output the results"

        # Convert output files to FileRef model. Generated files are stored in
        # the execution session directory, so that is their storage session.
        output_files = [
            FileRef(
                id=file["id"],
                name=file.get("relative_path", file["filename"]),
                storage_session_id=file["session_id"],
                path=file["filepath"],
            )
            for file in result.get("files", [])
        ]

        # Get language-specific version information
        version_info = ""
        if request.lang == "py":
            version_info = f"Python {sys.version.split()[0]}"
        elif request.lang == "r":
            version_info = "R (Jupyter R-notebook)"
        elif request.lang == "bash":
            version_info = "Bash (Jupyter scipy-notebook)"
        elif request.lang == "js":
            version_info = "JavaScript (Node.js 24)"
        elif request.lang == "ts":
            version_info = "TypeScript (Node.js 24, type stripping)"
        else:
            version_info = f"Unknown language: {request.lang}"

        response = CodeExecutionResponse(
            run=result,
            language=request.lang,
            version=version_info,
            session_id=session_id,
            files=output_files,
        )

        logger.info(
            "Code execution completed successfully",
            extra={
                "response": {
                    "run": {
                        "stdout": result.get("stdout", ""),
                        "stderr": result.get("stderr", ""),
                        "output": result.get("output"),
                        "status": result.get("status"),
                    },
                    "language": request.lang,
                    "version": version_info,
                    "session_id": session_id,
                    "files": [f.model_dump() for f in output_files],
                }
            },
        )

        return response
    except Exception as e:
        logger.error(f"Error executing code: {str(e)}", exc_info=True)
        raise HTTPException(status_code=503, detail=str(e))


@router.post(
    "/upload",
    response_model=UploadResponse,
    responses={413: {"model": ErrorResponse}},
    description="Upload files for code execution",
    summary="Upload files",
    response_description="Returns information about the uploaded files",
    tags=["files"],
)
async def upload_files(
    files: List[UploadFile] = File(...),
    entity_id: Optional[str] = Form(None),
):
    """Upload files for code execution."""
    try:
        session_id = generate_id()
        logger.info(f"Starting file upload for session: {session_id}")
        uploaded_files = []

        for upload_file in files:
            logger.info(f"Processing upload of file: {upload_file.filename}")

            # Handle both stream and regular file uploads
            if hasattr(upload_file, "file") and isinstance(upload_file.file, (bytes, BytesIO)):
                content = upload_file.file if isinstance(upload_file.file, bytes) else upload_file.file.read()
                file_size = len(content)
                logger.info(f"Got file content from bytes/BytesIO, size: {file_size}")
            else:
                content = await upload_file.read()
                file_size = len(content)
                logger.info(f"Got file content from async read, size: {file_size}")

            if file_size > settings.FILE_MAX_UPLOAD_SIZE:
                logger.warning(f"File {upload_file.filename} exceeds size limit")
                raise HTTPException(status_code=413, detail=f"File {upload_file.filename} exceeds size limit")

            logger.info(f"Saving file {upload_file.filename} to disk")
            file_info = await file_manager.save_file(
                session_id=session_id, file_content=content, filename=upload_file.filename
            )
            logger.info(f"File saved successfully with ID: {file_info['id']}")

            uploaded_files.append(
                FileObject(
                    name=upload_file.filename,
                    id=file_info["id"],
                    session_id=session_id,
                    size=file_info["size"],
                    content_type=upload_file.content_type or "application/octet-stream",
                    metadata={
                        "content-type": upload_file.content_type or "application/octet-stream",
                        "original-filename": upload_file.filename,
                    },
                )
            )

        logger.info(f"Successfully uploaded {len(uploaded_files)} files for session: {session_id}")
        return UploadResponse(message="success", session_id=session_id, files=uploaded_files)
    except ValueError as e:
        logger.error(f"Upload failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error during upload: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during file upload")


@router.get(
    "/download/{session_id}/{file_id}",
    responses={404: {"model": ErrorResponse}},
    description="Download a file by session ID and file ID",
    summary="Download a file",
    response_description="Returns the file as a streaming response",
    tags=["files"],
)
async def download_file(session_id: str, file_id: str):
    """Download a file."""
    try:
        file_info = await file_manager.get_file(session_id, file_id)
        logger.debug("Retrieved file info", extra={"file_info": file_info})

        return StreamingResponse(
            content=iter([file_info["content"]]),
            media_type=file_info["contentType"],
            headers={"Content-Disposition": f'attachment; filename="{file_info["metadata"]["original-filename"]}"'},
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"File {file_id} not found")


@router.get(
    "/files/{session_id}",
    response_model=List[FileObject],
    description="List all files associated with a session",
    summary="List session files",
    response_description="Returns a list of file objects for the session",
    tags=["files"],
)
async def list_files(session_id: str = Path(..., description=PathParams.model_fields["session_id"].description)):
    """List files for a session."""
    files = await file_manager.list_files(session_id)
    return [
        FileObject(
            name=file["name"],
            id=file["id"],
            session_id=session_id,
            size=file["size"],
            lastModified=file["lastModified"],
            etag=file["etag"],
            contentType=file["contentType"],
            metadata=FileMetadata(**file["metadata"]),
        )
        for file in files
    ]


@router.delete(
    "/files/{session_id}/{file_id}",
    response_model=SuccessResponse,
    responses={404: {"model": ErrorResponse}},
    description="Delete a specific file by session ID and file ID",
    summary="Delete a file",
    response_description="Returns a success message if file was deleted",
    tags=["files"],
)
async def delete_file(
    session_id: str = Path(..., description=PathParams.model_fields["session_id"].description),
    file_id: str = Path(..., description=PathParams.model_fields["file_id"].description),
):
    """Delete a file."""
    try:
        await file_manager.delete_file(session_id, file_id)
        return {"message": "File deleted successfully"}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"File {file_id} not found")

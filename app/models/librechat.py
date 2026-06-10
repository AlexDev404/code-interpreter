from typing import Optional, List, Dict, Literal
from pydantic import BaseModel

from .base import FileObject, ExecuteResponse, UploadResponse, Error


class LibreChatFileRef(BaseModel):
    """File reference returned in execution responses.

    LibreChat downloads each generated file from
    /download/{storage_session_id}/{id} and uses `name` (which may contain
    directories) to place it back at the same /mnt/data path later.
    """

    id: str
    name: str
    storage_session_id: Optional[str] = None
    # Unchanged input passthroughs are marked inherited so LibreChat skips
    # re-downloading them as generated artifacts
    inherited: Optional[bool] = None


class LibreChatUploadFileObject(BaseModel):
    """LibreChat-specific upload file object."""

    fileId: str  # just file ID with slightly different name
    filename: str  # just file name with slightly different name


class LibreChatUploadResponse(BaseModel):
    """Upload response: LibreChat reads `storage_session_id` and `files[0].fileId`."""

    message: str
    storage_session_id: str
    files: List[LibreChatUploadFileObject]

    @classmethod
    def from_base(cls, response: UploadResponse) -> "LibreChatUploadResponse":
        """Convert base UploadResponse to LibreChat format."""
        return cls(
            message="success",
            storage_session_id=response.session_id,
            files=[LibreChatUploadFileObject(fileId=f.id, filename=f.name) for f in response.files],
        )


class LibreChatBatchUploadFileResult(BaseModel):
    """Per-file result in a batch upload response."""

    status: Literal["success", "error"]
    filename: str
    fileId: Optional[str] = None
    error: Optional[str] = None


class LibreChatBatchUploadResponse(BaseModel):
    """Batch upload response: all files share one storage session."""

    message: str
    storage_session_id: str
    files: List[LibreChatBatchUploadFileResult]
    succeeded: int
    failed: int


class LibreChatSessionObjectInfo(BaseModel):
    """Response for the session object liveness check.

    LibreChat compares `lastModified` against a 23h window to decide
    whether to re-upload the file.
    """

    lastModified: str


class LibreChatFileObject(BaseModel):
    """File object in session file listings (GET /files/{session_id}).

    LibreChat's `normalizeSessionFile` reads `id`, `storage_session_id`,
    `name` ("session_id/fileId" form) and `metadata['original-filename']`.
    """

    name: str  # Format: session_id/fileId
    id: str
    storage_session_id: str
    lastModified: str
    metadata: Dict[str, Optional[str]]

    @classmethod
    def from_base(cls, file: FileObject) -> "LibreChatFileObject":
        """Convert base FileObject to LibreChat format."""
        original_filename = (file.metadata.original_filename if file.metadata else None) or file.name
        return cls(
            name=f"{file.session_id}/{file.id}",
            id=file.id,
            storage_session_id=file.session_id,
            lastModified=file.lastModified,
            metadata={
                "content-type": file.contentType,
                "original-filename": original_filename,
            },
        )


class LibreChatExecuteResponse(BaseModel):
    """LibreChat-specific execution response."""

    session_id: str
    stdout: str
    stderr: str
    files: Optional[List[LibreChatFileRef]] = None

    @classmethod
    def from_base(cls, response: ExecuteResponse) -> "LibreChatExecuteResponse":
        """Convert base ExecuteResponse to LibreChat format."""
        return cls(
            session_id=response.session_id,
            stdout=response.run.stdout or "",
            stderr=response.run.stderr or "",
            files=(
                [
                    LibreChatFileRef(id=f.id, name=f.name, storage_session_id=f.storage_session_id)
                    for f in response.files
                ]
                if response.files
                else None
            ),
        )


class LibreChatError(BaseModel):
    """LibreChat-specific error response."""

    message: str

    @classmethod
    def from_base(cls, error: Error) -> "LibreChatError":
        """Convert base Error to LibreChat format."""
        return cls(message=f"{error.error}: {error.details}" if error.details else error.error)

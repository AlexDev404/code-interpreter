import docker
from loguru import logger
from pathlib import Path
from typing import Dict, List, Optional, Any, Literal, Set, Tuple
import time
from docker.errors import APIError, ImageNotFound
import asyncio
import contextlib
import fcntl
from dataclasses import dataclass
from datetime import datetime
import hashlib
import mimetypes
from app.shared.const import UPLOAD_PATH
from app.utils.generate_id import generate_id
import aiodocker
import json
import os

from ..shared.config import get_settings
from .database import db_manager

settings = get_settings()


@dataclass
class ContainerMetrics:
    start_time: datetime
    container_id: str
    memory_usage: int = 0
    cpu_usage: float = 0.0


@dataclass
class FileState:
    """Tracks the state of a file for change detection."""

    path: Path
    size: int
    mtime: float
    md5_hash: str
    exists: bool = True


class DockerExecutor:
    """Executes code in Docker containers with file management."""

    WORK_DIR = "/mnt/data"  # Working directory will be the same as data mount point
    DATA_MOUNT = "/mnt/data"  # Mount point for session data

    # Language-specific execution commands
    LANGUAGE_EXECUTORS = {
        "py": ["python", "-c"],
        "r": ["Rscript", "-e"],
        "bash": ["bash", "-c"],
        "js": ["node", "-e"],
        # Node 24 strips TypeScript types natively; run eval input as a TS ES module
        "ts": ["node", "--input-type=module-typescript", "-e"],
    }

    # Unprivileged user (user:group) inside each language's container image.
    # Jupyter images ship with jovyan:users, the official Node image with node:node.
    DEFAULT_CONTAINER_USER = ("jovyan", "users")
    LANGUAGE_CONTAINER_USERS = {
        "js": ("node", "node"),
        "ts": ("node", "node"),
    }

    # Language-specific messages
    LANGUAGE_SPECIFIC_MESSAGES = {
        "py": {"empty_output": "Empty. Make sure to explicitly print() the results in Python"},
        "r": {"empty_output": "Empty. Make sure to use print() or cat() to display results in R"},
        "bash": {"empty_output": "Empty. Make sure the command writes its results to stdout (e.g. echo, cat)"},
        "js": {"empty_output": "Empty. Make sure to explicitly console.log() the results in JavaScript"},
        "ts": {"empty_output": "Empty. Make sure to explicitly console.log() the results in TypeScript"},
    }

    # Label used to identify persistent containers managed by this executor
    CONTAINER_LABEL = "code-interpreter-session"

    def __init__(self):
        self._container_semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_CONTAINERS)
        self._active_containers: Dict[str, ContainerMetrics] = {}
        self._lock = asyncio.Lock()
        self._docker = None  # Will be initialized in initialize()
        self._image_pull_locks: Dict[str, asyncio.Lock] = {}
        # Maps session_id -> container id for persistent containers
        self._session_containers: Dict[str, str] = {}
        # Tracks last activity time for each session container
        self._session_last_activity: Dict[str, datetime] = {}
        self._cleanup_task: Optional[asyncio.Task] = None

    async def initialize(self):
        """Initialize the Docker client."""
        try:
            if self._docker is None:
                self._docker = aiodocker.Docker()
            else:
                # Check if the client is still valid
                if not await self._validate_docker_connection():
                    # Reinitialize if there was an error
                    await self.close()
                    self._docker = aiodocker.Docker()

            # Verify Docker connectivity
            try:
                await self._docker.version()
            except Exception as exc:
                raise RuntimeError(
                    "Cannot connect to Docker socket. Make sure /var/run/docker.sock is mounted:\n"
                    "  volumes:\n"
                    "    - /var/run/docker.sock:/var/run/docker.sock"
                ) from exc

            # Recover existing persistent containers
            await self._recover_session_containers()

            # Start idle container cleanup task
            if self._cleanup_task is None:
                self._cleanup_task = asyncio.create_task(self._idle_container_cleanup_loop())

            logger.info("Docker client initialized successfully")
            return self
        except Exception as e:
            logger.error(f"Error initializing Docker client: {str(e)}")
            raise

    async def close(self):
        """Close the Docker client."""
        if self._cleanup_task is not None:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
        if self._docker is not None:
            await self._docker.close()
            self._docker = None

    async def _recover_session_containers(self):
        """Recover existing persistent session containers on startup."""
        try:
            containers = await self._docker.containers.list(
                all=True,
                filters=json.dumps({"label": [self.CONTAINER_LABEL]}),
            )
            for container in containers:
                info = await container.show()
                labels = info.get("Config", {}).get("Labels", {})
                session_id = labels.get("session-id")
                if session_id and info["State"]["Running"]:
                    self._session_containers[session_id] = container.id
                    self._session_last_activity[session_id] = datetime.now()
                    logger.info(f"Recovered persistent container {container.id} for session {session_id}")
                elif session_id and not info["State"]["Running"]:
                    # Remove stopped containers from a previous run
                    try:
                        await container.delete(force=True)
                        logger.info(f"Removed stopped persistent container {container.id} for session {session_id}")
                    except Exception as e:
                        logger.warning(f"Failed to remove stopped container {container.id}: {e}")
        except Exception as e:
            logger.warning(f"Error recovering session containers: {e}")

    async def _idle_container_cleanup_loop(self):
        """Periodically check for and remove idle persistent containers."""
        while True:
            try:
                await asyncio.sleep(3600)  # Check every hour
                await self._cleanup_idle_containers()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in idle container cleanup loop: {e}")

    async def _cleanup_idle_containers(self):
        """Remove persistent containers that have been idle longer than CONTAINER_MAX_IDLE_TIME."""
        max_idle_seconds = settings.CONTAINER_MAX_IDLE_TIME
        now = datetime.now()
        sessions_to_remove = []

        async with self._lock:
            for session_id, last_activity in self._session_last_activity.items():
                idle_seconds = (now - last_activity).total_seconds()
                if idle_seconds > max_idle_seconds:
                    sessions_to_remove.append(session_id)

        for session_id in sessions_to_remove:
            await self._remove_session_container(session_id)
            logger.info(f"Removed idle container for session {session_id} (idle for > {max_idle_seconds}s)")

    async def _remove_session_container(self, session_id: str):
        """Remove a persistent container for a session."""
        async with self._lock:
            container_id = self._session_containers.pop(session_id, None)
            self._session_last_activity.pop(session_id, None)

        if container_id and self._docker:
            try:
                container = await self._docker.containers.get(container_id)
                await container.delete(force=True)
                logger.info(f"Deleted persistent container {container_id} for session {session_id}")
            except Exception as e:
                logger.warning(f"Error removing container {container_id}: {e}")

    async def _get_or_create_container(self, session_id: str, lang: str, config: Dict[str, Any]):
        """Get an existing persistent container for the session or create a new one."""
        async with self._lock:
            container_id = self._session_containers.get(session_id)

        # Try to reuse existing container
        if container_id:
            try:
                container = await self._docker.containers.get(container_id)
                info = await container.show()
                if info["State"]["Running"]:
                    # Update last activity
                    async with self._lock:
                        self._session_last_activity[session_id] = datetime.now()
                    logger.info(f"Reusing persistent container {container_id} for session {session_id}")
                    return container
                else:
                    # Container stopped unexpectedly, remove tracking
                    logger.warning(f"Persistent container {container_id} is not running, creating new one")
                    async with self._lock:
                        self._session_containers.pop(session_id, None)
                        self._session_last_activity.pop(session_id, None)
                    try:
                        await container.delete(force=True)
                    except Exception:
                        pass
            except Exception as e:
                logger.warning(f"Error accessing container {container_id}: {e}, creating new one")
                async with self._lock:
                    self._session_containers.pop(session_id, None)
                    self._session_last_activity.pop(session_id, None)

        # Create new persistent container
        container = await self._docker.containers.create(config=config)
        await container.start()

        # Wait for container to be running
        start_time = time.time()
        while True:
            info = await container.show()
            if info["State"]["Running"]:
                break
            if time.time() - start_time > 10:
                raise RuntimeError("Container failed to start properly")
            await asyncio.sleep(0.1)

        # Track the container
        async with self._lock:
            self._session_containers[session_id] = container.id
            self._session_last_activity[session_id] = datetime.now()
            self._active_containers[container.id] = ContainerMetrics(
                start_time=datetime.now(), container_id=container.id
            )

        logger.info(f"Created persistent container {container.id} for session {session_id}")
        return container

    @contextlib.contextmanager
    def _file_lock(self, path: Path):
        """Provide file-based locking for concurrent operations."""
        lock_path = path.parent / f"{path.name}.lock"
        lock_file = open(lock_path, "w+")
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            lock_file.close()
            lock_path.unlink(missing_ok=True)

    def _scan_directory(self, directory: Path) -> Dict[str, FileState]:
        """
        Recursively scan a directory and collect file states.
        Returns a dictionary mapping relative file paths to their FileState objects.
        """
        file_states = {}

        if not directory.exists():
            logger.warning(f"Directory {directory} does not exist")
            return file_states

        # Walk through the directory recursively
        for root, _, files in os.walk(directory):
            root_path = Path(root)

            # Compute relative path from the base directory
            rel_root = root_path.relative_to(directory)

            for filename in files:
                # Skip lock files
                if filename.endswith(".lock"):
                    continue

                file_path = root_path / filename

                # Compute relative path for dictionary key
                if rel_root == Path("."):
                    rel_path = filename
                else:
                    rel_path = str(rel_root / filename)

                try:
                    # Get file stats
                    stat = file_path.stat()
                    size = stat.st_size
                    mtime = stat.st_mtime

                    # Calculate MD5 hash for content comparison
                    md5_hash = hashlib.md5(file_path.read_bytes()).hexdigest()

                    # Store file state
                    file_states[rel_path] = FileState(path=file_path, size=size, mtime=mtime, md5_hash=md5_hash)
                    logger.debug(f"Scanned file: {rel_path}, size: {size}, hash: {md5_hash}")
                except (PermissionError, FileNotFoundError) as e:
                    logger.warning(f"Error scanning file {file_path}: {str(e)}")
                    continue

        return file_states

    def _find_changed_files(self, before_states: Dict[str, FileState], after_states: Dict[str, FileState]) -> Set[str]:
        """
        Compare before and after file states to identify new or modified files.
        Returns a set of relative paths of changed files.
        """
        changed_files = set()

        # Find new or modified files
        for rel_path, after_state in after_states.items():
            if rel_path not in before_states:
                # New file
                logger.info(f"New file detected: {rel_path}")
                changed_files.add(rel_path)
            else:
                before_state = before_states[rel_path]
                # Check if file content was modified. mtime alone is not enough:
                # rewriting identical content updates the timestamp but is not a change.
                if before_state.size != after_state.size or before_state.md5_hash != after_state.md5_hash:
                    logger.info(
                        f"Modified file detected: {rel_path}, before={before_state.size}:{before_state.md5_hash}:{before_state.mtime}, after={after_state.size}:{after_state.md5_hash}:{after_state.mtime}"
                    )
                    changed_files.add(rel_path)
                else:
                    logger.info(
                        f"Unchanged file: {rel_path}, size={after_state.size}, hash={after_state.md5_hash}, mtime={after_state.mtime}"
                    )

        # Add debug logs for summarizing scan results
        for rel_path in before_states:
            if rel_path not in after_states:
                logger.info(f"File deleted: {rel_path}")

        logger.info(
            f"Before scan: {len(before_states)} files, After scan: {len(after_states)} files, Changed: {len(changed_files)} files"
        )

        return changed_files

    async def _update_container_metrics(self, container) -> None:
        """Update metrics for a running container."""
        try:
            stats_data = await container.stats(stream=False)

            # Handle empty stats data
            if not stats_data:
                logger.warning(f"No stats data available for container {container.id}")
                return

            # aiodocker returns stats data differently, handle it appropriately
            stats = stats_data[0] if isinstance(stats_data, list) and stats_data else stats_data
            if not stats:
                logger.warning(f"No stats available for container {container.id}")
                return

            # Calculate memory usage - handle both possible formats
            memory_usage = 0
            if isinstance(stats, dict):
                memory_stats = stats.get("memory_stats", {})
                memory_usage = memory_stats.get("usage", 0)
            elif isinstance(stats, bytes):
                # If stats is returned as bytes, decode it
                try:
                    stats = json.loads(stats.decode())
                    memory_stats = stats.get("memory_stats", {})
                    memory_usage = memory_stats.get("usage", 0)
                except json.JSONDecodeError:
                    logger.warning(f"Could not decode stats for container {container.id}")
                    return

            # Calculate CPU usage
            cpu_stats = stats.get("cpu_stats", {})
            precpu_stats = stats.get("precpu_stats", {})

            cpu_usage_stats = cpu_stats.get("cpu_usage", {})
            precpu_usage_stats = precpu_stats.get("cpu_usage", {})

            cpu_delta = cpu_usage_stats.get("total_usage", 0) - precpu_usage_stats.get("total_usage", 0)
            system_delta = cpu_stats.get("system_cpu_usage", 0) - precpu_stats.get("system_cpu_usage", 0)

            cpu_usage = (cpu_delta / system_delta) * 100.0 if system_delta > 0 else 0.0

            logger.info(f"Container {container.id} memory usage: {memory_usage}, CPU usage: {cpu_usage}")

            async with self._lock:
                if container.id in self._active_containers:
                    self._active_containers[container.id].memory_usage = memory_usage
                    self._active_containers[container.id].cpu_usage = cpu_usage
        except Exception as e:
            logger.error(f"Error updating metrics for container {container.id}: {str(e)}")

    def _clean_output(self, raw_output: bytes) -> str:
        """Clean Docker multiplexed output format."""
        # Skip the first 8 bytes of each frame (header) and combine the rest
        output_parts = []
        i = 0
        while i < len(raw_output):
            # Each frame starts with 8 bytes of header
            if i + 8 > len(raw_output):
                break
            # The fourth byte indicates the stream (1 = stdout, 2 = stderr)
            # The next 4 bytes contain the size of the frame
            frame_size = int.from_bytes(raw_output[i + 4 : i + 8], byteorder="big")
            # Extract the frame content
            if i + 8 + frame_size > len(raw_output):
                break
            frame_data = raw_output[i + 8 : i + 8 + frame_size]
            output_parts.append(frame_data)
            i += 8 + frame_size

        return b"".join(output_parts).decode("utf-8").strip()

    async def execute(
        self,
        code: str,
        session_id: str,
        lang: Literal["py", "r", "bash", "js", "ts"],
        files: Optional[List[Dict[str, Any]]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute code in a Docker container with file management."""
        container = None
        config = config or {}

        try:
            # Ensure Docker client is initialized and valid
            if self._docker is None:
                await self.initialize()
            else:
                # Verify Docker client is still valid
                if not await self._validate_docker_connection():
                    logger.warning("Docker client validation failed, reinitializing")
                    await self.close()
                    await self.initialize()

            # Create session directory before anything else
            session_path: Path = UPLOAD_PATH / session_id
            logger.info(f"Session path: {session_path}")
            session_path.mkdir(parents=True, exist_ok=True)

            # Log debug information
            logger.info(f"Session directory: {session_path}")
            logger.info(f"Session directory contents: {list(session_path.glob('*'))}")
            logger.info(f"Code to execute: {code}")

            # Scan directory before execution to track file state
            logger.info(f"Scanning directory {session_path} before code execution")
            before_file_states = self._scan_directory(session_path)
            logger.info(f"Found {len(before_file_states)} files before execution")

            async with self._container_semaphore:
                try:
                    # Ensure the image is available
                    image_name = settings.LANGUAGE_CONTAINERS.get(lang)
                    logger.info(f"Using container image: {image_name}")

                    try:
                        # Check if image exists
                        await self._docker.images.inspect(image_name)
                        logger.info(f"Image {image_name} is available")
                    except Exception as e:
                        # Check if it's a 404 error (image not found)
                        if isinstance(e, aiodocker.exceptions.DockerError) and e.status == 404:
                            # Get or create a lock for this specific image
                            if image_name not in self._image_pull_locks:
                                self._image_pull_locks[image_name] = asyncio.Lock()

                            # Acquire the lock for this image to prevent multiple pulls
                            async with self._image_pull_locks[image_name]:
                                # Check again if the image exists (another request might have pulled it while we were waiting)
                                try:
                                    await self._docker.images.inspect(image_name)
                                    logger.info(f"Image {image_name} is now available (pulled by another request)")
                                except Exception as check_again_error:
                                    if (
                                        isinstance(check_again_error, aiodocker.exceptions.DockerError)
                                        and check_again_error.status == 404
                                    ):
                                        # Pull the image if not available
                                        logger.info(f"Image {image_name} not found, pulling...")
                                        try:
                                            # Pull using aiodocker
                                            await self._docker.images.pull(image_name)
                                            logger.info(f"Successfully pulled image {image_name}")
                                        except Exception as pull_error:
                                            logger.error(f"Failed to pull image: {str(pull_error)}")
                                            return {
                                                "stdout": "",
                                                "stderr": f"Failed to pull required Docker image: {image_name}. Error: {str(pull_error)}",
                                                "status": "error",
                                                "files": [],
                                            }
                                    else:
                                        # Re-raise if it's not a 404 error
                                        logger.error(f"Error checking for image {image_name}: {str(check_again_error)}")
                                        raise
                        else:
                            # Re-raise if it's not a 404 error
                            logger.error(f"Error checking for image {image_name}: {str(e)}")
                            raise

                    # Get container configuration, with provided config overriding settings
                    memory_limit_mb = config.get("memory_limit_mb", settings.CONTAINER_MEMORY_LIMIT_MB)
                    cpu_limit = config.get("cpu_limit", settings.CONTAINER_CPU_LIMIT)
                    network_enabled = config.get("network_enabled", settings.DOCKER_NETWORK_ENABLED)
                    root_access = config.get("root_access", settings.CONTAINER_ROOT_ACCESS)

                    logger.info(
                        f"Container config - Memory: {memory_limit_mb}MB, CPU: {cpu_limit}, Network: {network_enabled}, Root: {root_access}"
                    )

                    # Create container config
                    container_config = {
                        "Image": image_name,
                        "Cmd": ["sleep", "infinity"],
                        "WorkingDir": f"{self.DATA_MOUNT}/{session_id}",
                        "NetworkDisabled": not network_enabled,
                        "Labels": {
                            self.CONTAINER_LABEL: "true",
                            "session-id": session_id,
                            "language": lang,
                        },
                        "HostConfig": {
                            "Memory": memory_limit_mb * 1024 * 1024,  # Convert MB to bytes
                            "NanoCpus": int(cpu_limit * 1e9),  # Convert CPU cores to nano CPUs
                            "Mounts": [
                                {
                                    "Type": "volume",
                                    "Source": settings.UPLOADS_VOLUME_NAME,
                                    "Target": self.DATA_MOUNT,
                                }
                            ],
                        },
                    }

                    # Get or create persistent container for this session
                    container = await self._get_or_create_container(session_id, lang, container_config)

                    # Fix permissions for mounted directory
                    exec_user, exec_group = self.LANGUAGE_CONTAINER_USERS.get(lang, self.DEFAULT_CONTAINER_USER)

                    # If root access is enabled, run as root
                    if root_access:
                        exec_user = "root"
                        exec_group = "root"

                    exec = await container.exec(
                        cmd=["chown", "-R", f"{exec_user}:{exec_group}", f"{self.DATA_MOUNT}/{session_id}"],
                        user="root",
                        stdout=True,
                        stderr=True,
                    )
                    # Use raw API call to get output
                    exec_url = f"exec/{exec._id}/start"
                    async with self._docker._query(
                        exec_url,
                        method="POST",
                        headers={"Content-Type": "application/json"},
                        data=json.dumps({"Detach": False, "Tty": False}),
                    ) as response:
                        output = await response.read()
                        output_text = self._clean_output(output)

                    # Execute the code with the appropriate interpreter
                    logger.info(f"Code to execute: {code}")
                    logger.info(f"Language: {lang}")

                    # Get the execution command for the specified language
                    exec_cmd = self.LANGUAGE_EXECUTORS.get(lang, self.LANGUAGE_EXECUTORS["py"])
                    logger.info(f"Using execution command: {exec_cmd}")

                    # Execute the code with the appropriate interpreter
                    exec = await container.exec(cmd=[*exec_cmd, code], user=exec_user, stdout=True, stderr=True)
                    # Use raw API call to get output
                    exec_url = f"exec/{exec._id}/start"
                    async with self._docker._query(
                        exec_url,
                        method="POST",
                        headers={"Content-Type": "application/json"},
                        data=json.dumps({"Detach": False, "Tty": False}),
                    ) as response:
                        output = await response.read()
                        output_text = self._clean_output(output)

                    # Check execution status
                    exec_inspect = await exec.inspect()
                    if exec_inspect["ExitCode"] != 0:
                        return {"stdout": "", "stderr": output_text, "status": "error", "files": []}

                    # Scan directory after execution to detect changes
                    logger.info(f"Scanning directory {session_path} after code execution")
                    after_file_states = self._scan_directory(session_path)
                    logger.info(f"Found {len(after_file_states)} files after execution")

                    # Identify changed files
                    changed_file_paths = self._find_changed_files(before_file_states, after_file_states)
                    logger.info(f"Detected {len(changed_file_paths)} changed files: {changed_file_paths}")

                    # Process only new or modified files
                    output_files = []
                    existing_filenames = {file["name"] for file in (files or [])}
                    logger.info(f"Existing filenames: {existing_filenames}")

                    for rel_path in changed_file_paths:
                        file_path = session_path / rel_path
                        if file_path.is_file():
                            file_id = generate_id()
                            file_size = file_path.stat().st_size
                            logger.info(f"Processing changed file: {file_path}, size: {file_size}")

                            # Calculate file metadata
                            content_type, _ = mimetypes.guess_type(file_path.name) or ("application/octet-stream", None)
                            etag = hashlib.md5(str(file_path.stat().st_mtime).encode()).hexdigest()

                            # Prepare file data for database
                            # Use directory structure in filepath if present
                            filepath = f"{session_id}/{rel_path}"
                            filename = Path(rel_path).name

                            file_data = {
                                "id": file_id,
                                "session_id": session_id,
                                "filename": filename,
                                "filepath": filepath,
                                "size": file_size,
                                "content_type": content_type,
                                "original_filename": filename,
                                "etag": etag,
                                # Relative path within the session dir; preserves directory
                                # structure so LibreChat can restage nested artifacts
                                "relative_path": rel_path,
                            }
                            logger.info(f"Saving file metadata to database: {file_data}")

                            # Save to database
                            await db_manager.add_file(file_data)
                            output_files.append(file_data)

                    return {
                        "stdout": output_text,
                        "stderr": "",
                        "status": "ok",
                        "files": output_files,
                        "metrics": {
                            "memory_usage": self._active_containers.get(container.id, ContainerMetrics(start_time=datetime.now(), container_id=container.id)).memory_usage,
                            "cpu_usage": self._active_containers.get(container.id, ContainerMetrics(start_time=datetime.now(), container_id=container.id)).cpu_usage,
                            "execution_time": (
                                datetime.now() - self._active_containers.get(container.id, ContainerMetrics(start_time=datetime.now(), container_id=container.id)).start_time
                            ).total_seconds(),
                        },
                    }

                except Exception as e:
                    logger.error(f"Error in docker execution: {str(e)}")
                    return {
                        "stdout": "",
                        "stderr": "Failed to execute code. Please try again.",
                        "status": "error",
                        "files": [],
                    }

                finally:
                    # Update metrics but do NOT delete the container (persistent)
                    if container:
                        try:
                            asyncio.create_task(self._update_container_metrics(container))
                        except Exception as e:
                            logger.error(f"Error updating container metrics: {str(e)}")

        except Exception as e:
            logger.error(f"Error in docker execution: {str(e)}")
            return {
                "stdout": "",
                "stderr": "Failed to execute code. Please try again.",
                "status": "error",
                "files": [],
            }

    async def get_active_containers(self) -> List[Dict[str, Any]]:
        """Get information about currently running containers."""
        async with self._lock:
            return [
                {"container_id": container_id, "metrics": metrics.__dict__}
                for container_id, metrics in self._active_containers.items()
            ]

    async def _validate_docker_connection(self):
        """Validate that the Docker connection is working properly."""
        try:
            # Instead of using ping(), try to get the Docker version
            # which is a simple API call that should work if the connection is valid
            await self._docker.version()
            logger.debug("Docker connection validated")
            return True
        except Exception as e:
            logger.warning(f"Docker connection validation failed: {str(e)}")
            return False


# Create singleton instance
docker_executor = DockerExecutor()

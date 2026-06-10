from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_bash_code_execution():
    """Test executing a simple bash command."""
    response = client.post("/v1/execute", json={"code": "echo 'Hello from Bash!'", "lang": "bash"})

    assert response.status_code == 200
    result = response.json()
    assert result["run"]["status"] == "ok"
    assert "Hello from Bash!" in result["run"]["stdout"]
    assert result["run"]["stderr"] == ""
    assert isinstance(result["files"], list)
    assert result["language"] == "bash"
    assert "Bash" in result["version"]


def test_bash_python_invocation():
    """Test running python via bash, as LibreChat's bash_tool does."""
    response = client.post("/v1/execute", json={"code": "python3 -c 'print(1232**4)'", "lang": "bash"})

    assert response.status_code == 200
    result = response.json()
    assert result["run"]["status"] == "ok"
    assert str(1232**4) in result["run"]["stdout"]


def test_bash_code_execution_error():
    """Test executing a bash command that fails."""
    response = client.post("/v1/execute", json={"code": "cat /nonexistent/file.txt", "lang": "bash"})

    assert response.status_code == 200
    result = response.json()
    assert result["run"]["status"] == "error"
    assert isinstance(result["files"], list)


def test_bash_empty_output_message():
    """Test that a command with no output returns the bash-specific hint."""
    response = client.post("/v1/execute", json={"code": "true", "lang": "bash"})

    assert response.status_code == 200
    result = response.json()
    assert result["run"]["status"] == "ok"
    assert result["run"]["stdout"] == "Empty. Make sure the command writes its results to stdout (e.g. echo, cat)"


def test_bash_session_continuity():
    """Test that files persist across executions when session_id is sent in the body."""
    # First call: write a file, no session_id provided
    response1 = client.post(
        "/v1/execute", json={"code": "echo 'persisted content' > /mnt/data/note.txt", "lang": "bash"}
    )

    assert response1.status_code == 200
    result1 = response1.json()
    assert result1["run"]["status"] == "ok"
    session_id = result1["session_id"]
    assert session_id
    # The new file should be detected as an output file
    assert any(f["name"] == "note.txt" for f in result1["files"])

    # Second call: continue the same session via session_id and read the file back
    response2 = client.post(
        "/v1/execute", json={"code": "cat /mnt/data/note.txt", "lang": "bash", "session_id": session_id}
    )

    assert response2.status_code == 200
    result2 = response2.json()
    assert result2["run"]["status"] == "ok"
    assert "persisted content" in result2["run"]["stdout"]
    assert result2["session_id"] == session_id


def test_session_id_path_traversal_rejected():
    """Test that a session_id with path traversal is rejected with a 422 before execution."""
    response = client.post(
        "/v1/execute",
        json={"code": "cat /mnt/data/secret", "lang": "bash", "session_id": "../../etc/passwd"},
    )

    assert response.status_code == 422


def test_librechat_bash_exec():
    """Test the LibreChat exec route with bash, matching bash_tool's request/response contract."""
    response = client.post("/v1/librechat/exec", json={"code": "echo 'librechat bash'", "lang": "bash"})

    assert response.status_code == 200
    result = response.json()
    assert "session_id" in result
    assert "librechat bash" in result["stdout"]
    assert "stderr" in result


def test_librechat_bash_session_continuity():
    """Test LibreChat-style session continuation: write in call 1, cat in call 2."""
    response1 = client.post(
        "/v1/librechat/exec", json={"code": "printf 'step one' > /mnt/data/state.txt", "lang": "bash"}
    )

    assert response1.status_code == 200
    session_id = response1.json()["session_id"]

    response2 = client.post(
        "/v1/librechat/exec", json={"code": "cat /mnt/data/state.txt", "lang": "bash", "session_id": session_id}
    )

    assert response2.status_code == 200
    result2 = response2.json()
    assert "step one" in result2["stdout"]
    assert result2["session_id"] == session_id

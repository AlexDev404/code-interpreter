from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_js_code_execution():
    """Test executing simple JavaScript code with Node.js."""
    response = client.post("/v1/execute", json={"code": "console.log('Hello from Node!')", "lang": "js"})

    assert response.status_code == 200
    result = response.json()
    assert result["run"]["status"] == "ok"
    assert "Hello from Node!" in result["run"]["stdout"]
    assert result["run"]["stderr"] == ""
    assert isinstance(result["files"], list)
    assert result["language"] == "js"
    assert "JavaScript" in result["version"]


def test_js_code_execution_error():
    """Test executing JavaScript code that throws."""
    response = client.post("/v1/execute", json={"code": "throw new Error('boom')", "lang": "js"})

    assert response.status_code == 200
    result = response.json()
    assert result["run"]["status"] == "error"
    assert isinstance(result["files"], list)


def test_js_empty_output_message():
    """Test that JavaScript code with no output returns the js-specific hint."""
    response = client.post("/v1/execute", json={"code": "const x = 1 + 1", "lang": "js"})

    assert response.status_code == 200
    result = response.json()
    assert result["run"]["status"] == "ok"
    assert result["run"]["stdout"] == "Empty. Make sure to explicitly console.log() the results in JavaScript"


def test_js_file_output():
    """Test that files written by JavaScript code are detected as output files."""
    response = client.post(
        "/v1/execute",
        json={
            "code": "const fs = require('fs'); fs.writeFileSync('/mnt/data/out.txt', 'node output'); console.log('written')",
            "lang": "js",
        },
    )

    assert response.status_code == 200
    result = response.json()
    assert result["run"]["status"] == "ok"
    assert any(f["name"] == "out.txt" for f in result["files"])


def test_ts_code_execution():
    """Test executing TypeScript code with type annotations via Node.js type stripping."""
    code = "const greet = (name: string): string => `Hello, ${name}!`; console.log(greet('TypeScript'))"
    response = client.post("/v1/execute", json={"code": code, "lang": "ts"})

    assert response.status_code == 200
    result = response.json()
    assert result["run"]["status"] == "ok"
    assert "Hello, TypeScript!" in result["run"]["stdout"]
    assert result["language"] == "ts"
    assert "TypeScript" in result["version"]


def test_ts_interface_execution():
    """Test that interfaces and typed objects work in TypeScript."""
    code = (
        "interface Point { x: number; y: number }\n"
        "const p: Point = { x: 3, y: 4 };\n"
        "console.log(Math.sqrt(p.x ** 2 + p.y ** 2))"
    )
    response = client.post("/v1/execute", json={"code": code, "lang": "ts"})

    assert response.status_code == 200
    result = response.json()
    assert result["run"]["status"] == "ok"
    assert "5" in result["run"]["stdout"]


def test_ts_empty_output_message():
    """Test that TypeScript code with no output returns the ts-specific hint."""
    response = client.post("/v1/execute", json={"code": "const x: number = 42", "lang": "ts"})

    assert response.status_code == 200
    result = response.json()
    assert result["run"]["status"] == "ok"
    assert result["run"]["stdout"] == "Empty. Make sure to explicitly console.log() the results in TypeScript"

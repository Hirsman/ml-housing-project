from pathlib import Path

REQUIRED_PATHS = [
    "backend",
    "frontend",
    "src",
    "tests",
    "requirements.txt",
    "pyproject.toml",
]

FORBIDDEN_PATTERNS = [
    "train",
    "pipeline",
]


def test_required_paths_exist():
    root = Path(".")
    missing = [p for p in REQUIRED_PATHS if not (root / p).exists()]
    assert not missing, f"Missing project elements: {missing}"


def test_no_training_code_in_backend():
    backend = Path("backend")
    assert backend.exists(), "backend folder missing"

    forbidden_files = []

    for file in backend.rglob("*.py"):
        name = file.name.lower()
        if any(pattern in name for pattern in FORBIDDEN_PATTERNS):
            forbidden_files.append(str(file))

    assert not forbidden_files, f"Training code found in backend: {forbidden_files}"
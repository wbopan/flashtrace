"""Frozen Aider Polyglot task inventory and execution-grounded verification."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


POLYGLOT_COMMIT = "7e0611e77b54e2dea774cdc0aa00cf9f7ed6144f"
LANGUAGES = ("cpp", "go", "java", "javascript", "python", "rust")
IGNORED_SOLUTION_FILES = {"CMakeLists.txt", "Cargo.toml"}

WHOLE_FILE_SYSTEM_PROMPT = """Act as an expert software developer.
Take requests for changes to the supplied code.
Return every changed file using exactly this format:

path/to/filename.ext
```
entire updated file content
```

The filename must be alone on the line before its fence. Return the complete
file without omissions or placeholder comments. Do not change tests. Do not
add commentary outside file listings."""


@dataclass(frozen=True)
class PolyglotTask:
    task_id: str
    language: str
    exercise: str
    relative_path: str
    solution_files: tuple[str, ...]
    test_files: tuple[str, ...]
    instructions: str
    prompt: str
    source_sha256: str


@dataclass(frozen=True)
class ParsedEdit:
    files: dict[str, str]
    format_compliant: bool
    errors: tuple[str, ...]


@dataclass(frozen=True)
class VerificationResult:
    passed: bool
    exit_code: int | None
    timed_out: bool
    infrastructure_error: str | None
    output: str
    command: tuple[str, ...]
    applied_files: tuple[str, ...]
    apply_errors: tuple[str, ...]


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_config(task_dir: Path) -> dict[str, Any]:
    config_path = task_dir / ".meta" / "config.json"
    return json.loads(config_path.read_text(encoding="utf-8"))


def _solution_files(config: dict[str, Any]) -> tuple[str, ...]:
    files = config.get("files", {}).get("solution", [])
    return tuple(
        sorted(
            str(path)
            for path in files
            if str(path) not in IGNORED_SOLUTION_FILES
        )
    )


def _instructions(task_dir: Path) -> str:
    pieces = []
    for name in ("introduction.md", "instructions.md", "instructions.append.md"):
        path = task_dir / ".docs" / name
        if path.exists():
            pieces.append(path.read_text(encoding="utf-8").strip())
    return "\n\n".join(piece for piece in pieces if piece)


def _render_prompt(
    task_dir: Path,
    instructions: str,
    solution_files: Sequence[str],
) -> str:
    listings = []
    for relative in solution_files:
        content = (task_dir / relative).read_text(encoding="utf-8")
        listings.append(f"{relative}\n```\n{content.rstrip()}\n```")
    file_list = " ".join(solution_files)
    supplied_files = "\n\n".join(listings)
    return (
        f"# Instructions\n\n{instructions}\n\n"
        f"# Supplied files\n\n{supplied_files}\n\n"
        f"Use the above instructions to modify the supplied files: {file_list}\n"
        "Do not change existing public function or class names because the tests "
        "reference them. Only use standard libraries. Return complete updated files."
    )


def discover_tasks(root: Path) -> list[PolyglotTask]:
    """Discover the complete pinned 225-task Polyglot inventory."""

    root = root.resolve()
    tasks = []
    for language in LANGUAGES:
        practice = root / language / "exercises" / "practice"
        for task_dir in sorted(path for path in practice.iterdir() if path.is_dir()):
            config = _read_config(task_dir)
            solution_files = _solution_files(config)
            if not solution_files:
                continue
            test_files = tuple(
                sorted(str(path) for path in config.get("files", {}).get("test", []))
            )
            instructions = _instructions(task_dir)
            prompt = _render_prompt(task_dir, instructions, solution_files)
            relative_path = task_dir.relative_to(root).as_posix()
            source_payload = {
                "relative_path": relative_path,
                "solution_files": {
                    path: (task_dir / path).read_text(encoding="utf-8")
                    for path in solution_files
                },
                "test_files": test_files,
                "instructions": instructions,
            }
            tasks.append(
                PolyglotTask(
                    task_id=f"polyglot-{language}-{task_dir.name}",
                    language=language,
                    exercise=task_dir.name,
                    relative_path=relative_path,
                    solution_files=solution_files,
                    test_files=test_files,
                    instructions=instructions,
                    prompt=prompt,
                    source_sha256=_sha256_text(
                        json.dumps(source_payload, sort_keys=True, ensure_ascii=False)
                    ),
                )
            )
    return tasks


def write_manifest(
    tasks: Sequence[PolyglotTask],
    path: Path,
    *,
    source_root: Path,
) -> None:
    header = {
        "schema_version": 1,
        "benchmark": "aider_polyglot",
        "source_repo": "Aider-AI/polyglot-benchmark",
        "source_commit": POLYGLOT_COMMIT,
        "source_root": str(source_root.resolve()),
        "task_count": len(tasks),
        "languages": {
            language: sum(task.language == language for task in tasks)
            for language in LANGUAGES
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"manifest": header}, ensure_ascii=False) + "\n")
        for task in tasks:
            handle.write(json.dumps(asdict(task), ensure_ascii=False) + "\n")


_FENCED_BLOCK = re.compile(
    r"(?ms)^(?P<filename>[^\n`]+?)\s*\n"
    r"```[^\n]*\n(?P<content>.*?)^```\s*$"
)
_ANY_FENCED_BLOCK = re.compile(r"(?ms)```[^\n]*\n(?P<content>.*?)^```")


def parse_whole_file_edit(
    completion: str,
    expected_files: Sequence[str],
) -> ParsedEdit:
    """Parse Aider whole-file listings while retaining format diagnostics."""

    expected = set(expected_files)
    parsed: dict[str, str] = {}
    errors = []
    for match in _FENCED_BLOCK.finditer(completion):
        filename = match.group("filename").strip()
        if filename not in expected:
            continue
        if filename in parsed:
            errors.append(f"duplicate file listing: {filename}")
            continue
        parsed[filename] = match.group("content")

    format_compliant = bool(parsed) and not errors
    if not parsed and len(expected_files) == 1:
        fallback = _ANY_FENCED_BLOCK.search(completion)
        if fallback:
            parsed[expected_files[0]] = fallback.group("content")
            errors.append("single-file fence omitted the required filename")

    if not parsed:
        errors.append("no expected whole-file listing found")
    missing = sorted(expected - set(parsed))
    if missing:
        errors.append(f"missing expected files: {', '.join(missing)}")
    return ParsedEdit(
        files=parsed,
        format_compliant=format_compliant and not missing,
        errors=tuple(errors),
    )


def _test_command(language: str) -> tuple[str, ...]:
    commands = {
        "python": ("pytest",),
        "rust": ("cargo", "test", "--", "--include-ignored"),
        "go": ("go", "test", "./..."),
        "javascript": ("/aider/benchmark/npm-test.sh",),
        "cpp": ("/aider/benchmark/cpp-test.sh",),
        "java": ("./gradlew", "test"),
    }
    return commands[language]


def _prepare_java_tests(workdir: Path, test_files: Iterable[str]) -> None:
    for relative in test_files:
        if not relative.endswith(".java"):
            continue
        path = workdir / relative
        if path.exists():
            text = path.read_text(encoding="utf-8")
            path.write_text(
                re.sub(r"@Disabled\([^)]*\)\s*\n", "", text),
                encoding="utf-8",
            )


def verify_edit(
    task: PolyglotTask,
    parsed: ParsedEdit,
    *,
    source_root: Path,
    image: str = "flashtrace-aider-benchmark",
    cache_root: Path = Path("data/external/.cache/aider"),
    timeout_seconds: int = 180,
) -> VerificationResult:
    """Apply an edit to a fresh task copy and run the official test command."""

    source_dir = source_root / task.relative_path
    apply_errors = list(parsed.errors)
    gradle_cache = cache_root.resolve() / "gradle"
    gradle_cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"{task.task_id}-") as temp_dir:
        # C++ Exercism derives source filenames from the exercise directory name.
        workdir = Path(temp_dir) / task.exercise
        shutil.copytree(source_dir, workdir)
        applied = []
        for relative, content in parsed.files.items():
            if relative not in task.solution_files:
                apply_errors.append(f"refused non-solution path: {relative}")
                continue
            destination = (workdir / relative).resolve()
            if workdir.resolve() not in destination.parents:
                apply_errors.append(f"refused path outside task directory: {relative}")
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(content, encoding="utf-8")
            applied.append(relative)

        if not applied:
            return VerificationResult(
                passed=False,
                exit_code=None,
                timed_out=False,
                infrastructure_error=None,
                output="No generated solution file could be applied.",
                command=_test_command(task.language),
                applied_files=(),
                apply_errors=tuple(apply_errors),
            )

        _prepare_java_tests(workdir, task.test_files)
        command = _test_command(task.language)
        container_workdir = f"/workspace/{task.exercise}"
        cidfile = Path(temp_dir) / "docker.cid"
        docker_command = [
            "docker",
            "run",
            "--rm",
            "--cidfile",
            str(cidfile),
            "--network",
            "none",
            "--memory",
            "12g",
            "--memory-swap",
            "12g",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "--env",
            "HOME=/tmp/home",
            "--env",
            "GRADLE_USER_HOME=/gradle-cache",
            "--env",
            "CARGO_HOME=/tmp/cargo",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,exec,nosuid,size=3g",
            "--mount",
            f"type=bind,source={gradle_cache},target=/gradle-cache",
            "-v",
            f"{workdir}:{container_workdir}:rw",
            "-w",
            container_workdir,
            image,
            *command,
        ]
        try:
            result = subprocess.run(
                docker_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout_seconds,
                encoding="utf-8",
                errors="replace",
            )
            output = result.stdout.replace(str(workdir), task.task_id)
            return VerificationResult(
                passed=result.returncode == 0,
                exit_code=result.returncode,
                timed_out=False,
                infrastructure_error=(
                    "docker_runtime"
                    if result.returncode == 125
                    else (
                        "test_command_unavailable"
                        if result.returncode in {126, 127}
                        else None
                    )
                ),
                output=output,
                command=command,
                applied_files=tuple(applied),
                apply_errors=tuple(apply_errors),
            )
        except subprocess.TimeoutExpired as exc:
            if cidfile.exists():
                container_id = cidfile.read_text(encoding="utf-8").strip()
                if container_id:
                    subprocess.run(
                        ["docker", "rm", "--force", container_id],
                        check=False,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=30,
                    )
            stdout = exc.stdout or ""
            if isinstance(stdout, bytes):
                stdout = stdout.decode("utf-8", errors="replace")
            return VerificationResult(
                passed=False,
                exit_code=None,
                timed_out=True,
                infrastructure_error=None,
                output=stdout + "\nTests timed out.",
                command=command,
                applied_files=tuple(applied),
                apply_errors=tuple(apply_errors),
            )


def test_feedback(
    result: VerificationResult,
    solution_files: Sequence[str],
    *,
    max_characters: int = 12_000,
) -> str:
    """Create deterministic second-turn feedback from actual execution."""

    output = result.output
    if len(output) > max_characters:
        head = output[: max_characters // 2]
        tail = output[-max_characters // 2 :]
        output = head + "\n...[deterministically truncated]...\n" + tail
    files = " ".join(solution_files)
    return (
        "The previous edit was applied and the official tests produced:\n\n"
        f"```\n{output.rstrip()}\n```\n\n"
        "The tests are authoritative. Do not modify them. "
        f"Fix the complete implementation in {files} and return whole-file listings "
        "in the required format."
    )

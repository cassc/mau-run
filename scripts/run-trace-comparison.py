#!/usr/bin/env python3
"""Compare execution traces produced by Mau and a Go-Ethereum based runner.

This script expands Ethereum state tests into single-transaction fixtures,
converts them into hex/PTX artifacts, runs both Mau and go-ethereum on each
fixture, and compares the resulting PC traces.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, cast

EXCLUDE_TESTS = [
    # Add any test patterns to exclude here
    "vmPerformance",
    "stQuadraticComplexityTest",
    "stTimeConsuming",
]

REPO_ROOT = Path(__file__).resolve().parent.parent

TRACE_LINE_RE = re.compile(
    r"^\[TRACE\]\s+pc=0x(?P<pc>[0-9a-fA-F]+)\s+opcode=0x(?P<opcode>[0-9a-fA-F]+)"
    r"(?:\s+size=(?P<depth>\d+)(?:\s+top=(?P<stack>0x[0-9a-fA-F]+))?)?"
)
TRACE_HEADER_RE = re.compile(r"^\[TRACE\]\s+kernel=(?P<kernel>\w+)\s+records=(?P<count>\d+)")


@dataclass
class ExpandedTestCase:
    """Represents one concrete test case expanded from a multi-index JSON."""

    case_id: str
    rootname: str
    indexes: tuple[int, int, int]
    source_json: Path
    expanded_json: Path
    variant_dir: Path
    hex_path: Path
    ptx_path: Path
    rel_input: Path


@dataclass
class TraceCapture:
    """Holds trace details for an executor run."""

    pcs: list[int] = field(default_factory=lambda: [])
    opcodes: list[int] = field(default_factory=lambda: [])
    first_stacks: list[str | None] = field(default_factory=lambda: [])
    stack_depths: list[int | None] = field(default_factory=lambda: [])
    kernel: str | None = None
    raw_stdout: str = ""
    raw_stderr: str = ""

    @property
    def pc_count(self) -> int:
        return len(self.pcs)


@dataclass
class CaseReport:
    case: ExpandedTestCase
    status: str
    detail: str
    mau: TraceCapture | None = None
    goevm: TraceCapture | None = None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Mau GPU execution traces against go-ethereum.")
    parser.add_argument(
        "--ethtest-dir",
        type=Path,
        required=True,
        help="Root directory that contains the GeneralStateTests JSON files.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        help="Directory to store expanded fixtures, artifacts, and logs. " "Defaults to a temporary directory.",
    )
    parser.add_argument(
        "--fork",
        default="Shanghai",
        help="Ethereum fork key to extract from each test (default: Shanghai).",
    )
    parser.add_argument(
        "--geth-bin",
        default=str(REPO_ROOT / "resources" / "goevm-glibc2.31-bin"),
        help="Path to the go-ethereum-compatible runner (default: resources/goevm-glibc2.31-bin).",
    )
    parser.add_argument(
        "--mau-bin",
        default=str(Path("target/release/state_test")),
        help="Path to the Mau state_test binary (default: target/release/state_test).",
    )
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        metavar="PATTERN",
        help="Only process tests whose identifier matches the glob pattern. " "May be supplied multiple times.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="PATTERN",
        help="Additional glob patterns to skip (applied after the built-in blacklist).",
    )
    parser.add_argument(
        "--keep-artifacts",
        action="store_true",
        help="Preserve the working directory instead of removing it on success.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    parser.add_argument(
        "--mau-kernel",
        default="main_contract",
        help="Name of the kernel section to extract from Mau traces (default: main_contract).",
    )
    parser.add_argument(
        "--mau-trim",
        type=int,
        default=0,
        help="Trim the first N PCs from Mau traces before comparison (default: 0).",
    )
    parser.add_argument(
        "--ptx-mcpu",
        default=None,
        help="Optional override passed as --mcpu when invoking hex-to-ptx.py.",
    )
    parser.add_argument(
        "--mau-timeout",
        type=int,
        default=30,
        help="Timeout in seconds for each Mau state_test invocation (default: 30).",
    )
    parser.add_argument(
        "--goevm-timeout",
        type=int,
        default=30,
        help="Timeout in seconds for each go-ethereum invocation (default: 30).",
    )
    parser.add_argument(
        "--hex-timeout",
        type=int,
        default=10,
        help="Timeout in seconds for json-to-hex.py (default: 10).",
    )
    parser.add_argument(
        "--ptx-timeout",
        type=int,
        default=10,
        help="Timeout in seconds for hex-to-ptx.py (default: 10).",
    )
    parser.add_argument(
        "--mau-docker-image",
        default=None,
        help="If set, run Mau inside this Docker image (e.g. augustus/mau-ityfuzz:latest).",
    )
    parser.add_argument(
        "--mau-docker-gpus",
        default="all",
        help="Value passed to `docker run --gpus` when using --mau-docker-image (default: all).",
    )
    parser.add_argument(
        "--add-docker-flag",
        action="store_true",
        help="Add --device /dev/nvidia0 --device /dev/nvidiactl --device /dev/nvidia-uvm to docker command.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        help="Optional path for dumping a JSON summary (defaults to work-dir/summary.json).",
    )
    return parser.parse_args(argv)


def debug(msg: str, *, args: argparse.Namespace) -> None:
    if args.verbose:
        print(f"DEBUG {msg}")


def ensure_binary_exists(binary: str) -> None:
    if shutil.which(binary):
        return
    candidate = Path(binary)
    if candidate.exists() and os.access(candidate, os.X_OK):
        return
    raise FileNotFoundError(f"Required binary not found or not executable: {binary}")


def normalize_stack_value(value: str | None) -> str | None:
    """Normalize a stack word to a 0x-prefixed, 32-byte lowercase hex string."""
    if value is None:
        return None
    token = value.strip()
    if token in {"-", "?"}:
        return None
    if token.startswith(("0x", "0X")):
        token = token[2:]
    token = token.lower()
    if not re.fullmatch(r"[0-9a-f]*", token):
        return None
    token = token.lstrip("0")
    if not token:
        token = "0"
    token = token.zfill(64)
    return "0x" + token[-64:]


def match_any(patterns: Iterable[str], target: str) -> bool:
    return any(fnmatch(target, pattern) for pattern in patterns)


def expand_ethtests(args: argparse.Namespace, work_dir: Path) -> list[ExpandedTestCase]:
    cases: list[ExpandedTestCase] = []
    input_root = args.ethtest_dir.resolve()
    if not input_root.exists():
        raise FileNotFoundError(f"Ethereum tests root not found: {input_root}")

    debug(f"Scanning {input_root} for JSON fixtures", args=args)

    for json_path in sorted(input_root.rglob("*.json")):
        rel_path = json_path.relative_to(input_root)
        rel_str = str(rel_path)

        if any(skip in rel_str for skip in EXCLUDE_TESTS):
            debug(f"Skipping (default blacklist) {rel_str}", args=args)
            continue
        if args.exclude and match_any(args.exclude, rel_str):
            debug(f"Skipping (user exclude) {rel_str}", args=args)
            continue

        with json_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)

        for rootname, body in data.items():
            case_identifier = f"{rel_str}:{rootname}"
            if args.include and not match_any(args.include, case_identifier):
                continue

            transaction = body.get("transaction")
            if not transaction:
                debug(f"Skipping {case_identifier} (missing transaction)", args=args)
                continue

            post = body.get("post", {}).get(args.fork)
            if not post:
                debug(
                    f"Skipping {case_identifier} (missing post for {args.fork})",
                    args=args,
                )
                continue

            tx_data: list[str] = transaction.get("data") or []
            tx_gas: list[str] = transaction.get("gasLimit") or []
            tx_value: list[str] = transaction.get("value") or []

            for entry in post:
                indexes: dict[str, int] = entry.get("indexes") or {}
                try:
                    data_index = int(indexes["data"])
                    gas_index = int(indexes["gas"])
                    value_index = int(indexes["value"])
                except (KeyError, ValueError, TypeError):
                    debug(f"Skipping {case_identifier} (invalid indexes)", args=args)
                    continue

                try:
                    selected_data = tx_data[data_index]
                    selected_gas = tx_gas[gas_index]
                    selected_value = tx_value[value_index]
                except IndexError:
                    debug(
                        f"Skipping {case_identifier} (index out of range {indexes})",
                        args=args,
                    )
                    continue

                variant_name = f"{rootname}-{data_index}-{gas_index}-{value_index}"
                variant_dir = work_dir / rel_path.parent / variant_name
                variant_dir.mkdir(parents=True, exist_ok=True)

                expanded = copy.deepcopy(body)
                expanded_post = copy.deepcopy(entry)
                expanded_post["indexes"] = {"data": 0, "gas": 0, "value": 0}
                expanded["post"] = {args.fork: [expanded_post]}

                expanded_tx = copy.deepcopy(transaction)
                expanded_tx["data"] = [selected_data]
                expanded_tx["gasLimit"] = [selected_gas]
                expanded_tx["value"] = [selected_value]
                expanded["transaction"] = expanded_tx

                doc = {rootname: expanded}
                expanded_json = variant_dir / json_path.name
                with expanded_json.open("w", encoding="utf-8") as fh:
                    json.dump(doc, fh, ensure_ascii=False, indent=2)

                hex_path = expanded_json.with_suffix(".hex")
                ptx_path = expanded_json.with_suffix(".ptx")

                case_id = f"{rel_path}:{variant_name}"
                cases.append(
                    ExpandedTestCase(
                        case_id=case_id,
                        rootname=rootname,
                        indexes=(data_index, gas_index, value_index),
                        source_json=json_path,
                        expanded_json=expanded_json,
                        variant_dir=variant_dir,
                        hex_path=hex_path,
                        ptx_path=ptx_path,
                        rel_input=rel_path,
                    )
                )
    debug(f"Expanded {len(cases)} test cases", args=args)
    return cases


def run_subprocess(
    cmd: Sequence[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    cmd_list = list(map(str, cmd))
    process = subprocess.Popen(
        cmd_list,
        cwd=str(cwd) if cwd else None,
        env=dict(env) if env else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        # Ensure the entire process group is terminated to avoid orphaned workers.
        try:
            if os.name != "nt":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except ProcessLookupError:
            pass
        finally:
            stdout, stderr = process.communicate()
        raise TimeoutError(f"Command '{' '.join(cmd_list)}' timed out after {timeout} seconds") from exc
    return subprocess.CompletedProcess(
        args=cmd_list,
        returncode=process.returncode,
        stdout=stdout,
        stderr=stderr,
    )


def build_hex(case: ExpandedTestCase, args: argparse.Namespace) -> None:
    if case.hex_path.exists():
        return
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent.parent / "json-to-hex.py"),
        str(case.expanded_json),
        "--test",
        case.rootname,
        "--output",
        str(case.hex_path),
    ]
    result = run_subprocess(cmd, cwd=REPO_ROOT, timeout=args.hex_timeout)
    case.variant_dir.joinpath("json-to-hex.stdout.txt").write_text(result.stdout)
    case.variant_dir.joinpath("json-to-hex.stderr.txt").write_text(result.stderr)
    if result.returncode != 0:
        snippet = (result.stderr or result.stdout).strip().splitlines()[-5:]
        detail = "\n".join(snippet)
        raise RuntimeError(f"json-to-hex.py failed ({result.returncode}) for {case.case_id}: {detail}")


def build_ptx(case: ExpandedTestCase, args: argparse.Namespace) -> None:
    if case.ptx_path.exists():
        return
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent.parent / "hex-to-ptx.py"),
        str(case.hex_path),
        "--output",
        str(case.ptx_path),
        "--cleanup",
    ]
    if args.ptx_mcpu:
        cmd.extend(["--mcpu", args.ptx_mcpu])
    env = os.environ.copy()
    env.setdefault("MAU_TRACE_PC", "1")
    resources_dir = REPO_ROOT / "resources"
    env["PATH"] = f"{resources_dir}:{env.get('PATH', '')}"
    result = run_subprocess(cmd, cwd=REPO_ROOT, env=env, timeout=args.ptx_timeout)
    case.variant_dir.joinpath("hex-to-ptx.stdout.txt").write_text(result.stdout)
    case.variant_dir.joinpath("hex-to-ptx.stderr.txt").write_text(result.stderr)
    if result.returncode != 0:
        snippet = (result.stderr or result.stdout).strip().splitlines()[-5:]
        detail = "\n".join(snippet)
        raise RuntimeError(f"hex-to-ptx.py failed ({result.returncode}) for {case.case_id}: {detail}")


def parse_mau_trace(stdout: str, *, preferred_kernel: str) -> dict[str, TraceCapture]:
    kernels: dict[str, TraceCapture] = {}
    current_kernel: str | None = None
    for line in stdout.splitlines():
        header_match = TRACE_HEADER_RE.match(line)
        if header_match:
            current_kernel = header_match.group("kernel")
            assert current_kernel is not None  # Type narrowing for setdefault
            kernels.setdefault(current_kernel, TraceCapture(kernel=current_kernel))
            continue
        if current_kernel is None:
            continue
        entry_match = TRACE_LINE_RE.match(line)
        if not entry_match:
            continue
        capture = kernels.setdefault(current_kernel, TraceCapture(kernel=current_kernel))
        capture.pcs.append(int(entry_match.group("pc"), 16))
        capture.opcodes.append(int(entry_match.group("opcode"), 16))
        depth_token = entry_match.group("depth")
        stack_token = entry_match.group("stack")
        if depth_token is not None:
            depth = int(depth_token)
            capture.stack_depths.append(depth)
            if depth > 0 and stack_token:
                capture.first_stacks.append(normalize_stack_value(stack_token))
            else:
                capture.first_stacks.append(None)
        else:
            capture.stack_depths.append(None)
            capture.first_stacks.append(None)
    if not kernels and stdout:
        # No trace lines found; capture entire stdout for debugging.
        kernels["_raw"] = TraceCapture(pcs=[], opcodes=[], kernel=None, raw_stdout=stdout)
    return kernels


def run_mau_in_docker(
    case: ExpandedTestCase, args: argparse.Namespace, env: Mapping[str, str]
) -> subprocess.CompletedProcess[str]:
    docker_cmd: list[str] = ["docker", "run", "--rm"]

    gpus = getattr(args, "mau_docker_gpus", None)
    if gpus:
        docker_cmd.extend(["--gpus", gpus])

    if getattr(args, "add_docker_flag", False):
        docker_cmd.extend(["--device", "/dev/nvidia0"])
        docker_cmd.extend(["--device", "/dev/nvidiactl"])
        docker_cmd.extend(["--device", "/dev/nvidia-uvm"])

    if os.name != "nt":
        try:
            uid = os.getuid()
            gid = os.getgid()
        except AttributeError:
            uid = gid = None
        if uid is not None and gid is not None:
            docker_cmd.extend(["--user", f"{uid}:{gid}"])

    mount_dirs: set[str] = {str(REPO_ROOT.resolve())}
    for path in (
        case.variant_dir,
        case.expanded_json.parent,
        case.hex_path.parent,
        case.ptx_path.parent,
    ):
        mount_dirs.add(str(path.resolve()))

    for mount in sorted(mount_dirs):
        docker_cmd.extend(["-v", f"{mount}:{mount}"])

    docker_cmd.extend(["-w", str(REPO_ROOT.resolve())])

    forwarded_env = {
        key: env[key]
        for key in (
            "MAU_TRACE_PC",
            "MAU_TRACE_STACK",
            "CUDA_VISIBLE_DEVICES",
            "RUST_LOG",
        )
        if key in env
    }
    for key, value in sorted(forwarded_env.items()):
        docker_cmd.extend(["-e", f"{key}={value}"])

    docker_cmd.append(args.mau_docker_image)

    docker_cmd.append(str(args.mau_bin))
    docker_cmd.extend(
        [
            str(case.expanded_json.resolve()),
            str(case.ptx_path.resolve()),
            str(case.hex_path.resolve()),
        ]
    )

    return run_subprocess(docker_cmd, timeout=args.mau_timeout)


def run_mau(case: ExpandedTestCase, args: argparse.Namespace) -> TraceCapture:
    env = os.environ.copy()
    env.setdefault("MAU_TRACE_PC", "1")
    if args.mau_docker_image:
        result = run_mau_in_docker(case, args, env)
    else:
        cmd: list[str] = [
            args.mau_bin,
            str(case.expanded_json),
            str(case.ptx_path),
            str(case.hex_path),
        ]
        result = run_subprocess(cmd, cwd=REPO_ROOT, env=env, timeout=args.mau_timeout)
    case.variant_dir.joinpath("mau.stdout.txt").write_text(result.stdout)
    case.variant_dir.joinpath("mau.stderr.txt").write_text(result.stderr)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError(f"Mau execution failed ({result.returncode}) for {case.case_id}")

    kernels = parse_mau_trace(result.stdout, preferred_kernel=args.mau_kernel)
    target_kernel = args.mau_kernel
    capture = kernels.get(target_kernel)
    if capture is None and kernels:
        # Fallback to the first available kernel.
        capture = next(iter(kernels.values()))
    if capture is None:
        capture = TraceCapture(pcs=[], opcodes=[], kernel=None)
    capture.raw_stdout = result.stdout
    capture.raw_stderr = result.stderr

    if args.mau_trim:
        capture.pcs = capture.pcs[args.mau_trim :]
        capture.opcodes = capture.opcodes[args.mau_trim :]
    return capture


def parse_goevm_trace(output: str) -> TraceCapture:
    capture = TraceCapture(kernel="goevm")
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "pc" in obj:
            obj_dict = cast(dict[str, Any], obj)
            pc_value = obj_dict["pc"]
            pc: int
            if isinstance(pc_value, str):
                pc = int(pc_value, 16) if pc_value.startswith("0x") else int(pc_value, 10)
            elif isinstance(pc_value, int):
                pc = pc_value
            else:
                pc = int(pc_value)
            capture.pcs.append(pc)
            opcode_value = obj_dict.get("op")
            if isinstance(opcode_value, str):
                # 'op' may be an opcode name; skip if not numeric.
                try:
                    capture.opcodes.append(int(opcode_value, 16))
                except ValueError:
                    capture.opcodes.append(-1)
            elif isinstance(opcode_value, int):
                capture.opcodes.append(opcode_value)
            else:
                capture.opcodes.append(-1)
            stack_list_raw = obj_dict.get("stack")
            if isinstance(stack_list_raw, list):
                stack_list = cast(list[Any], stack_list_raw)
                capture.stack_depths.append(len(stack_list))
                if stack_list:
                    top_val: Any = stack_list[-1]
                    if isinstance(top_val, str):
                        capture.first_stacks.append(normalize_stack_value(top_val))
                    else:
                        capture.first_stacks.append(None)
                else:
                    capture.first_stacks.append(None)
            else:
                capture.stack_depths.append(None)
                capture.first_stacks.append(None)
    capture.raw_stdout = output
    capture.raw_stderr = ""
    return capture


def run_goevm(case: ExpandedTestCase, args: argparse.Namespace) -> TraceCapture:
    cmd: list[str] = [
        str(args.geth_bin),
        "--json",
        "--noreturndata",
        "--nomemory",
        "statetest",
        str(case.expanded_json),
    ]
    result = run_subprocess(cmd, cwd=REPO_ROOT, timeout=args.goevm_timeout)
    case.variant_dir.joinpath("goevm.stdout.txt").write_text(result.stdout)
    case.variant_dir.joinpath("goevm.stderr.txt").write_text(result.stderr)
    # go-ethereum writes traces to stderr
    output = result.stderr or result.stdout
    if result.returncode != 0:
        raise RuntimeError(f"go-ethereum execution failed ({result.returncode}) for {case.case_id}")
    capture = parse_goevm_trace(output)
    capture.raw_stdout = result.stdout
    capture.raw_stderr = result.stderr
    return capture


def compare_traces(case: ExpandedTestCase, mau: TraceCapture, goevm: TraceCapture) -> CaseReport:
    if not mau.pcs:
        return CaseReport(
            case=case,
            status="mau-trace-missing",
            detail="No Mau trace captured",
            mau=mau,
            goevm=goevm,
        )
    if not goevm.pcs:
        return CaseReport(
            case=case,
            status="goevm-trace-missing",
            detail="No go-ethereum trace captured",
            mau=mau,
            goevm=goevm,
        )

    if mau.pc_count != goevm.pc_count:
        detail = f"PC count mismatch (mau={mau.pc_count}, goevm={goevm.pc_count})"
        return CaseReport(case=case, status="pc-mismatch", detail=detail, mau=mau, goevm=goevm)

    for idx, (m_pc, g_pc) in enumerate(zip(mau.pcs, goevm.pcs)):
        if m_pc != g_pc:
            detail = f"Trace mismatch at step {idx}: mau=0x{m_pc:x}, goevm=0x{g_pc:x}"
            return CaseReport(case=case, status="trace-mismatch", detail=detail, mau=mau, goevm=goevm)

    # PCs match, now check stacks if available
    stack_lengths_ok = (
        len(mau.first_stacks) == mau.pc_count
        and len(goevm.first_stacks) == goevm.pc_count
        and len(mau.stack_depths) == mau.pc_count
        and len(goevm.stack_depths) == goevm.pc_count
    )

    if stack_lengths_ok:
        for idx in range(mau.pc_count):
            m_depth = mau.stack_depths[idx]
            g_depth = goevm.stack_depths[idx]
            # Skip stack comparison when either side has size=0
            if m_depth == 0 or g_depth == 0:
                continue
            m_val = mau.first_stacks[idx]
            g_val = goevm.first_stacks[idx]
            if m_val is None or g_val is None:
                continue
            if m_val != g_val:
                detail = f"Top-of-stack mismatch at step {idx}: mau={m_val}, goevm={g_val}"
                return CaseReport(
                    case=case,
                    status="stack-mismatch",
                    detail=detail,
                    mau=mau,
                    goevm=goevm,
                )
        detail = f"PCs and stack values match (pcs={mau.pc_count})"
        return CaseReport(case=case, status="match-full", detail=detail, mau=mau, goevm=goevm)

    detail = f"PCs match (pcs={mau.pc_count})"
    return CaseReport(case=case, status="match-pc", detail=detail, mau=mau, goevm=goevm)


def write_summary(work_dir: Path, reports: list[CaseReport], summary_path: Path | None) -> None:
    summary: dict[str, Any] = {
        "stats": {
            "total": len(reports),
            "match_full": sum(1 for r in reports if r.status == "match-full"),
            "match_pc": sum(1 for r in reports if r.status == "match-pc"),
            "pc_mismatch": sum(1 for r in reports if r.status == "pc-mismatch"),
            "trace_mismatch": sum(1 for r in reports if r.status == "trace-mismatch"),
            "stack_mismatch": sum(1 for r in reports if r.status == "stack-mismatch"),
            "stack_depth_mismatch": sum(1 for r in reports if r.status == "stack-depth-mismatch"),
            "mau_trace_missing": sum(1 for r in reports if r.status == "mau-trace-missing"),
            "goevm_trace_missing": sum(1 for r in reports if r.status == "goevm-trace-missing"),
            "failures": sum(
                1
                for r in reports
                if r.status
                not in {
                    "match-full",
                    "match-pc",
                    "pc-mismatch",
                    "trace-mismatch",
                    "stack-mismatch",
                    "mau-trace-missing",
                    "goevm-trace-missing",
                }
            ),
        },
        "cases": [
            {
                "id": report.case.case_id,
                "status": report.status,
                "detail": report.detail,
                "mau_pc_count": report.mau.pc_count if report.mau else None,
                "goevm_pc_count": report.goevm.pc_count if report.goevm else None,
            }
            for report in reports
        ],
    }
    target = summary_path or (work_dir / "summary.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(summary, indent=2))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    if args.mau_docker_image:
        try:
            ensure_binary_exists("docker")
        except FileNotFoundError as exc:
            print(exc, file=sys.stderr)
            return 1
    else:
        try:
            ensure_binary_exists(args.mau_bin)
        except FileNotFoundError as exc:
            print(exc, file=sys.stderr)
            return 1

    try:
        ensure_binary_exists(args.geth_bin)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    if args.work_dir:
        work_dir = args.work_dir.resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
        created_tmp = False
    else:
        work_dir = Path(tempfile.mkdtemp(prefix="trace-comparison-", dir=os.getcwd()))
        created_tmp = True

    args.work_dir = work_dir

    print(f"Artifacts will be written to {work_dir}")

    try:
        cases = expand_ethtests(args, work_dir)
        if not cases:
            print("No test cases found after applying filters.")
            return 0

        reports: list[CaseReport] = []
        for idx, case in enumerate(cases, start=1):
            print(f"[{idx}/{len(cases)}] {case.case_id}")
            try:
                build_hex(case, args)
                build_ptx(case, args)
                mau_trace = run_mau(case, args)
                goevm_trace = run_goevm(case, args)
                report = compare_traces(case, mau_trace, goevm_trace)
            except Exception as exc:  # noqa: BLE001 - we need to flag the failure
                detail = f"Execution failed: {exc}"
                report = CaseReport(case=case, status="error", detail=detail)
                print(f"  ERROR: {detail}")
                reports.append(report)
                continue

            status_label = report.status.replace("-", " ").title()
            detail = report.detail
            print(f"  {status_label}: {detail}")
            reports.append(report)

        write_summary(work_dir, reports, args.summary_json)

        total_cases = len(reports)
        stats = {
            "match_full": sum(1 for r in reports if r.status == "match-full"),
            "match_pc": sum(1 for r in reports if r.status == "match-pc"),
            "pc_mismatch": sum(1 for r in reports if r.status == "pc-mismatch"),
            "trace_mismatch": sum(1 for r in reports if r.status == "trace-mismatch"),
            "stack_mismatch": sum(1 for r in reports if r.status == "stack-mismatch"),
            "mau_missing": sum(1 for r in reports if r.status == "mau-trace-missing"),
            "goevm_missing": sum(1 for r in reports if r.status == "goevm-trace-missing"),
            "error": sum(1 for r in reports if r.status == "error"),
        }
        print("Summary:")
        print(
            f"  Total cases: {total_cases}, "
            f"PC+Stack matches: {stats['match_full']}, "
            f"PC-only matches: {stats['match_pc']}, "
            f"PC mismatches: {stats['pc_mismatch']}, "
            f"Trace mismatches: {stats['trace_mismatch']}, "
            f"Stack mismatches: {stats['stack_mismatch']}, "
            f"Mau missing: {stats['mau_missing']}, "
            f"go-ethereum missing: {stats['goevm_missing']}, "
            f"Errors: {stats['error']}"
        )

        exit_code = 0 if (stats["match_full"] + stats["match_pc"]) == len(cases) else 1
        return exit_code
    finally:
        if created_tmp and not args.keep_artifacts:
            shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())

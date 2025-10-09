import argparse
import re
import shutil
import subprocess
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile contract deployment hex into PTX using ptxsema and LLVM tools."
    )
    parser.add_argument(
        "hex_path",
        type=Path,
        help="Path to the deployment bytecode in hex format.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Destination for the generated .ptx file (defaults to HEX path with .ptx suffix).",
    )
    parser.add_argument(
        "--ptxsema",
        type=Path,
        help="Path to the ptxsema binary (auto-detects ./ptxsema or ./resources/ptxsema).",
    )
    parser.add_argument(
        "--rt",
        type=Path,
        default=Path("resources/rt.o.bc"),
        help="Runtime bitcode to link against (default: resources/rt.o.bc).",
    )
    parser.add_argument(
        "--llc",
        type=Path,
        help="Path to the llc-16 binary (defaults to llc-16 in $PATH).",
    )
    parser.add_argument(
        "--mcpu",
        default="sm_86",
        help="Target GPU architecture passed to llc-16 (default: sm_86).",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Remove the generated .ll and -kernel.bc files after the PTX is produced.",
    )
    return parser.parse_args()


def resolve_tool(user_value: Path | None, candidates: Iterable[Path | None], tool_name: str) -> Path:
    if user_value:
        if user_value.exists():
            return user_value
        raise FileNotFoundError(f"{tool_name} not found at {user_value}")
    for candidate in candidates:
        if candidate is None:
            continue
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Unable to locate {tool_name}. Use --{tool_name.replace('-', '_')} to specify it.")


def run(cmd: list[str]) -> None:
    print(" ".join(str(part) for part in cmd))
    subprocess.run(cmd, check=True)


def enforce_kernel_entries(ptx_path: Path) -> None:
    text = ptx_path.read_text()
    patched_any = False
    for name in ("main_contract", "updateBits", "main_deployer"):
        pattern = re.compile(rf"(\.visible\s+)\.func(\s+{name}\s*\()", re.MULTILINE)
        if pattern.search(text):
            text, count = pattern.subn(r"\1.entry\2", text, count=1)
            if count:
                print(f"Patched {name} definition to .entry")
                patched_any = True
    if patched_any:
        ptx_path.write_text(text)
    else:
        print("No kernel patch applied (entries already exported?)")


def main() -> None:
    args = parse_args()
    hex_path = args.hex_path.resolve()
    if not hex_path.exists():
        raise FileNotFoundError(f"Hex file not found: {hex_path}")

    ptx_path = (args.output or hex_path.with_suffix(".ptx")).resolve()
    ll_path = ptx_path.with_suffix(".ll")
    kernel_bc_path = ptx_path.with_name(f"{ptx_path.stem}-kernel.bc")

    ptxsema_path = resolve_tool(
        args.ptxsema,
        (
            Path("./ptxsema"),
            Path("./resources/ptxsema"),
            Path(shutil.which("ptxsema")) if shutil.which("ptxsema") else None,
        ),
        "ptxsema",
    )
    llc_path = resolve_tool(
        args.llc,
        (
            Path("llc-16") if Path("llc-16").exists() else None,
            Path("./resources/llc-16") if Path("./resources/llc-16").exists() else None,
            Path(shutil.which("llc-16")) if shutil.which("llc-16") else None,
            Path(shutil.which("llc")) if shutil.which("llc") else None,
        ),
        "llc",
    )

    if llc_path.name != "llc-16" and not llc_path.name.startswith("llc"):
        raise ValueError(f"Expected llc-like binary, got {llc_path}")

    for path in (ll_path, kernel_bc_path, ptx_path):
        path.unlink(missing_ok=True)

    run([str(ptxsema_path), str(hex_path), "-o", str(ll_path), "--hex", "--dump"])

    run(["llvm-link", str(ll_path), str(args.rt.resolve()), "-o", str(kernel_bc_path)])

    run([str(llc_path), f"-mcpu={args.mcpu}", str(kernel_bc_path), "-o", str(ptx_path)])

    enforce_kernel_entries(ptx_path)

    if args.cleanup:
        ll_path.unlink(missing_ok=True)
        kernel_bc_path.unlink(missing_ok=True)

    print(f"Wrote {ptx_path}")


if __name__ == "__main__":
    main()

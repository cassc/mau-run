from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, cast

JSONDict = Dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Ethereum statetest JSON into deployable creation bytecode."
    )
    parser.add_argument(
        "json_path",
        type=Path,
        help="Path to the statetest JSON file (e.g. resources/expPower2.json).",
    )
    parser.add_argument(
        "--test",
        dest="test_name",
        help="Name of the test entry inside the JSON file (defaults to the only entry).",
    )
    parser.add_argument(
        "--account",
        metavar="ADDRESS",
        help="Hex account address from the pre-state to use (defaults to the transaction 'to' field).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path for the output .hex file (defaults to input path with .hex extension).",
    )
    return parser.parse_args()


def push(value: int) -> bytes:
    if value < 0:
        raise ValueError("value must be non-negative")
    byte_len = max(1, (value.bit_length() + 7) // 8)
    if byte_len > 32:
        raise ValueError("value too large for PUSH (max 32 bytes)")
    return bytes([0x5F + byte_len]) + value.to_bytes(byte_len, "big")


def select_test(data: JSONDict, test_name: Optional[str]) -> Tuple[str, JSONDict]:
    if test_name:
        try:
            candidate = data[test_name]
        except KeyError as exc:
            raise KeyError(f"Test '{test_name}' not found in JSON file.") from exc
        if not isinstance(candidate, dict):
            raise ValueError(f"Test '{test_name}' is not a JSON object.")
        return test_name, cast(JSONDict, candidate)
    if len(data) == 1:
        only_key, only_value = next(iter(data.items()))
        if not isinstance(only_value, dict):
            raise ValueError(f"Test '{only_key}' is not a JSON object.")
        return only_key, cast(JSONDict, only_value)
    available = ", ".join(sorted(data.keys()))
    raise ValueError(f"Multiple tests found ({available}); specify one with --test.")


def normalize_hex(value: str) -> str:
    value = value.lower()
    return value if value.startswith("0x") else f"0x{value}"


def resolve_account(doc: JSONDict, account_arg: Optional[str]) -> Tuple[str, JSONDict]:
    pre_state: Dict[str, JSONDict] = doc.get("pre", {})

    if account_arg:
        target = normalize_hex(account_arg)
    else:
        tx_obj: Optional[Dict[str, Any]] = doc.get("transaction")
        tx_to: Optional[str] = None
        if isinstance(tx_obj, dict):
            candidate = tx_obj.get("to")
            if isinstance(candidate, str):
                tx_to = candidate
        if not tx_to:
            raise ValueError("No account provided and transaction 'to' field missing; use --account.")
        target = normalize_hex(tx_to)

    normalized_pre: Dict[str, JSONDict] = {
        normalize_hex(addr): data for addr, data in pre_state.items()
    }
    account_doc = normalized_pre.get(target)
    if account_doc is None:
        available = ", ".join(sorted(normalized_pre.keys()))
        raise KeyError(f"Account {target} not found in pre-state. Available accounts: {available}")
    return target, account_doc


def load_runtime(account_doc: JSONDict) -> bytes:
    code_hex = account_doc.get("code", "")
    if not isinstance(code_hex, str):
        raise ValueError("Invalid JSON structure: expected 'code' to be a hex string.")
    code_hex = code_hex[2:] if code_hex.startswith("0x") else code_hex
    if code_hex == "":
        return b""
    return bytes.fromhex(code_hex)


def build_creation(runtime: bytes) -> bytes:
    size_push = push(len(runtime))
    offset = 0
    while True:
        offset_push = push(offset)
        header = bytearray()
        header += size_push          # length
        header += offset_push        # offset to runtime
        header += b"\x60\x00"        # dest = 0
        header += b"\x39"            # CODECOPY
        header += size_push
        header += b"\x60\x00"
        header += b"\xf3"            # RETURN
        new_offset = len(header)
        if new_offset == offset:
            break
        offset = new_offset
    return bytes(header) + runtime


def main() -> None:
    args = parse_args()
    raw_data = json.loads(args.json_path.read_text())
    if not isinstance(raw_data, dict):
        raise ValueError("Top-level JSON structure must be an object.")
    data = cast(JSONDict, raw_data)
    test_name, doc = select_test(data, args.test_name)
    account_address, account_doc = resolve_account(doc, args.account)
    runtime = load_runtime(account_doc)
    creation = build_creation(runtime)

    out_path = args.output or args.json_path.with_suffix(".hex")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(creation.hex())

    print(
        f"Wrote {out_path} ({len(creation)} bytes) "
        f"for test '{test_name}' account {account_address}"
    )


if __name__ == "__main__":
    main()

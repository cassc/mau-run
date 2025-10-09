import json
from pathlib import Path

json_path = Path("resources/expPower2.json")
out_path = Path("resources/expPower2.hex")

doc = json.loads(json_path.read_text())["expPower2"]
runtime = bytes.fromhex(doc["pre"]["0xcccccccccccccccccccccccccccccccccccccccc"]["code"][2:])

def push(value: int) -> bytes:
    if value <= 0xFF:
        return bytes([0x60, value])
    if value <= 0xFFFF:
        return bytes([0x61]) + value.to_bytes(2, "big")
    if value <= 0xFFFFFF:
        return bytes([0x62]) + value.to_bytes(3, "big")
    raise ValueError("value too large for PUSH")

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

creation = bytes(header) + runtime
out_path.write_text(creation.hex())
print(f"Wrote {out_path} ({len(creation)} bytes)")


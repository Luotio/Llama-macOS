#!/usr/bin/env python3
"""Print key architecture metadata from a GGUF file (stdlib only)."""
import struct
import sys

# gguf_type: 0 u8, 1 i8, 2 u16, 3 i16, 4 u32, 5 i32, 6 f32, 7 bool,
#            8 string, 9 array, 10 u64, 11 i64, 12 f64
SCALAR = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?",
          10: "Q", 11: "q", 12: "d"}


def rd(f):
    magic = f.read(4)
    assert magic == b"GGUF", "not a GGUF file"
    version = struct.unpack("<I", f.read(4))[0]
    n_tensors, kv_count = struct.unpack("<QQ", f.read(16))
    meta = {}
    for _ in range(kv_count):
        klen = struct.unpack("<Q", f.read(8))[0]
        key = f.read(klen).decode()
        t = struct.unpack("<I", f.read(4))[0]
        if t == 8:  # string
            vlen = struct.unpack("<Q", f.read(8))[0]
            meta[key] = f.read(vlen).decode()
        elif t in SCALAR:
            fmt = SCALAR[t]
            meta[key] = struct.unpack("<" + fmt, f.read(struct.calcsize(fmt)))[0]
        elif t == 9:  # array
            at = struct.unpack("<I", f.read(4))[0]
            n = struct.unpack("<Q", f.read(8))[0]
            if at == 8:
                vals = []
                for _ in range(n):
                    vlen = struct.unpack("<Q", f.read(8))[0]
                    vals.append(f.read(vlen).decode())
                meta[key] = vals
            elif at in SCALAR:
                fmt = SCALAR[at]
                meta[key] = list(struct.unpack("<" + fmt * n,
                                               f.read(struct.calcsize(fmt) * n)))
            else:
                raise ValueError(f"unsupported array type {at}")
        else:
            raise ValueError(f"unsupported value type {t} for {key}")
    return version, n_tensors, meta


def main():
    path = sys.argv[1]
    with open(path, "rb") as f:
        version, n_tensors, m = rd(f)
    if len(sys.argv) > 2:  # dump a specific key verbatim
        print(m.get(sys.argv[2], "<key not found>"))
        return
    print(f"gguf version {version}, tensors: {n_tensors}")
    for k in sorted(m):
        if any(s in k for s in ("arch", "layer", "head", "ctx", "embedding",
                                "block", "ffn", "experts", "version")):
            print(f"  {k} = {m[k]}")


if __name__ == "__main__":
    main()

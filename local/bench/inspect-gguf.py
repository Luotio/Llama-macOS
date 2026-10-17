#!/usr/bin/env python3
# inspect GGUF header(s): arch, key metadata, tensor table summary
import re
import struct
import sys
from math import prod

# ggml_type -> bytes per element (block quant types use the packed ratio)
# enum values from ggml/include/ggml.h of this build
# values verified against this model's offset gaps (see type-check.py)
TYPE_BYTES = {0: 4, 1: 2, 2: 0.5625, 3: 0.625, 6: 0.75, 7: 0.8125,
              8: 1.0625, 9: 1.125, 10: 0.5625, 11: 0.6875, 12: 0.75,
              13: 0.6875, 14: 0.8203125, 15: 1.0625, 16: 0.34375,
              17: 0.4375, 18: 0.5625, 19: 0.34375, 20: 0.5625,
              21: 0.4296875, 22: 0.4375, 23: 0.53125, 24: 1, 25: 2,
              26: 4, 27: 8, 28: 8, 29: 0.4375, 30: 2, 34: 0.25,
              35: 0.375, 41: 0.25, 42: 0.375}

class R:
    def __init__(self, data):
        self.data = data
        self.pos = 0
    def rd(self, fmt):
        size = struct.calcsize(fmt)
        v = struct.unpack_from("<" + fmt, self.data, self.pos)
        self.pos += size
        return v[0] if len(v) == 1 else v
    def rs(self):
        n = self.rd("Q")
        s = self.data[self.pos:self.pos+n].decode("utf-8", "replace")
        self.pos += n
        return s
    def rv(self, t):
        if t == 0:   return self.rd("B")
        if t == 1:   return self.rd("b")
        if t == 2:   return self.rd("H")
        if t == 3:   return self.rd("h")
        if t == 4:   return self.rd("I")
        if t == 5:   return self.rd("i")
        if t == 6:   return self.rd("f")
        if t == 7:   return self.rd("?")
        if t == 8:   return self.rs()
        if t == 9:
            et = self.rd("I"); n = self.rd("Q")
            return [self.rv(et) for _ in range(n)]
        if t == 10:  return self.rd("Q")
        if t == 11:  return self.rd("q")
        if t == 12:  return self.rd("d")
        raise ValueError(f"bad kv type {t}")

def parse(path):
    # only the header (KV + tensor table) is needed; cap the read so multi-GB
    # shards do not have to be pulled into RAM
    import os
    size = min(os.path.getsize(path), 64 * 1024 * 1024)
    with open(path, "rb") as f:
        data = f.read(size)
    r = R(data)
    magic = data[:4]
    assert magic == b"GGUF", f"bad magic {magic}"
    r.pos = 4
    version = r.rd("I")
    n_tensors = r.rd("Q")
    n_kv = r.rd("Q")
    kv = {}
    for _ in range(n_kv):
        k = r.rs()
        t = r.rd("I")
        v = r.rv(t)
        if isinstance(v, list) and len(v) > 8:
            v = f"<array[{len(v)}]>"
        kv[k] = v
    names = []
    sum_bytes = 0
    last_end = 0
    overflow = 0
    import os
    fsize = os.path.getsize(path)
    for _ in range(n_tensors):
        name = r.rs()
        ndim = r.rd("I")
        dims = [r.rd("Q") for _ in range(ndim)]
        ty = r.rd("I")
        off = r.rd("Q")
        names.append((name, dims, ty))
        b = prod(dims) * TYPE_BYTES.get(ty, 0)
        sum_bytes += b
        last_end = max(last_end, off + b)
        if off + b > fsize:
            overflow += 1
    print(f"  [check] file={fsize/1e9:.2f} GB sum={sum_bytes/1e9:.2f} GB "
          f"last_end={last_end/1e9:.2f} GB overflow={overflow}/{n_tensors}")
    return version, kv, names

all_names = []
for path in sys.argv[1:]:
    version, kv, names = parse(path)
    print(f"== {path} version={version} n_tensors={len(names)}")
    for k in sorted(kv):
        if any(s in k for s in ("architecture", "ple", "ngram", "nextn", "mtp",
                                "block_count", "head_count", "context_length",
                                "embedding_length", "expert")):
            print(f"  {k} = {kv[k]}")
    all_names.extend(names)

print(f"\ntotal tensors: {len(all_names)}")
pats = {}
for n, d, t in all_names:
    p = re.sub(r"\.\d+\.", ".N.", n)
    pats.setdefault(p, []).append((n, d, t))
tot = 0
for p in sorted(pats):
    cnt = len(pats[p])
    ns, ds, ts = pats[p][0]
    b = int(prod(ds) * TYPE_BYTES[ts]) if ts in TYPE_BYTES else -1
    tot += cnt * max(b, 0)
    print(f"{cnt:4d}  {p}  dims={ds} type={ts} ~{b/1e9:7.2f} GB")
print(f"total weights: ~{tot/1e9:.1f} GB")

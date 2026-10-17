#!/usr/bin/env python3
# per-tensor size from offset gaps -> implied bytes/element per ggml_type
import os
import struct
import sys
from math import prod

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
        if t == 0:
            return self.rd("B")
        if t == 1:
            return self.rd("b")
        if t == 2:
            return self.rd("H")
        if t == 3:
            return self.rd("h")
        if t == 4:
            return self.rd("I")
        if t == 5:
            return self.rd("i")
        if t == 6:
            return self.rd("f")
        if t == 7:
            return self.rd("?")
        if t == 8:
            return self.rs()
        if t == 9:
            et = self.rd("I")
            n = self.rd("Q")
            return [self.rv(et) for _ in range(n)]
        if t == 10:
            return self.rd("Q")
        if t == 11:
            return self.rd("q")
        if t == 12:
            return self.rd("d")
        raise ValueError(f"kv type {t}")

for path in sys.argv[1:]:
    fsize = os.path.getsize(path)
    with open(path, "rb") as f:
        head = f.read(min(fsize, 64 * 1024 * 1024))
    r = R(head)
    assert struct.unpack_from("<4s", head, 0)[0] == b"GGUF"
    r.pos = 4
    r.rd("I")
    n_tensors = r.rd("Q")
    n_kv = r.rd("Q")
    for _ in range(n_kv):
        r.rs()
        r.rv(r.rd("I"))
    tensors = []
    for _ in range(n_tensors):
        name = r.rs()
        ndim = r.rd("I")
        dims = [r.rd("Q") for _ in range(ndim)]
        ty = r.rd("I")
        off = r.rd("Q")
        tensors.append((name, dims, ty, off))
    tensors.sort(key=lambda t: t[3])
    print(f"== {path.split('/')[-1]} size={fsize:,}")
    # gap to next tensor is an upper bound on the true size; the minimum gap
    # within a type group is the tightest estimate of bytes/element
    by_type = {}
    for i, (name, dims, ty, off) in enumerate(tensors):
        end = tensors[i+1][3] if i+1 < len(tensors) else fsize
        e = prod(dims)
        if e == 0:
            continue
        bpe = (end - off) / e
        by_type.setdefault(ty, []).append((bpe, name, dims))
    for ty in sorted(by_type):
        items = sorted(by_type[ty])
        bpe, name, dims = items[0]
        print(f"  type={ty} n={len(items)} min_gap_bpe={bpe:.6f} "
              f"({bpe*8:.4f} bpc) e.g. {name} dims={dims}")

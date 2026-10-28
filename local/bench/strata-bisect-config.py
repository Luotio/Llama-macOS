#!/usr/bin/env python3
"""Generate local/strata/strata-bisect.json from the base config with a chosen STRATA_ env set.

Usage: python3 local/bench/strata-bisect-config.py [NEW6 var names...]
The three pre-Oct-27 vars (STRATA_HC_Q8, STRATA_ROUTE_TAIL_SKIP, STRATA_SM70_TABLE) are always on.
Pass any subset of the six vars added by commit ec8181f673 to test them:
  STRATA_BF16_TC STRATA_PROMPT_ATTN_OLD STRATA_ATTN_PRE75 STRATA_EXP_MODE STRATA_MMVQ_IL STRATA_GR_FAST
No args = baseline (the pre-Oct-27 env set). Writes strata-bisect.json + uses strata-bisect.log.
"""
import json, sys

BASE = "/mnt/git/local-llama-cpp/local/strata/strata-unsloth-ud-iq4_xs.json"
OUT = "/mnt/git/local-llama-cpp/local/strata/strata-bisect.json"
OLD = {"STRATA_HC_Q8": "1", "STRATA_ROUTE_TAIL_SKIP": "0", "STRATA_SM70_TABLE": "1"}
NEW6 = ["STRATA_BF16_TC", "STRATA_PROMPT_ATTN_OLD", "STRATA_ATTN_PRE75",
        "STRATA_EXP_MODE", "STRATA_MMVQ_IL", "STRATA_GR_FAST"]

cfg = json.load(open(BASE))
envs = sys.argv[1:]
for v in envs:
    assert v in NEW6, f"unknown var {v}; choose from {NEW6}"
cfg["env"] = dict(OLD)
for v in envs:
    cfg["env"][v] = "1"
cfg["log"] = "/mnt/git/local-llama-cpp/local/strata/strata-bisect.log"
with open(OUT, "w") as f:
    json.dump(cfg, f, indent=1)
    f.write("\n")
print("bisect env:", ", ".join(sorted(cfg["env"])))

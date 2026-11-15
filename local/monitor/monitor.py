#!/usr/bin/env python3
"""llama.cpp + GPU + host monitor with in-memory sparklines and a Nord Pool
spot-price GPU power throttle.

Polls, every MON_INTERVAL seconds:
  * llama-server /metrics endpoint(s)  (urls, one or more)
  * nvidia-smi                         (one call per cycle, all NVIDIA GPUs)
  * AMD GPUs via sysfs hwmon (/sys/class/drm/card*/device), augmented with
    amd-smi JSON output when the ROCm SMIA tool is installed
  * PCIe AER correctable-error counters from sysfs (GPU side AND the
    upstream root port, so both directions of the link are covered)
  * /proc: CPU, RAM, swap, load, network, uptime

Every price.fetch_interval seconds it makes sure the in-memory price table
covers now .. now+24h for the configured zone (FI by default), fetching the
missing days from free mirrors of the Nord Pool / SDAC day-ahead clearing
prices (Spotwatts primary, Volton fallback - Nord Pool's own Market Data API
needs paid credentials; either mirror serves the identical auction results).
The dashboard shows a spot-price chart with the threshold line.

When throttle.enabled, every throttle.interval seconds the price of the
CURRENT 15-minute market time unit (MTU) is compared against
threshold_c_per_kwh: above -> each configured GPU gets its minimum power
limit, at/below -> it is restored to restore_watts ("default" = the driver's
factory default). Caps are rewritten only when they differ from the target,
so the loop is quiet most cycles and also repairs caps changed elsewhere.
Setting caps needs root: run the service as root, or give its user NOPASSWD
rights for nvidia-smi -pl and tee (see local/systemd/llama-mon.sudoers).

A fixed window of samples is kept in memory only; nothing is written to
disk (except the service log). The app serves one dashboard page whose
canvas sparklines are redrawn by the browser each poll cycle.

Configuration precedence: built-in defaults < JSON config file < MON_*
environment variables (so the OpenRC conf.d keeps working). Config file
search: --config PATH, $MON_CONFIG, /etc/llama-mon/config.json. The repo
ships local/monitor/config.example.json; live installs keep theirs in
/etc/llama-mon/config.json (outside the repo). See README.md there for the
full schema (gpus visibility filter, price sources, throttle devices).

Environment overrides (defaults in parentheses):
  MON_HOST     bind address                       (127.0.0.1)
  MON_PORT     HTTP port                          (8085)
  MON_URLS     llama-server base URL(s), comma or
               space separated                    (http://127.0.0.1:8080)
  MON_INTERVAL poll interval, seconds             (2)
  MON_WINDOW   history to keep, seconds           (1800)
  MON_GPUS     GPU identifiers to show, comma/space separated NVIDIA UUIDs
               or amd:cardN / amd:BDF ids         (all)

Python 3.9+, standard library only.
"""

import datetime as dt
import json
import os
import re
import subprocess
import threading
import time
import urllib.request
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

POINTS = 300  # sparkline points handed to the browser (10 min at 2 s)

# ---------------------------------------------------------------- inputs

def http_get(url, timeout):
    req = urllib.request.Request(url, headers={'User-Agent': 'llama-mon/1.0'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode('utf-8', 'replace')


def parse_prom(text):
    """Flat Prometheus text format -> {metric: float}. Labeled series ignored."""
    out = {}
    for line in text.splitlines():
        if not line or line[0] == '#' or '{' in line:
            continue
        name, _, val = line.rpartition(' ')
        try:
            out[name] = float(val)
        except ValueError:
            pass
    return out


class Backend:
    """One inference endpoint. The backend kind (llama-server emitting
    Prometheus text, vs Strata emitting JSON) is re-detected on EVERY poll
    from the response body, so an endpoint that swaps backends across a
    restart is handled without restarting the monitor."""

    def __init__(self, base, interval=2.0):
        self.base = base.rstrip('/')
        self.label = urlsplit(self.base).netloc or self.base
        self.interval = max(0.5, float(interval))
        self.up = False
        self.kind = None            # 'llama' | 'strata' | None (not yet probed)
        self.model = None
        self.model_ts = 0.0
        # llama.cpp Prometheus baseline (cumulative *_total counters)
        self.l_prev = None
        self.l_prev_wall = None
        # Strata baseline (cumulative totals + request ids already turned into
        # backfill blocks, so each finished request contributes exactly once)
        self.s_prev_totals = None
        self.s_prev_wall = None
        self.s_seen_ids = set()
        self.seq_max = 0            # high-water observed sequence length

    # ---- detection + dispatch -------------------------------------------
    def sample(self, now):
        try:
            body = http_get(self.base + '/metrics', 3.0)
        except Exception:
            self.up = False
            self.l_prev = self.s_prev_totals = None
            return {'up': 0}
        self.up = True
        if body.lstrip()[:1] == '{':
            try:
                data = json.loads(body)
            except ValueError:
                data = None
            if isinstance(data, dict) and ('live' in data or 'totals' in data):
                self.kind = 'strata'
                s = self._sample_strata(now, data)
            else:
                return {'up': 1, 'kind': self.kind}   # JSON, not our shape
        else:
            self.kind = 'llama'
            s = self._sample_llama(now, parse_prom(body))
        s['kind'] = self.kind
        self._refresh_model(now)
        return s

    def _gap(self, prev_wall, now):
        """A long gap (endpoint switched backend kind, monitor paused, host
        suspend) means cumulative counters jumped: skip rates for one tick."""
        return prev_wall is None or (now - prev_wall) > self.interval * 3 + 5

    def _ticks(self, busy_s):
        return max(1, int(round(busy_s / self.interval)))

    # ---- llama.cpp: Prometheus text, counters arrive in LUMPS -----------
    def _sample_llama(self, now, m):
        s = {'req_proc': m.get('llamacpp:requests_processing'),
             'req_def': m.get('llamacpp:requests_deferred'),
             'seq_max': m.get('llamacpp:n_tokens_max'),
             'prompt_wall': 0.0, 'gen_wall': 0.0}
        if self.l_prev is not None and not self._gap(self.l_prev_wall, now):
            p = self.l_prev

            def delta(key):
                a, b = m.get(key), p.get(key)
                # a < b => server restarted: no rate for this tick
                return a - b if (a is not None and b is not None and a >= b) else None

            dpt = delta('llamacpp:prompt_tokens_total')
            dps = delta('llamacpp:prompt_seconds_total')
            dtp = delta('llamacpp:tokens_predicted_total')
            dts = delta('llamacpp:tokens_predicted_seconds_total')
            dd = delta('llamacpp:spec_decode_num_draft_tokens_total')
            da = delta('llamacpp:spec_decode_num_accepted_tokens_total')
            dc = delta('llamacpp:prompt_tokens_cached_total')
            # llama flushes prompt tokens+seconds in ONE scrape at the first
            # output token, and predicted tokens+seconds in ONE scrape at the
            # end of the slot. So d/dt is one spike per request. Instead take
            # the busy speed (tokens / busy-seconds) and spread it back over
            # the trailing ticks that phase actually ran -> solid block.
            if dpt and (dps or 0) > 0.05:
                sp = dpt / dps
                s['prompt_tps'] = sp
                s['_pp'] = (sp, self._ticks(dps))
            if dtp and (dts or 0) > 0.05:
                sp = dtp / dts
                s['gen_tps'] = sp
                s['_gp'] = (sp, self._ticks(dts))
            if dd and da is not None:
                s['spec_acc'] = 100.0 * da / dd
            if dc and (dc + (dpt or 0)) > 0:
                s['cache_pct'] = 100.0 * dc / (dc + dpt)
        self.l_prev = {k: v for k, v in m.items() if k.endswith('_total')}
        self.l_prev_wall = now
        return s

    # ---- Strata: JSON metrics, EXACT per-request records + live rates ---
    def _sample_strata(self, now, d):
        live = d.get('live') or {}
        state = live.get('state')
        busy = bool(state) and state != 'idle'
        s = {'req_proc': 1 if busy else 0,
             'req_def': live.get('queued') or 0}
        pf = live.get('prefill_tok_s_mean')
        tk = live.get('tok_s')
        # live streaming rates give immediate feedback while a request runs;
        # consecutive busy polls naturally form a solid block on the sparkline.
        s['prompt_wall'] = pf if (busy and pf) else 0.0
        s['gen_wall'] = tk if (busy and tk) else 0.0
        if pf:
            s['prompt_tps'] = pf
        if tk:
            s['gen_tps'] = tk
        cur = (live.get('prompt_total') or 0) + (live.get('generated') or 0)
        if cur > self.seq_max:
            self.seq_max = cur
        s['seq_max'] = self.seq_max or None
        # aggregate spec-accept between completions, from cumulative totals
        t = d.get('totals') or {}
        if self.s_prev_totals is not None and not self._gap(self.s_prev_wall, now):
            do = (t.get('output_tokens', 0) or 0) - (self.s_prev_totals.get('output_tokens', 0) or 0)
            da = (t.get('drafts_accepted', 0) or 0) - (self.s_prev_totals.get('drafts_accepted', 0) or 0)
            dd = (t.get('drafts_offered', 0) or 0) - (self.s_prev_totals.get('drafts_offered', 0) or 0)
            if dd and do:
                s['spec_acc'] = 100.0 * da / dd
        self.s_prev_totals = t
        self.s_prev_wall = now
        # the most recent NOT-yet-seen finished request -> exact block+stats
        reqs = d.get('requests') or []
        for r in reversed(reqs):
            rid = r.get('id')
            if rid in self.s_seen_ids:
                continue
            self.s_seen_ids.add(rid)
            pt = r.get('prompt_total') or r.get('prompt_tokens') or 0
            ot = r.get('output_tokens') or r.get('engine_generated') or 0
            pms = r.get('prompt_ms') or 0.0
            dms = r.get('decode_ms') or 0.0
            reused = r.get('reused') or 0
            if pt + ot > self.seq_max:
                self.seq_max = pt + ot
            uncached = max(pt - reused, 0)
            if pms > 1 and uncached > 0:
                sp = round(uncached / (pms / 1000.0), 1)
                s['_pp'] = (sp, self._ticks(pms / 1000.0))
                s['prompt_tps'] = sp
                s['last_prompt_tok'] = uncached
            if dms > 1 and ot > 0:
                sp = round(ot / (dms / 1000.0), 1)
                s['_gp'] = (sp, self._ticks(dms / 1000.0))
                s['gen_tps'] = sp
                s['last_decode_tok'] = ot
            if pt > 0:
                s['cache_pct'] = round(100.0 * reused / pt, 1)
            if r.get('drafts_offered') and r.get('drafts_accepted') is not None:
                s['spec_acc'] = round(100.0 * r['drafts_accepted'] / r['drafts_offered'], 1)
            break
        ids = {r.get('id') for r in reqs}
        if len(self.s_seen_ids) > 512:
            self.s_seen_ids &= ids
        return s

    def _refresh_model(self, now):
        if now - self.model_ts <= 30:
            return
        self.model_ts = now
        try:
            if self.kind == 'strata':
                mid = json.loads(http_get(self.base + '/health', 3.0)).get('model')
            else:
                data = json.loads(http_get(self.base + '/v1/models', 3.0)).get('data') or []
                mid = data[0]['id'] if data else None
            if mid:
                self.model = mid
        except Exception:
            pass


# ---------------------------------------------------------------- config

DEFAULTS = {
    'host': '127.0.0.1',
    'port': 8085,
    'urls': ['http://127.0.0.1:8080'],
    'interval': 2.0,
    'window': 1800,
    'gpus': [],                      # visible GPU identifiers; [] = all
    'price': {
        'enabled': True,
        'zone': 'fi',
        'timezone': 'Europe/Helsinki',
        'sources': ['spotwatts', 'volton'],
        'include_vat': False,
        'fallback_vat_percent': 25.5,
        'revalidate_hours': 6,
        'fetch_interval': 60,        # coverage/refetch check cadence, seconds
        'chart_back_hours': 6,       # chart window: past .. ahead
        'chart_fwd_hours': 24,
    },
    'throttle': {
        'enabled': False,
        'threshold_c_per_kwh': 10.0,
        'interval': 60,
        'devices': [                 # identifier = same forms as gpus[]
            # {"gpu": "GPU-...", "restore_watts": 250,
            #  "low_watts": null, "fallback_low_watts": null}
        ],
    },
}


def _deep_merge(base, over):
    out = dict(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(argv_path=None):
    """defaults < config file < MON_* env overrides."""
    cfg = json.loads(json.dumps(DEFAULTS))  # deep copy
    path = argv_path or os.environ.get('MON_CONFIG') or '/etc/llama-mon/config.json'
    used = None
    if path and os.path.exists(path):
        try:
            with open(path, encoding='utf-8') as f:
                cfg = _deep_merge(cfg, json.load(f))
            used = path
        except Exception as e:
            print('monitor: bad config %s: %r - using defaults/env' % (path, e),
                  flush=True)
    cfg['_config_file'] = used

    def env(name, cast):
        v = os.environ.get(name)
        if v is None or v == '':
            return None
        try:
            return cast(v)
        except ValueError:
            return None
    if env('MON_HOST', str) is not None:
        cfg['host'] = env('MON_HOST', str)
    if env('MON_PORT', int) is not None:
        cfg['port'] = env('MON_PORT', int)
    urls = env('MON_URLS', str)
    if urls:
        cfg['urls'] = [u for u in re.split(r'[,\s]+', urls) if u]
    for name, key, cast in (('MON_INTERVAL', 'interval', float),
                            ('MON_WINDOW', 'window', int)):
        if env(name, cast) is not None:
            cfg[key] = env(name, cast)
    gpus = env('MON_GPUS', str)
    if gpus:
        cfg['gpus'] = [u for u in re.split(r'[,\s]+', gpus) if u]
    cfg['interval'] = max(0.5, float(cfg['interval']))
    return cfg


# ------------------------------------------------------------- spot prices
# Free mirrors of the Nord Pool / SDAC day-ahead clearing results (both are
# verified live here):
#   spotwatts  https://spotwatts.com/api/v1/{zone}/{YYYY-MM-DD}.json
#              "Prices: Spotwatts (https://spotwatts.com/)" - attribution
#              required only if the numbers ever get republished publicly.
#   volton     https://public-data.volton.energy/v1/day-ahead-spot-fi/
#              Volton Technology OU, CC-BY-4.0. FI zone only.

PRICE_UA = {'User-Agent': 'llama-mon/1.0 (personal spot-price GPU throttle)'}


def _price_get_json(url, timeout=20):
    req = urllib.request.Request(url, headers=PRICE_UA)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode('utf-8')), None
    except Exception as e:  # HTTPError 404 = tomorrow not published yet
        return None, str(e)


def _parse_iso(s):
    return dt.datetime.fromisoformat(s)


def _norm_intervals(rows, fallback_vat_pct):
    """rows: [{start,end,price,price_vat}] in c/kWh -> epoch-keyed table."""
    out = {}
    for row in rows:
        s, e = _parse_iso(row['start']), _parse_iso(row['end'])
        ex, vat = row.get('price'), row.get('price_vat')
        if ex is None and vat is None:
            continue
        if vat is None and ex is not None:
            vat = ex * (1 + fallback_vat_pct / 100.0)
        out[int(s.timestamp())] = {'end': int(e.timestamp()), 'ex_vat': ex, 'vat': vat}
    return out


def fetch_spotwatts_day(zone, date_str):
    data, err = _price_get_json(
        'https://spotwatts.com/api/v1/%s/%s.json' % (zone.lower(), date_str))
    if data is None:
        return None, 'spotwatts: %s' % err
    rows = [{'start': i['start'], 'end': i['end'],
             'price': round(i['price'] * 100.0, 5),       # EUR/kWh -> c/kWh
             'price_vat': None if i.get('price_vat') is None
             else round(i['price_vat'] * 100.0, 5)}
            for i in data.get('intervals', [])]
    return (rows, None) if rows else (None, 'spotwatts: empty day')


def fetch_volton_day(zone, date_str, tz):
    if zone.lower() != 'fi':
        return None, 'volton: only the FI zone is exposed'
    data, err = _price_get_json(
        'https://public-data.volton.energy/v1/day-ahead-spot-fi/latest.json')
    if data is None:
        return None, 'volton: %s' % err
    by = {}
    for r in data.get('rows', []):
        if r.get('price_eur_mwh') is None or not r.get('mtu_start'):
            continue
        by[int(_parse_iso(r['mtu_start'].replace('Z', '+00:00')).timestamp())] = \
            r['price_eur_mwh']
    starts = sorted(by)
    rows = []
    for i, s_ts in enumerate(starts):
        # MTU length = gap to the next row (hourly before Oct 2025, 15 min since)
        ln = starts[i + 1] - s_ts if i + 1 < len(starts) else 900
        s = dt.datetime.fromtimestamp(s_ts, dt.timezone.utc)
        e = dt.datetime.fromtimestamp(s_ts + min(ln, 3600), dt.timezone.utc)
        rows.append({'start': s.isoformat(), 'end': e.isoformat(),
                     # 1 EUR/MWh = 0.1 c/kWh (1 MWh = 1000 kWh)
                     'price': by[s_ts] / 10.0, 'price_vat': None})
    day = [r for r in rows
           if _parse_iso(r['start']).astimezone(tz).strftime('%Y-%m-%d') == date_str]
    return (day, None) if day else (None, 'volton: no rows for %s yet' % date_str)


class PriceBank:
    """In-memory day-ahead price table covering now .. now+24h, refetched on a
    timer thread. Days already cached are re-fetched at most every
    revalidate_hours (auction results do get revised); tomorrow's 96 quarters
    are picked up once published (~14:00 EET)."""

    def __init__(self, pcfg):
        self.cfg = pcfg
        self.tz = ZoneInfo(pcfg.get('timezone', 'Europe/Helsinki'))
        self.enabled = bool(pcfg.get('enabled', True))
        self.lock = threading.Lock()
        self.table = {}          # epoch_start -> {end, ex_vat, vat}
        self.fetched = {}        # YYYY-MM-DD -> ts of last successful fetch
        self.last_err = None
        self.next_check = 0.0

    def _fetch_day(self, source, date_str):
        zone = self.cfg.get('zone', 'fi')
        if source == 'spotwatts':
            return fetch_spotwatts_day(zone, date_str)
        if source == 'volton':
            return fetch_volton_day(zone, date_str, self.tz)
        return None, 'unknown price source %r' % source

    def _covering(self, start_ts):
        hit = self.table.get(start_ts)
        if hit is not None:
            return hit
        for s, e in self.table.items():
            if s <= start_ts < e['end']:
                return e
        return None

    def ensure(self, now):
        """Fetch any local day needed to cover now .. now+24h. Called by the
        refresher thread; returns nothing, state readable under self.lock."""
        reval = float(self.cfg.get('revalidate_hours', 6)) * 3600
        end_ts = now + 24 * 3600

        def local_day(ts):
            return (dt.datetime.fromtimestamp(ts, dt.timezone.utc)
                    .astimezone(self.tz).strftime('%Y-%m-%d'))
        with self.lock:
            days, cur = set(), int(now)
            while cur < end_ts:
                hit = self._covering(cur)
                if hit is None:
                    days.add(local_day(cur))
                    cur += 900
                else:
                    cur = hit['end']
            # today/tomorrow stay worth a periodic refresh while they intersect
            # the window: prices get revised, tomorrow only appears after ~14
            for probe in (now, end_ts):
                d = local_day(probe)
                age = now - self.fetched.get(d, 0)
                if d not in days and (d not in self.fetched or age > reval):
                    days.add(d)
            days = sorted(days)
            errs = []
            for date_str in days:
                for source in self.cfg.get('sources', ['spotwatts', 'volton']):
                    rows, err = self._fetch_day(source, date_str)
                    if rows:
                        self.table.update(_norm_intervals(
                            rows, self.cfg.get('fallback_vat_percent', 25.5)))
                        self.fetched[date_str] = now
                        break
                    errs.append(err)
            keep_from = local_day(now - 86400)
            self.fetched = {k: v for k, v in self.fetched.items() if k >= keep_from}
            self.last_err = '; '.join(errs) if errs else None

    def current(self, now_ts):
        with self.lock:
            e = self._covering(int(now_ts))
            if e is None:
                return None
            key = 'vat' if self.cfg.get('include_vat') else 'ex_vat'
            p = e.get(key)
            return p if p is not None else e.get('ex_vat')

    def curve(self, now_ts):
        """[{start,end,price}] over [now-back, now+fwd] for the chart."""
        back = float(self.cfg.get('chart_back_hours', 6)) * 3600
        fwd = float(self.cfg.get('chart_fwd_hours', 24)) * 3600
        lo, hi = now_ts - back, now_ts + fwd
        with self.lock:
            pts = [{'start': s, 'end': e['end'],
                    'price': e.get('vat') if self.cfg.get('include_vat')
                    else e.get('ex_vat')}
                   for s, e in self.table.items()
                   if e['end'] > lo and s < hi]
        pts.sort(key=lambda p: p['start'])
        return pts


# ------------------------------------------------------------------ AMD GPUs
# Stats come from sysfs (works without ROCm installed): gpu_busy_percent,
# mem_info_vram_*, hwmon temp/power files, PCI link attributes. amd-smi (ROCm
# >= 6.1, "SMI tool") is used only to augment names/UUIDs when it is present;
# its JSON layout has changed across releases, so lookups accept several key
# spellings. No AMD card exists in this box -> the path is written blind;
# verify on an AMD machine before trusting it.

AMD_VENDOR_ID = '0x1002'
GEN_BY_RATE = ((32.0, 5), (16.0, 4), (8.0, 3), (5.0, 2), (2.5, 1))


def _pci_gen(bdf):
    v = _read('/sys/bus/pci/devices/%s/current_link_speed' % bdf)
    m = re.match(r'(\d+(?:\.\d+)?)', v or '')
    if not m:
        return None
    rate = float(m.group(1))
    for lo, gen in GEN_BY_RATE:
        if rate >= lo - 0.01:
            return gen
    return 1


def _amd_hwmon(dev_dir):
    try:
        hm = os.path.join(dev_dir, 'hwmon')
        for d in sorted(os.listdir(hm)):
            yield os.path.join(hm, d)
    except OSError:
        pass


def _amd_read_first(dev_dir, names, scale=1.0):
    """First numeric value found among hwmon files called `names` (µW/mW/mC
    scaled to W/W/C by `scale`)."""
    for sub in _amd_hwmon(dev_dir):
        for n in names:
            p = os.path.join(sub, n)
            if os.path.exists(p):
                v = _read(p)
                try:
                    return float(v) * scale
                except (TypeError, ValueError):
                    pass
    return None


_BDF_RE = re.compile(r'^[0-9a-f]{4,8}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9a-f]$')


def _amd_smi_info():
    """bdf -> {name, uuid} from amd-smi static JSON, {} when unavailable.
    The top-level key and nesting vary across ROCm releases ("gpu_data" on
    current amd-smi, "static_info" on older ones), so walk each device blob
    for the first BDF-looking string, market name and UUID instead of
    hard-coding one layout."""
    rc, out = _sh(['amd-smi', 'static', '--json'])
    if rc != 0:
        return {}
    try:
        data = json.loads(out)
    except ValueError:
        return {}

    def walk(obj, pairs):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    walk(v, pairs)
                else:
                    pairs.append((str(k).lower(), v))
        elif isinstance(obj, list):
            for v in obj:
                walk(v, pairs)
        return pairs

    def val(pairs, *keys):
        for want in keys:
            for k, v in pairs:
                if k == want and v not in (None, ''):
                    return v
        return None

    items = data if isinstance(data, list) else \
        (data.get('gpu_data') or data.get('static_info') or
         data.get('data') or data.get('devices') or [])
    out_map = {}
    for it in items if isinstance(items, list) else []:
        if not isinstance(it, dict):
            continue
        pairs = walk(it, [])
        bdf = val(pairs, 'bus_location', 'pci_address', 'bdf', 'pci_bus_id',
                  'bus_id', 'bus')
        if bdf is None:      # last resort: any value that parses as a BDF
            bdf = next((v.strip() for k, v in pairs
                        if isinstance(v, str) and _BDF_RE.match(v.strip().lower())),
                       None)
        name = val(pairs, 'market_name', 'marketing_name', 'sku',
                   'market_name_brand')
        uuid = val(pairs, 'uuid')
        if bdf:
            bdf = str(bdf).strip().lower()
            if re.match(r'^[0-9a-f]{2}:[0-9a-f]{2}\.[0-9a-f]$', bdf):
                bdf = '0000:' + bdf          # domain-less "0b:00.0" form
            elif len(bdf) == 12 and bdf[4] == ':' and bdf[:2] != '00':
                bdf = '0000:' + bdf         # amd-smi may drop the full domain
            out_map[bdf] = {'name': name, 'uuid': uuid}
    return out_map


def query_amd():
    """All AMD cards as monitor card dicts (same keys as the NVIDIA path),
    [] when no AMD GPU exists."""
    cards = []
    try:
        entries = sorted(os.listdir('/sys/class/drm'))
    except OSError:
        return cards
    smi = _amd_smi_info()
    for ent in entries:
        m = re.fullmatch(r'card(\d+)', ent)
        if not m:
            continue
        dev_dir = '/sys/class/drm/%s/device' % ent
        if _read(os.path.join(dev_dir, 'vendor')) != AMD_VENDOR_ID:
            continue
        bdf = os.path.basename(os.path.realpath(dev_dir)).lower()
        extra = smi.get(bdf, {})
        dev_id = (_read(os.path.join(dev_dir, 'device')) or '?').lstrip('0x')
        name = extra.get('name') or ('AMD GPU (%s)' % dev_id)
        busy = _read(os.path.join(dev_dir, 'gpu_busy_percent'))
        vt = _read(os.path.join(dev_dir, 'mem_info_vram_total'))
        vu = _read(os.path.join(dev_dir, 'mem_info_vram_used'))

        def mb(x):
            try:
                return float(x) / 1048576.0
            except (TypeError, ValueError):
                return None
        width = _read('/sys/bus/pci/devices/%s/current_link_width' % bdf)
        cards.append({
            'vendor': 'amd', 'uuid': extra.get('uuid') or ('amd:' + bdf),
            'name': name, 'bdf': bdf, 'index': None, 'card': 'card%s' % m.group(1),
            'dev_dir': dev_dir,
            'temp': _amd_read_first(dev_dir, ('temp1_input', 'edge_temp'), 0.001),
            'util': _num(busy or ''),
            'mem_util': None,
            'mem_total': mb(vt), 'mem_used': mb(vu),
            'power': _amd_read_first(dev_dir, ('power1_average', 'power1_input'), 1e-6),
            'power_limit': _amd_read_first(dev_dir, ('power1_cap',), 1e-6),
            'cap_min': _amd_read_first(dev_dir, ('power1_cap_min',), 1e-6),
            'cap_max': _amd_read_first(dev_dir, ('power1_cap_max',), 1e-6),
            'cap_default': None,
            'gen': _pci_gen(bdf), 'width': _num(width or ''),
        })
    return cards


def set_power_cap(card, watts):
    """Write a power cap to a monitor card dict; raises RuntimeError on failure."""
    w = ('%.2f' % float(watts)).rstrip('0').rstrip('.')
    if card.get('vendor') == 'amd':
        cap = None
        for sub in _amd_hwmon(card['dev_dir']):
            if os.path.exists(os.path.join(sub, 'power1_cap')):
                cap = os.path.join(sub, 'power1_cap')
                break
        if cap:
            try:
                with open(cap, 'w') as f:
                    f.write(str(int(float(watts) * 1e6)))   # hwmon = microwatts
                return
            except OSError:
                rc, out = _sh_priv(['tee', cap], input=str(int(float(watts) * 1e6)))
                if rc == 0:
                    return
                raise RuntimeError('writing %s failed: %s' % (cap, out.strip()[:200]))
        pp = os.path.join(card['dev_dir'], 'pp_power_limit')
        if os.path.exists(pp):
            try:
                with open(pp, 'w') as f:
                    f.write(w)
                return
            except OSError:
                rc, out = _sh_priv(['tee', pp], input=w)
                if rc == 0:
                    return
        rc, out = _sh_priv(['amd-smi', 'set', '--device', card['bdf'],
                            '--power-limit', w])
        if rc == 0:
            return
        raise RuntimeError('amd cap setting failed (sysfs + amd-smi): %s'
                           % out.strip()[:200])
    rc, out = _sh_priv(['nvidia-smi', '-i', card['uuid'], '-pl', w])
    if rc != 0:
        raise RuntimeError('nvidia-smi -pl failed: %s' % out.strip()[:200])


NV_FIELDS = ('index,uuid,name,pci.bus_id,temperature.gpu,utilization.gpu,'
             'utilization.memory,memory.total,memory.used,power.draw,power.limit,'
             'pcie.link.gen.current,pcie.link.width.current,'
             'power.default_limit,power.min_limit,power.max_limit')


def _num(x):
    x = x.strip().strip('[]')
    if x in ('', 'N/A', 'Not Supported'):
        return None
    try:
        f = float(x)
    except ValueError:
        return None
    return int(f) if f.is_integer() else f


def _sh(cmd, timeout=10, input=None):
    """Run argv list -> (rc, stdout+stderr text)."""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True,
                           input=input, timeout=timeout)
        return p.returncode, (p.stdout or '') + (p.stderr or '')
    except Exception as e:
        return 127, str(e)


def _perm_err(text):
    t = (text or '').lower()
    return ('permission' in t or 'insufficient' in t
            or ('access' in t and 'denied' in t))


def _sh_priv(cmd, input=None):
    """Run a command that needs privilege to change power caps: direct first,
    then sudo -n (for non-root services; see the installer's sudoers drop-in)."""
    rc, out = _sh(cmd, input=input)
    if rc != 0 and _perm_err(out):
        rc, out = _sh(['sudo', '-n'] + cmd, input=input)
    return rc, out


def _read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def _parent_bdf(bdf):
    try:
        p = os.path.dirname(os.path.realpath('/sys/bus/pci/devices/' + bdf))
        return os.path.basename(p)
    except OSError:
        return None


def read_aer(bdf):
    """{label: count} from aer_dev_correctable, or None."""
    try:
        toks = open('/sys/bus/pci/devices/%s/aer_dev_correctable' % bdf).read().split()
    except OSError:
        return None
    d = {}
    for i in range(0, len(toks) - 1, 2):
        try:
            d[toks[i]] = int(toks[i + 1])
        except ValueError:
            pass
    return d


AER_DETAIL = ('BadTLP', 'BadDLLP', 'Timeout', 'Rollover')


def _card_ids(card):
    """Lowercased identifiers a config entry may match against."""
    ids = {str(card.get('uuid') or '').lower()}
    if card.get('bdf'):
        ids.add(card['bdf'].lower())
        ids.add('amd:' + card['bdf'].lower())
    if card.get('card'):
        ids.add(card['card'])
        ids.add('amd:' + card['card'])
    if card.get('index') is not None:
        ids.add(str(card['index']))
        ids.add('nvidia:%s' % card['index'])
    ids.discard('')
    return ids


def card_matches(card, wanted):
    """wanted = [] (all) or list of identifier strings; NVIDIA UUID entries
    also accept a unique substring of the UUID for handy typing."""
    if not wanted:
        return True
    ids = _card_ids(card)
    uuid = str(card.get('uuid') or '').lower()
    for tok in wanted:
        t = tok.lower()
        if t in ids or (uuid and t in uuid):
            return True
    return False


class GpuBank:
    def __init__(self, gpu_filter):
        self.filter = [str(x) for x in gpu_filter]
        self.cards = []        # metadata + last live values (for the API)
        self.aer_detail = {}   # bdf -> {'dev': {...}, 'port': {...}}
        self.prev_aer = {}     # bdf -> (cumulative, wall)
        self.lock = threading.Lock()

    def _query_nvidia(self):
        try:
            out = subprocess.run(
                ['nvidia-smi', '--query-gpu', NV_FIELDS, '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=10)
        except Exception:
            return []
        if out.returncode != 0:
            return []
        cards = []
        for line in out.stdout.splitlines():
            f = [x.strip() for x in line.split(',')]
            if len(f) < 16:
                continue
            bdf = f[3]
            if len(bdf) > 12:
                bdf = bdf[4:]  # 00000000:81:00.0 -> 0000:81:00.0
            cards.append({
                'vendor': 'nvidia',
                'uuid': f[1], 'name': f[2], 'bdf': bdf, 'index': _num(f[0]),
                'temp': _num(f[4]), 'util': _num(f[5]), 'mem_util': _num(f[6]),
                'mem_total': _num(f[7]), 'mem_used': _num(f[8]),
                'power': _num(f[9]), 'power_limit': _num(f[10]),
                'gen': _num(f[11]), 'width': _num(f[12]),
                'cap_default': _num(f[13]), 'cap_min': _num(f[14]),
                'cap_max': _num(f[15]),
            })
        return cards

    def _query(self):
        cards = self._query_nvidia() + query_amd()
        if not cards and not self.filter:
            return None      # nothing probed at all: keep the previous state
        return [c for c in cards if card_matches(c, self.filter)]

    def snapshot(self):
        with self.lock:
            return [dict(c) for c in self.cards]

    def sample(self, now):
        cards = self._query()
        if cards is None:
            return {}
        with self.lock:
            if not self.cards or [c['bdf'] for c in cards] != [c['bdf'] for c in self.cards]:
                for c in cards:
                    c['port'] = _parent_bdf(c['bdf'])
                self.cards = cards
            else:
                # topology unchanged: refresh the live fields (temp/util/mem/power)
                # in the cached cards; the fresh query result carries no 'port'.
                for old, new in zip(self.cards, cards):
                    old.update(new)
            s = {}
            for i, c in enumerate(self.cards):
                dev = read_aer(c['bdf'])
                port = read_aer(c['port']) if c['port'] else None
                self.aer_detail[c['bdf']] = {'dev': dev, 'port': port}
                cum, have = 0, False
                for d in (dev, port):
                    if d and d.get('TOTAL_ERR_COR') is not None:
                        cum += d['TOTAL_ERR_COR']
                        have = True
                if have:
                    s['g%d.aer_cum' % i] = cum
                    prev = self.prev_aer.get(c['bdf'])
                    if prev and now > prev[1]:
                        s['g%d.aer_rate' % i] = max(0.0, (cum - prev[0]) / (now - prev[1]) * 60.0)
                    else:
                        s['g%d.aer_rate' % i] = 0.0
                    self.prev_aer[c['bdf']] = (cum, now)
                s['g%d.temp' % i] = c['temp']
                s['g%d.util' % i] = c['util']
                s['g%d.mem_used' % i] = (c['mem_used'] / 1024.0) if c['mem_used'] is not None else None
                s['g%d.mem_pct' % i] = (100.0 * c['mem_used'] / c['mem_total']
                                        if c['mem_total'] and c['mem_used'] is not None else None)
                s['g%d.power' % i] = c['power']
                s['g%d.plimit' % i] = c['power_limit']
            return s

    def meta(self):
        out = []
        with self.lock:
            cards = list(self.cards)
            detail = dict(self.aer_detail)
        for i, c in enumerate(cards):
            det = detail.get(c['bdf'], {})
            dev, port = det.get('dev'), det.get('port')

            def brief(d):
                return {k: d.get(k, 0) for k in AER_DETAIL} if d else None

            out.append({'id': i, 'name': c['name'], 'uuid': c['uuid'],
                        'vendor': c.get('vendor', 'nvidia'),
                        'index': c['index'], 'card': c.get('card'),
                        'bdf': c['bdf'],
                        'link': 'Gen%s x%s' % (c['gen'], c['width']),
                        'power_limit': c['power_limit'],
                        'cap_min': c.get('cap_min'), 'cap_max': c.get('cap_max'),
                        'cap_default': c.get('cap_default'),
                        'mem_total': c['mem_total'],
                        'aer_dev_total': dev.get('TOTAL_ERR_COR') if dev else None,
                        'aer_port_total': port.get('TOTAL_ERR_COR') if port else None,
                        'aer_dev': brief(dev), 'aer_port': brief(port)})
        return out


# ------------------------------------------------------- spot-price throttle

def _amd_cap_now(card):
    """Current AMD power cap in W read straight from sysfs (fresh, cheap)."""
    return _amd_read_first(card['dev_dir'], ('power1_cap',), 1e-6)


class Throttle:
    """Reconciles GPU power caps against the current FI spot price, the way
    gpu-power.py did from cron, but inside the monitor process: above the
    threshold -> minimum power limit, at/below -> restore_watts. Writes only
    when current != target, so it also repairs caps changed elsewhere."""

    def __init__(self, tcfg, price, gpus):
        self.cfg = tcfg
        self.price = price
        self.gpus = gpus
        self.enabled = bool(tcfg.get('enabled')) and price.enabled
        self.threshold = float(tcfg.get('threshold_c_per_kwh', 10.0))
        self.interval = max(10.0, float(tcfg.get('interval', 60)))
        self.lock = threading.Lock()
        self.high = None
        self.price_now = None
        self.last_check = None
        self.no_price_logged = False
        self.events = deque(maxlen=50)
        self.devices = [{'spec': dict(d), 'gpu': str(d.get('gpu')),
                         'name': None, 'state': '-', 'want': None,
                         'have': None, 'status': 'pending', 'error': None}
                        for d in (tcfg.get('devices') or [])]

    def _event(self, msg):
        line = '%s %s' % (time.strftime('%Y-%m-%d %H:%M:%S'), msg)
        print(line, flush=True)
        with self.lock:
            self.events.append(line)

    @staticmethod
    def _find_card(gpus, ident):
        for c in gpus.snapshot():
            if card_matches(c, [str(ident)]):
                return c
        return None

    def _target_watts(self, card, spec, high):
        """Watts we want this GPU to have right now; raises ValueError with a
        human reason when the target cannot be computed."""
        if high:
            if spec.get('low_watts') is not None:
                return float(spec['low_watts'])
            lo = card.get('cap_min')
            if lo is None:
                lo = spec.get('fallback_low_watts')
            if lo is None:
                raise ValueError('driver minimum power limit unknown - set '
                                 'low_watts or fallback_low_watts')
            return float(lo)
        w = spec.get('restore_watts', 'default')
        if w == 'default':
            # AMD sysfs has no "factory default": its default cap equals the
            # max unless someone overrode it, so cap_max is the best guess.
            w = card.get('cap_default')
            if w is None:
                w = card.get('cap_max')
            if w is None:
                raise ValueError('default power limit unknown - set '
                                 'restore_watts explicitly')
        return float(w)

    def _have_watts(self, card):
        if card.get('vendor') == 'amd':
            return _amd_cap_now(card)
        return card.get('power_limit')

    def check(self, now):
        price = self.price.current(now)
        with self.lock:
            self.price_now = price
            self.last_check = now
        if price is None:
            if not self.no_price_logged:
                self._event('throttle: no spot price for the current interval '
                            '(auction not published yet?) - caps left untouched')
                self.no_price_logged = True
            return
        self.no_price_logged = False
        high = price > self.threshold
        if high != self.high:
            self._event('spot %.2f c/kWh %s threshold %.2f -> %s' % (
                price, 'above' if high else 'at/below', self.threshold,
                'THROTTLE to minimum caps' if high else 'restore caps'))
            self.high = high
        if not self.gpus.snapshot():
            return  # GPU list not sampled yet (first cycles after start): defer
        cards_checked = []
        for dev in self.devices:
            spec = dev['spec']
            ident = spec.get('gpu')
            label = str(ident)
            card = None
            try:
                card = self._find_card(self.gpus, ident)
                if card is None:
                    raise ValueError('GPU %r not found (not present or hidden '
                                     'by the gpus visibility filter)' % ident)
                label = '%s (%s)' % (card.get('name'), card.get('uuid'))
                want = self._target_watts(card, spec, high)
                have = self._have_watts(card)
                if have is not None and abs(have - want) < 0.5:
                    status, err = 'ok', None
                else:
                    set_power_cap(card, want)
                    status = 'set %.1fW' % want
                    err = None
                    self._event('%s cap %s -> %.1fW (%s)' % (
                        label, '%.1fW' % have if have is not None else '?',
                        want, 'high price' if high else 'normal price'))
            except Exception as e:
                want, have, status, err = None, None, 'error', str(e)
                if dev['error'] != err:      # log each new failure once, not spam
                    self._event('throttle ERROR %s: %s' % (label, err))
            cards_checked.append({
                'label': label, 'gpu': str(ident), 'name': card.get('name') if card else None,
                'uuid': card.get('uuid') if card else None,
                'state': 'high' if high else 'low',
                'want': want, 'have': have, 'status': status, 'error': err})
            with self.lock:
                dev.update(state='high' if high else 'low', want=want,
                           have=have, status=status, error=err, name=label)
        return cards_checked

    def status(self):
        with self.lock:
            devs = [{k: v for k, v in d.items() if k != 'spec'} for d in self.devices]
            events = list(self.events)[-12:]
            return {'enabled': self.enabled,
                    'threshold': self.threshold,
                    'include_vat': bool(self.price.cfg.get('include_vat')),
                    'interval': self.interval,
                    'price': self.price_now, 'high': self.high,
                    'last_check': self.last_check,
                    'devices': devs, 'events': events}


class HostBank:
    def __init__(self):
        self.prev_cpu = None
        self.prev_net = None
        self.prev_wall = None
        self.cores = os.cpu_count() or 1
        self.mem_total_kb = None
        self.swap_total_kb = None

    def sample(self, now):
        s = {}
        try:
            f = [int(x) for x in open('/proc/stat').readline().split()[1:]]
            idle = f[3] + (f[4] if len(f) > 4 else 0)
            tot = sum(f)
            if self.prev_cpu and tot > self.prev_cpu[1]:
                s['h.cpu'] = 100.0 * (1.0 - (idle - self.prev_cpu[0]) / (tot - self.prev_cpu[1]))
            self.prev_cpu = (idle, tot)
        except Exception:
            pass
        try:
            la = open('/proc/loadavg').read().split()
            s['h.load1'] = float(la[0])
            s['h.load5'] = float(la[1])
            s['h.load15'] = float(la[2])
        except Exception:
            pass
        try:
            mi = {}
            for line in open('/proc/meminfo'):
                k, v = line.split(':', 1)
                mi[k] = int(v.split()[0])
            self.mem_total_kb = mi.get('MemTotal', self.mem_total_kb)
            self.swap_total_kb = mi.get('SwapTotal', self.swap_total_kb)
            avail = mi.get('MemAvailable')
            if self.mem_total_kb:
                s['h.ram'] = (self.mem_total_kb - (avail or 0)) / 1048576.0
            if self.swap_total_kb:
                s['h.swap'] = (self.swap_total_kb - mi.get('SwapFree', 0)) / 1048576.0
        except Exception:
            pass
        try:
            rx = tx = 0
            for line in open('/proc/net/dev').read().splitlines()[2:]:
                name, rest = line.split(':', 1)
                name = name.strip()
                if re.match(r'^(lo|veth|docker|br-|tap|virbr|kube|cni|flannel)', name):
                    continue
                f = rest.split()
                rx += int(f[0])
                tx += int(f[8])
            if self.prev_net is not None and now > self.prev_wall:
                dt = now - self.prev_wall
                s['h.rx'] = (rx - self.prev_net[0]) / dt / 1e6
                s['h.tx'] = (tx - self.prev_net[1]) / dt / 1e6
            self.prev_net = (rx, tx)
            self.prev_wall = now
        except Exception:
            pass
        try:
            s['h.uptime'] = float(open('/proc/uptime').read().split()[0])
        except Exception:
            pass
        return s


# ---------------------------------------------------------------- state

class State:
    def __init__(self, cfg):
        self.cfg = cfg
        self.backends = [Backend(u, cfg['interval']) for u in cfg['urls']]
        self.gpus = GpuBank(cfg.get('gpus') or [])
        self.host = HostBank()
        self.price = PriceBank(cfg['price'])
        self.throttle = Throttle(cfg['throttle'], self.price, self.gpus)
        self.interval = cfg['interval']
        self.started = time.time()
        self.hist = deque(maxlen=max(10, int(cfg['window'] / self.interval)))
        self.lock = threading.Lock()

    def collect(self):
        now = time.time()
        flat = {}
        patches = []   # (series-key, value, n_trailing_ticks)
        for i, b in enumerate(self.backends):
            for k, v in b.sample(now).items():
                if k in ('_pp', '_gp'):
                    key = 's%d.prompt_wall' % i if k == '_pp' else 's%d.gen_wall' % i
                    val, n = v
                    patches.append((key, round(val, 2), int(n)))
                else:
                    flat['s%d.%s' % (i, k)] = v
        flat.update(self.gpus.sample(now))
        flat.update(self.host.sample(now))
        flat['t'] = now
        with self.lock:
            self.hist.append(flat)
            # spread a just-finished phase's average speed back over the ticks
            # it actually ran (current tick + the N-1 older ones), so the
            # sparkline shows a solid block instead of a single spike.
            for key, val, n in patches:
                for j in range(1, min(n, len(self.hist)) + 1):
                    self.hist[-j][key] = val

    def api(self):
        with self.lock:
            buf = list(self.hist)
        n = min(POINTS, len(buf))
        tail = buf[-n:] if n else []
        keys, seen = [], set()
        for smp in reversed(tail):
            for k in smp:
                if k != 't' and k not in seen:
                    seen.add(k)
                    keys.append(k)
        series = {}
        for k in keys:
            series[k] = [round(v, 2) if isinstance(v, float) else v
                         for v in (smp.get(k) for smp in tail)]
        out = {
            'now': time.time(),
            'interval': self.interval,
            'points': n,
            'monitor_uptime': round(time.time() - self.started, 1),
            'servers': [{'id': i, 'label': b.label, 'up': b.up, 'model': b.model,
                         'kind': b.kind}
                        for i, b in enumerate(self.backends)],
            'gpus': self.gpus.meta(),
            'host': {'cores': self.host.cores,
                     'mem_total': (self.host.mem_total_kb / 1048576.0)
                     if self.host.mem_total_kb else None,
                     'swap_total': (self.host.swap_total_kb / 1048576.0)
                     if self.host.swap_total_kb else None},
            'series': series,
        }
        if self.price.enabled:
            now = time.time()
            out['price'] = {
                'zone': self.price.cfg.get('zone', 'fi').upper(),
                'current': self.price.current(now),
                'include_vat': bool(self.price.cfg.get('include_vat')),
                'threshold': self.throttle.threshold if self.throttle.enabled else None,
                'throttled': self.throttle.high if self.throttle.enabled else None,
                'curve': self.price.curve(now),
                'err': self.price.last_err,
            }
        out['throttle'] = self.throttle.status() if self.throttle.enabled else None
        return out


# ---------------------------------------------------------------- http

PAGE = r'''<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>llama.cpp monitor</title>
<style>
:root{color-scheme:light;--bg:#f6f8fa;--card:#ffffff;--line:#d0d7de;--tx:#1f2328;
--dim:#656d76;--acc:#0969da;--grn:#1a7f37;--amb:#9a6700;--red:#cf222e;--cyn:#0e7490;--vio:#8250df;}
*{box-sizing:border-box}
body{margin:0;padding:14px;background:var(--bg);color:var(--tx);
font:13px/1.45 -apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
h1{font-size:15px;margin:0 0 2px}
.top{display:flex;justify-content:space-between;align-items:baseline;gap:12px;flex-wrap:wrap;
margin-bottom:12px;padding:10px 14px;background:var(--card);border:1px solid var(--line);border-radius:10px}
.top .meta{color:var(--dim);font-size:12px}
.badge{display:inline-block;padding:1px 8px;border-radius:9px;font-size:11px;font-weight:600;margin-left:6px}
.badge.up{background:#dafbe1;color:var(--grn);border:1px solid #aceebb}
.badge.down{background:#ffebe9;color:var(--red);border:1px solid #ffcecb}
.badge.link{background:#ddf4ff;color:var(--acc);border:1px solid #b6e3ff}
.grid{display:grid;gap:12px;grid-template-columns:repeat(auto-fit,minmax(370px,1fr))}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px;min-width:0}
.card h2{font-size:13px;margin:0 0 10px;display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.card h2 .sub{color:var(--dim);font-weight:400;font-size:12px}
.stats{display:flex;flex-wrap:wrap;gap:6px 18px;margin:0 0 10px}
.stat{min-width:64px}
.stat .k{color:var(--dim);font-size:10.5px;text-transform:uppercase;letter-spacing:.05em}
.stat .v{font:600 14px/1.3 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
.sl{margin:7px 0}
.sl .row{display:flex;justify-content:space-between;font-size:11.5px;color:var(--dim);margin-bottom:1px}
.sl .val{font:600 12px ui-monospace,Menlo,Consolas,monospace;color:var(--tx)}
canvas{width:100%;height:46px;display:block}
.price canvas{height:78px}
.ev{margin-top:8px;color:var(--dim);font:11px/1.5 ui-monospace,Menlo,Consolas,monospace;
word-break:break-word}
.badge.hot{background:#ffebe9;color:var(--red);border:1px solid #ffcecb}
.badge.warn{background:#fff8c5;color:var(--amb);border:1px solid #eed27a}
.dual .val{color:var(--amb)}
</style></head><body>
<div class="top"><h1>llama.cpp monitor <span id="h-badge"></span></h1>
<div class="meta" id="h-meta"></div></div>
<div class="grid" id="grid"></div>
<script>
"use strict";
const $=id=>document.getElementById(id);
let refs=null, lastData=null, timer=null;

function fmt(v, u, d, dec){
  if (v==null || Number.isNaN(v)) return "–";
  d = (d===undefined) ? 1 : d; dec = (dec===undefined) ? true : dec;
  const x = dec ? v.toFixed(d) : String(Math.round(v));
  return u ? x+" "+u : x;
}
function fmtUptime(sec){
  if (sec==null) return "–";
  const d=Math.floor(sec/86400), h=Math.floor(sec%86400/3600), m=Math.floor(sec%3600/60);
  return (d?d+"d ":"")+(h?h+"h ":"")+m+"m";
}
function esc(s){return s.replace(/[&<>"]/g, c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));}

function spark(cv, a, b, o){
  o = o || {};
  const w = cv.clientWidth || 300, h = cv.clientHeight || 46;
  const dpr = window.devicePixelRatio || 1;
  if (cv.width !== Math.round(w*dpr) || cv.height !== Math.round(h*dpr)){
    cv.width = Math.round(w*dpr); cv.height = Math.round(h*dpr);
  }
  const x = cv.getContext("2d");
  x.setTransform(dpr,0,0,dpr,0,0);
  x.clearRect(0,0,w,h);
  x.strokeStyle = "#d0d7de";
  x.beginPath(); x.moveTo(0,h-.5); x.lineTo(w,h-.5); x.stroke();
  const clean = a.filter(v => v!=null);
  if (!clean.length) return;
  let mn = Math.min(...clean), mx = Math.max(...clean);
  const fixed = (o.min!=null) || (o.max!=null);
  if (o.min!=null) mn = Math.min(mn, o.min);
  if (o.max!=null) mx = Math.max(mx, o.max);
  if (mx-mn < 1e-9) mx = mn+1;
  if (!fixed){
    const pad = (mx-mn)*.12; mn -= pad; mx += pad;
  }
  const X = i => a.length>1 ? i/(a.length-1)*w : 0;
  const Y = v => h-2-(v-mn)/(mx-mn)*(h-4);
  const stroke = (arr, color, fill, scale) => {
    x.beginPath(); let on = false, lastX = 0;
    for (let i=0; i<arr.length; i++){
      const v = arr[i];
      if (v==null){ on = false; continue; }
      const px = X(i), py = scale ? scale(v) : Y(v);
      lastX = px;
      if (!on){ x.moveTo(px,py); on = true; } else x.lineTo(px,py);
    }
    x.strokeStyle = color; x.lineWidth = fill ? 1.4 : 1;
    if (!fill) x.setLineDash([3,3]);
    x.stroke(); x.setLineDash([]);
    if (fill){
      const g = x.createLinearGradient(0,0,0,h);
      g.addColorStop(0, color+"44"); g.addColorStop(1, color+"00");
      x.lineTo(lastX, h); x.lineTo(0, h); x.closePath();
      x.fillStyle = g; x.fill();
    }
  };
  if (b){
    stroke(a, o.color || "#0969da", true, null);
    const cb = b.filter(v => v!=null);
    if (cb.length){
      let m2 = Math.min(...cb), M2 = Math.max(...cb);
      if (M2-m2 < 1e-9) M2 = m2+1;
      const p2 = (M2-m2)*.12; m2 -= p2; M2 += p2;
      stroke(b, o.color2 || "#bf8700", false,
             v => h-2-(v-m2)/(M2-m2)*(h-4));
    }
  } else {
    stroke(a, o.color || "#0969da", true, null);
  }
  {
    const lab = v => v >= 1000 ? Math.round(v/100)/10+"k"
                               : (Number.isInteger(v) ? String(v) : v.toFixed(1));
    x.fillStyle = "#656d76";
    x.font = "10px ui-monospace,Menlo,Consolas,monospace";
    x.textAlign = "right";
    x.fillText(lab(mx), w-2, 10);
    if (fixed && mn !== mx) x.fillText(lab(mn), w-2, h-3);
  }
}

function drawPrice(cv, pts, o){
  // Day-ahead price chart: step line over [back..fwd], dashed threshold,
  // "now" marker; segments above the threshold drawn in red.
  const w = cv.clientWidth || 300, h = cv.clientHeight || 78;
  const dpr = window.devicePixelRatio || 1;
  if (cv.width !== Math.round(w*dpr) || cv.height !== Math.round(h*dpr)){
    cv.width = Math.round(w*dpr); cv.height = Math.round(h*dpr);
  }
  const x = cv.getContext("2d");
  x.setTransform(dpr,0,0,dpr,0,0);
  x.clearRect(0,0,w,h);
  x.strokeStyle = "#d0d7de";
  x.beginPath(); x.moveTo(0,h-.5); x.lineTo(w,h-.5); x.stroke();
  if (!pts || !pts.length){
    x.fillStyle = "#656d76"; x.font = "11px sans-serif"; x.textAlign = "center";
    x.fillText("no price data yet", w/2, h/2+4);
    return;
  }
  const t0 = pts[0].start, t1 = pts[pts.length-1].end;
  const vals = pts.map(p => p.price).filter(v => v != null);
  if (o.threshold != null) vals.push(o.threshold);
  let mn = Math.min(...vals), mx = Math.max(...vals);
  if (mx-mn < 1e-9) mx = mn+1;
  const pad = (mx-mn)*.12; mn -= pad; mx += pad;
  const X = t => (t-t0)/(t1-t0)*w;
  const Y = v => h-2-(v-mn)/(mx-mn)*(h-4);
  const seg = (from, to, color, width) => {   // [lo,hi) time window in one colour
    x.beginPath(); let on = false, lastX = 0, lastY = h;
    for (const p of pts){
      if (p.end <= from || p.start >= to || p.price == null) continue;
      const y = Y(p.price), xa = Math.max(X(p.start), X(from)), xb = Math.min(X(p.end), X(to));
      if (!on){ x.moveTo(xa,y); on = true; } else if (Math.abs(lastY-y) > .5){ x.lineTo(xa,y); }
      x.lineTo(xb,y); lastX = xb; lastY = y;
    }
    if (!on) return;
    x.strokeStyle = color; x.lineWidth = width; x.stroke();
  };
  if (o.threshold != null){
    // tint the above-threshold slices red, then draw the whole curve blue and
    // re-draw just the hot slices on top in red.
    for (const p of pts){
      if (p.price != null && p.price > o.threshold){
        const xa = X(p.start), xb = X(p.end), y = Y(p.price);
        x.fillStyle = "#cf222e18";
        x.fillRect(xa, y, xb-xa, h-2-y);
      }
    }
  }
  seg(t0-1, t1+1, "#0969da", 1.4);
  if (o.threshold != null) seg(t0-1, t1+1, "#cf222e", 1.6);
  if (o.threshold != null){
    x.setLineDash([4,3]); x.strokeStyle = "#bf8700"; x.lineWidth = 1;
    x.beginPath(); x.moveTo(0,Y(o.threshold)); x.lineTo(w,Y(o.threshold)); x.stroke();
    x.setLineDash([]);
  }
  const nowX = X(o.now);
  if (nowX >= 0 && nowX <= w){
    x.setLineDash([2,2]); x.strokeStyle = "#656d76";
    x.beginPath(); x.moveTo(nowX,0); x.lineTo(nowX,h); x.stroke(); x.setLineDash([]);
    for (const p of pts){
      if (p.price != null && o.now >= p.start && o.now < p.end){
        x.fillStyle = (o.threshold != null && p.price > o.threshold) ? "#cf222e" : "#1a7f37";
        x.beginPath(); x.arc(X(p.end === undefined ? p.start : Math.max(p.start, o.now)),
                             Y(p.price), 3, 0, 7); x.fill();
      }
    }
  }
  const lab = v => v.toFixed(v >= 100 ? 0 : 1);
  x.fillStyle = "#656d76"; x.font = "10px ui-monospace,Menlo,Consolas,monospace";
  x.textAlign = "right";  x.fillText(lab(mx)+" c", w-2, 10);
  x.textAlign = "left";   x.fillText(lab(mn)+" c", 2, h-3);
  x.textAlign = "center";
  x.fillText("now", Math.min(Math.max(nowX, 14), w-14), h-3);
}

function makeSpark(parent, label, key, unit, color, opts){
  opts = opts || {};
  const div = document.createElement("div");
  div.className = "sl" + (opts.extra ? " dual" : "");
  const row = document.createElement("div"); row.className = "row";
  const l = document.createElement("span"); l.textContent = label;
  const val = document.createElement("span"); val.className = "val";
  row.append(l, val);
  const cv = document.createElement("canvas");
  div.append(row, cv); parent.append(div);
  const o = { val, cv, unit, d: opts.d, dec: opts.dec, extra: opts.extra,
              min: opts.min, max: opts.max, color, key };
  o.refresh = ser => {
    const a = ser[o.key] || [];
    o.val.textContent = fmt(a[a.length-1], o.unit, o.d, o.dec);
    spark(cv, a, o.extra ? ser[o.extra] : null,
          { color: o.color, color2: o.extra ? "#bf8700" : null,
            min: o.min, max: o.max });
  };
  return o;
}

function lastv(ser, key){
  const a = ser && ser[key];
  if (!a) return null;
  for (let i = a.length - 1; i >= 0; i--) if (a[i] != null) return a[i];
  return null;
}
function fmtK(v){
  if (v == null) return "";
  v = Math.round(v);
  return v >= 10000 ? (v/1000).toFixed(v>=100000?0:1)+"k" : String(v);
}
function statVal(ser, key, unit, d, dec){
  const v = lastv(ser, key);
  if (v == null) return "\u2013";
  return (dec === false ? String(Math.round(v)) : v.toFixed(d)) + (unit ? " "+unit : "");
}

function setStat(parent, k, v){
  let el = parent.querySelector('.stat[data-k="'+k+'"]');
  if (!el){
    el = document.createElement("div"); el.className = "stat"; el.dataset.k = k;
    el.innerHTML = '<div class="k"></div><div class="v"></div>';
    el.querySelector(".k").textContent = k;
    parent.append(el);
  }
  el.querySelector(".v").textContent = (v==null || v==="") ? "–" : v;
}

function build(d){
  const grid = $("grid"); grid.innerHTML = "";
  const servers = [], gpus = [], host = {};
  d.servers.forEach(s => {
    const c = document.createElement("div"); c.className = "card";
    const h2 = document.createElement("h2");
    h2.innerHTML = '<span class="knd"></span> <span class="sub"></span> <span class="sbad"></span>';
    h2.querySelector(".sub").textContent = s.label;
    c.append(h2);
    const st = document.createElement("div"); st.className = "stats"; c.append(st);
    const rec = { st, knd: h2.querySelector(".knd"), badge: h2.querySelector(".sbad"), sparks: [
      makeSpark(c, "Prompt processing", "s"+s.id+".prompt_wall", "t/s",
                "#0969da", { min: 0 }),
      makeSpark(c, "Generation", "s"+s.id+".gen_wall", "t/s",
                "#1a7f37", { min: 0 })]};
    servers[s.id] = rec; grid.append(c);
  });
  d.gpus.forEach(g => {
    const i = g.id;
    const c = document.createElement("div"); c.className = "card";
    const h2 = document.createElement("h2");
    h2.innerHTML = '<span class="gn"></span> <span class="sub"></span> ' +
                   '<span class="badge link"></span><span class="tbad"></span>';
    h2.querySelector(".gn").textContent = g.name;
    h2.querySelector(".sub").textContent =
      (g.vendor === "amd" ? (g.card || "amd") : "idx " + g.index) + " · " + g.bdf;
    h2.querySelector(".badge.link").textContent = g.link || "?";
    c.append(h2);
    const st = document.createElement("div"); st.className = "stats"; c.append(st);
    const rec = { st, tbad: h2.querySelector(".tbad"), sparks: [
      makeSpark(c, "Temperature", "g"+i+".temp", "°C", "#cf222e", { d: 0, min: 0, max: 90 }),
      makeSpark(c, "GPU utilization", "g"+i+".util", "%", "#0969da", { d: 0, min: 0, max: 100 }),
      makeSpark(c, "VRAM", "g"+i+".mem_pct", "%", "#8250df", { d: 0, min: 0, max: 100 }),
      makeSpark(c, "Power", "g"+i+".power", "W", "#bf8700",
                { min: 0, max: g.power_limit || null }),
      makeSpark(c, "Power limit (cap)", "g"+i+".plimit", "W", "#656d76", { min: 0 }),
      makeSpark(c, "PCIe AER (rate /min + cumulative)", "g"+i+".aer_rate", "/min",
                "#cf222e", { extra: "g"+i+".aer_cum" })]};
    gpus[i] = rec; grid.append(c);
  });
  let priceRec = null;
  if (d.price){
    const c = document.createElement("div"); c.className = "card price";
    const h2 = document.createElement("h2");
    h2.innerHTML = '<span></span> <span class="sub"></span> <span class="pbad"></span>';
    h2.firstChild.textContent = d.price.zone + " day-ahead spot price";
    h2.querySelector(".sub").textContent =
      d.price.include_vat ? "incl VAT" : "excl VAT";
    c.append(h2);
    const st = document.createElement("div"); st.className = "stats"; c.append(st);
    const cv = document.createElement("canvas"); c.append(cv);
    const ev = document.createElement("div"); ev.className = "ev"; c.append(ev);
    priceRec = { st, cv, ev, badge: h2.querySelector(".pbad"), refresh: p => {
      setStat(priceRec.st, "now", p.current == null ? "\u2013"
              : p.current.toFixed(2)+" c/kWh");
      setStat(priceRec.st, "threshold", p.threshold == null ? "\u2013"
              : p.threshold.toFixed(2)+" c/kWh");
      drawPrice(cv, p.curve, { threshold: p.threshold, now: p.now });
      let b = "";
      if (p.throttled === true) b = '<span class="badge hot">THROTTLED</span>';
      else if (p.throttled === false) b = '<span class="badge up">OK</span>';
      priceRec.badge.innerHTML = b;
      const lines = (p.events || []).slice(-3);
      ev.textContent = p.err && !lines.length ? p.err : lines.join("\n");
      ev.style.whiteSpace = "pre-line";
    }};
    grid.append(c);
  }
  {
    const c = document.createElement("div"); c.className = "card";
    const h2 = document.createElement("h2");
    h2.innerHTML = 'Host <span class="sub hs"></span>';
    c.append(h2);
    const st = document.createElement("div"); st.className = "stats"; c.append(st);
    host.rec = { st, hs: h2.querySelector(".hs"), sparks: [
      makeSpark(c, "CPU", "h.cpu", "%", "#0969da", { min: 0, max: 100 }),
      makeSpark(c, "RAM used", "h.ram", "GB", "#1a7f37"),
      makeSpark(c, "Network (rx + tx)", "h.rx", "MB/s", "#0e7490", { extra: "h.tx" })]};
    grid.append(c);
  }
  refs = { servers, gpus, host: host.rec, price: priceRec };
}

function update(d){
  lastData = d;
  if (!refs) build(d);
  const ser = d.series;
  const upN = d.servers.filter(s => s.up).length;
  $("h-badge").innerHTML = upN === d.servers.length
    ? '<span class="badge up">'+d.servers.length+' server'+(d.servers.length>1?"s":"")+' up</span>'
    : '<span class="badge down">'+upN+"/"+d.servers.length+' up</span>';
  $("h-meta").textContent = "poll "+d.interval+"s · "+d.points+" pts shown · host up "+
    fmtUptime(ser["h.uptime"] ? ser["h.uptime"][ser["h.uptime"].length-1] : null)+
    " · monitor up "+fmtUptime(d.monitor_uptime);
  d.servers.forEach(s => {
    const r = refs.servers[s.id]; if (!r) return;
    r.knd.textContent = s.kind === "strata" ? "Strata"
                        : s.kind === "llama" ? "llama-server" : "endpoint";
    r.badge.innerHTML = s.up
      ? '<span class="badge up">UP</span>'+(s.model ? ' <span class="sub">'+esc(s.model)+"</span>" : "")
      : '<span class="badge down">DOWN</span>';
    const L = k => lastv(ser, "s"+s.id+"."+k);
    const num = (v, u, d) => v == null ? "\u2013" : v.toFixed(d) + (u ? " "+u : "");
    const intg = v => v == null ? "\u2013" : String(Math.round(v));
    setStat(r.st, "requests", intg(L("req_proc")));
    setStat(r.st, "queued", intg(L("req_def")));
    setStat(r.st, "max seq", L("seq_max") == null ? "\u2013" : fmtK(L("seq_max"))+" tok");
    setStat(r.st, "spec accept", num(L("spec_acc"), "%", 0));
    setStat(r.st, "cache hit", num(L("cache_pct"), "%", 0));
    const pt = L("prompt_tps"), ptok = L("last_prompt_tok");
    setStat(r.st, "prefill", pt == null ? "\u2013"
      : pt.toFixed(0)+" t/s"+(ptok != null ? " ("+fmtK(ptok)+" tok)" : ""));
    const gt = L("gen_tps"), gtok = L("last_decode_tok");
    setStat(r.st, "decode", gt == null ? "\u2013"
      : gt.toFixed(0)+" t/s"+(gtok != null ? " ("+fmtK(gtok)+" tok)" : ""));
  });
  d.gpus.forEach(g => {
    const r = refs.gpus[g.id]; if (!r) return;
    const i = g.id;
    let tb = "";
    if (d.throttle){
      const dev = d.throttle.devices.find(x => x.uuid && g.uuid && x.uuid === g.uuid);
      if (dev && dev.state === "high")
        tb = dev.error ? '<span class="badge hot">cap error</span>'
                       : '<span class="badge warn">capped</span>';
      else if (dev && dev.error)
        tb = '<span class="badge hot">throttle error</span>';
    }
    r.tbad.innerHTML = tb;
    setStat(r.st, "temp", statVal(ser, "g"+i+".temp", "°C", 0));
    setStat(r.st, "util", statVal(ser, "g"+i+".util", "%", 0));
    setStat(r.st, "vram", g.mem_total
      ? statVal(ser, "g"+i+".mem_used", " / "+(g.mem_total/1024).toFixed(0)+" GB")
      : "–");
    setStat(r.st, "power", statVal(ser, "g"+i+".power", "W")+
      (g.power_limit ? " / "+g.power_limit+" W" : ""));
    setStat(r.st, "aer total", statVal(ser, "g"+i+".aer_cum", null, 0, false));
    setStat(r.st, "aer/min", statVal(ser, "g"+i+".aer_rate", null, 1));
  });
  {
    const r = refs.host;
    r.hs.textContent = d.host.cores+" cores";
    const l1 = ser["h.load1"] && ser["h.load1"][ser["h.load1"].length-1];
    const l5 = ser["h.load5"] && ser["h.load5"][ser["h.load5"].length-1];
    const l15 = ser["h.load15"] && ser["h.load15"][ser["h.load15"].length-1];
    setStat(r.st, "load 1/5/15",
      (l1!=null) ? l1.toFixed(1)+" / "+l5.toFixed(1)+" / "+l15.toFixed(1) : "–");
    setStat(r.st, "ram", d.host.mem_total
      ? statVal(ser, "h.ram", " / "+d.host.mem_total.toFixed(0)+" GB") : "–");
    setStat(r.st, "swap", d.host.swap_total
      ? statVal(ser, "h.swap", " / "+d.host.swap_total.toFixed(0)+" GB") : "–");
    setStat(r.st, "uptime", fmtUptime(
      ser["h.uptime"] ? ser["h.uptime"][ser["h.uptime"].length-1] : null));
  }
  const all = [...refs.servers, ...Object.values(refs.gpus)].filter(Boolean);
  all.push(refs.host);
  for (const r of all) for (const sp of r.sparks) sp.refresh(ser);
  if (refs.price && d.price){
    refs.price.refresh(Object.assign({}, d.price, {
      now: d.now, events: d.throttle ? d.throttle.events : null }));
  }
}

async function tick(){
  try {
    const r = await fetch("/api/stats", { cache: "no-store" });
    const d = await r.json();
    update(d);
    if (timer) clearInterval(timer);
    timer = setInterval(tick, Math.round(d.interval*1000));
  } catch (e) {
    if (timer) clearInterval(timer);
    timer = setInterval(tick, 2000);
  }
}
window.addEventListener("resize", () => { if (refs && lastData) update(lastData); });
tick();
</script></body></html>
'''


class Handler(BaseHTTPRequestHandler):
    server_version = 'llama-mon/1.0'
    state = None  # set in main()

    def _send(self, code, body, ctype):
        data = body if isinstance(body, bytes) else body.encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = self.path.split('?', 1)[0]
        if path == '/':
            self._send(200, PAGE, 'text/html; charset=utf-8')
        elif path == '/api/stats':
            self._send(200, json.dumps(self.state.api()), 'application/json')
        elif path == '/healthz':
            self._send(200, 'ok\n', 'text/plain')
        else:
            self._send(404, 'not found\n', 'text/plain')

    def log_message(self, fmt, *args):
        if self.path not in ('/', '/api/stats', '/healthz'):
            super().log_message(fmt, *args)


def main():
    import argparse
    ap = argparse.ArgumentParser(description='llama-mon dashboard')
    ap.add_argument('--config', default=None,
                    help='JSON config file (default: $MON_CONFIG, then '
                         '/etc/llama-mon/config.json when present)')
    args = ap.parse_args()
    cfg = load_config(args.config)
    host, port = cfg['host'], int(cfg['port'])

    state = State(cfg)
    Handler.state = state

    if state.price.enabled:
        try:                       # first fetch synchronously so the dashboard
            state.price.ensure(time.time())
        except Exception as e:
            print('price: initial fetch failed: %r' % e, flush=True)

        def price_loop():
            every = max(30.0, float(state.price.cfg.get('fetch_interval', 60)))
            while True:
                try:
                    state.price.ensure(time.time())
                except Exception as e:
                    print('price: refresh error: %r' % e, flush=True)
                time.sleep(every)
        threading.Thread(target=price_loop, daemon=True).start()

    if state.throttle.enabled:
        def throttle_loop():
            while True:
                try:
                    state.throttle.check(time.time())
                except Exception as e:
                    print('throttle: check error: %r' % e, flush=True)
                time.sleep(state.throttle.interval)
        if not state.throttle.devices:
            print('throttle: enabled but no devices configured - nothing to do',
                  flush=True)
        else:
            threading.Thread(target=throttle_loop, daemon=True).start()

    def loop():
        while True:
            t0 = time.time()
            try:
                state.collect()
            except Exception as e:
                print('collect error: %r' % e, flush=True)
            time.sleep(max(0.0, state.interval - (time.time() - t0)))

    threading.Thread(target=loop, daemon=True).start()
    print('llama-mon: http://%s:%d/  servers=%s interval=%.1fs window=%ds gpus=%s '
          'config=%s price=%s throttle=%s'
          % (host, port, ','.join(cfg['urls']), state.interval, int(cfg['window']),
             cfg['gpus'] or 'all', cfg.get('_config_file') or '(defaults+env)',
             'on' if state.price.enabled else 'off',
             ('on %.1fW-threshold-devs=%d' % (state.throttle.threshold,
                                              len(state.throttle.devices))
              if state.throttle.enabled else 'off')), flush=True)
    ThreadingHTTPServer((host, port), Handler).serve_forever()


if __name__ == '__main__':
    main()

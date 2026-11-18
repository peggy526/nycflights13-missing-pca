"""Minimal reader for R .rda files (XDR serialization), enough for nycflights13.

R is not available in the build environment, so this reads the *official*
nycflights13 .rda files (github.com/tidyverse/nycflights13/data) directly,
giving exactly the same data you get from data("flights", package="nycflights13").
Supports: logical/integer/double/character vectors, lists, pairlists,
symbols, references, attributes and the common ALTREP wrappers.
"""
import bz2, gzip, lzma, struct
import numpy as np
import pandas as pd

NA_INT = -2147483648


def _decompress(raw):
    if raw[:2] == b"\x1f\x8b":
        return gzip.decompress(raw)
    if raw[:3] == b"BZh":
        return bz2.decompress(raw)
    if raw[:6] == b"\xfd7zXZ\x00":
        return lzma.decompress(raw)
    return raw


class RObj:
    def __init__(self, value, attrs=None):
        self.value, self.attrs = value, attrs or {}


class _Parser:
    def __init__(self, buf):
        self.b, self.p, self.refs = buf, 0, []

    def int(self):
        v = struct.unpack_from(">i", self.b, self.p)[0]
        self.p += 4
        return v

    def ints(self, n):
        a = np.frombuffer(self.b, ">i4", n, self.p).astype(np.int64)
        self.p += 4 * n
        return a

    def dbls(self, n):
        a = np.frombuffer(self.b, ">f8", n, self.p).astype(np.float64)
        self.p += 8 * n
        return a

    def length(self):
        n = self.int()
        if n == -1:  # long vector
            hi, lo = self.int(), self.int()
            n = (hi << 32) + lo
        return n

    def attrs_of(self, has_attr):
        if not has_attr:
            return {}
        a = self.item()
        return a.value if isinstance(a, RObj) and isinstance(a.value, dict) else {}

    def item(self):
        flags = self.int()
        t = flags & 0xFF
        has_obj_attr = bool(flags & (1 << 9))
        has_tag = bool(flags & (1 << 10))
        if t in (254, 242, 253, 241, 251, 247, 252):  # NULL / special envs
            return None
        if t == 255:  # REFSXP
            idx = flags >> 8
            if idx == 0:
                idx = self.int()
            return self.refs[idx - 1]
        if t == 1:  # SYMSXP
            s = self.item()
            self.refs.append(s)
            return s
        if t == 9:  # CHARSXP
            n = self.int()
            if n == -1:
                return None
            s = self.b[self.p:self.p + n].decode("utf-8", "replace")
            self.p += n
            return s
        if t in (2, 6, 17):  # LISTSXP / LANGSXP / DOTSXP -> dict(tag: value)
            out, keys = {}, 0
            while True:
                attrs = self.item() if has_obj_attr else None
                tag = self.item() if has_tag else None
                car = self.item()
                key = tag if tag is not None else f"_{keys}"
                keys += 1
                out[key] = car
                flags = self.int()
                t = flags & 0xFF
                has_obj_attr = bool(flags & (1 << 9))
                has_tag = bool(flags & (1 << 10))
                if t == 254:
                    break
                if t not in (2, 6, 17):  # improper list; parse remaining item
                    self.p -= 4
                    out[f"_{keys}"] = self.item()
                    break
            return RObj(out)
        if t in (10, 13):  # LGLSXP / INTSXP
            n = self.length()
            v = self.ints(n)
            return RObj(v, self.attrs_of(has_obj_attr)) if True else None
        if t == 14:  # REALSXP
            n = self.length()
            v = self.dbls(n)
            return RObj(v, self.attrs_of(has_obj_attr))
        if t == 16:  # STRSXP
            n = self.length()
            v = [self.item() for _ in range(n)]
            return RObj(v, self.attrs_of(has_obj_attr))
        if t in (19, 20):  # VECSXP / EXPRSXP
            n = self.length()
            v = [self.item() for _ in range(n)]
            return RObj(v, self.attrs_of(has_obj_attr))
        if t == 238:  # ALTREP_SXP
            info = self.item()
            state = self.item()
            attr = self.item()
            cls = info.value.get("_0") if isinstance(info, RObj) else None
            return self._altrep(cls, state, attr)
        if t == 4:  # ENVSXP
            self.int()  # locked
            env = RObj({})
            self.refs.append(env)
            self.item(); self.item(); self.item(); self.item()
            return env
        raise ValueError(f"Unsupported SEXP type {t} at byte {self.p}")

    def _altrep(self, cls, state, attr):
        attrs = attr.value if isinstance(attr, RObj) and isinstance(attr.value, dict) else {}
        if cls in ("compact_intseq", "compact_realseq"):
            n, start, step = [float(x) for x in state.value]
            return RObj(start + step * np.arange(int(n)), attrs)
        if cls in ("wrap_integer", "wrap_real", "wrap_string", "wrap_logical"):
            inner = state.value["_0"] if isinstance(state.value, dict) else state.value[0]
            return RObj(inner.value, attrs or inner.attrs)
        if cls == "deferred_string":
            arg = state.value["_0"] if isinstance(state.value, dict) else state.value[0]
            return RObj([None if (isinstance(x, float) and np.isnan(x)) else str(int(x)) if float(x).is_integer() else str(x)
                         for x in np.asarray(arg.value, dtype=float)], attrs)
        raise ValueError(f"Unsupported ALTREP class {cls}")


def _to_series(name, obj):
    v, attrs = obj.value, obj.attrs
    cls = attrs.get("class").value if attrs.get("class") is not None else []
    if isinstance(v, list):
        return pd.Series(v, name=name, dtype="object")
    if v.dtype == np.int64:
        s = pd.Series(v, name=name).astype("float64")
        s[v == NA_INT] = np.nan
        if "factor" in cls:
            levels = attrs["levels"].value
            return pd.Series(pd.Categorical.from_codes(np.where(v == NA_INT, -1, v - 1), levels), name=name)
        if (v != NA_INT).all():
            s = s.astype("int64")
        return s
    s = pd.Series(v, name=name)  # R NA_real_ is a NaN payload -> NaN here
    if "POSIXct" in cls:
        tz = attrs.get("tzone")
        tz = tz.value[0] if tz is not None and tz.value and tz.value[0] else "UTC"
        s = pd.to_datetime(s, unit="s", utc=True).dt.tz_convert(tz)
    return s


def read_rda(path):
    """Return {object_name: pandas.DataFrame} for every data frame in an .rda file."""
    with open(path, "rb") as fh:
        buf = _decompress(fh.read())
    if buf[:5] not in (b"RDX2\n", b"RDX3\n"):
        raise ValueError("not an XDR .rda file")
    p = _Parser(buf)
    p.p = 5
    if buf[5:7] != b"X\n":
        raise ValueError("only XDR format is supported")
    p.p = 7
    version = p.int(); p.int(); p.int()
    if version == 3:
        n = p.int(); p.p += n
    top = p.item()
    out = {}
    for name, obj in top.value.items():
        cols = obj.attrs["names"].value
        out[name] = pd.DataFrame({c: _to_series(c, col) for c, col in zip(cols, obj.value)})
    return out

"""Reader for the IceCube LLHFit protobuf result files (``<name>.pb.gz``).

The C++ side (libraries/results/IceCube/ICWriteResultsProto.cpp) serialises a
single ``result.ic.proto.FitResult`` message and gzips it.  This module decodes
that file with nothing but the standard library -- the wire format of the
schema in libraries/results/IceCube/proto/ic_result.proto is hard-coded below,
so no ``protobuf`` runtime and no generated ``*_pb2.py`` are required.

Usage:

    from ic_result import load

    res = load("Output.pb.gz")
    print(res["chi2"], res["converged"])
    print(res["parameters"]["AstroNorm"]["value"])
    for sample in res["samples"]:
        print(sample["name"], sample["pred_total"], len(sample["prediction"]))

Command line:

    python ic_result.py Output.pb.gz          # human-readable summary
    python ic_result.py --json Output.pb.gz   # full dump as JSON
"""

from __future__ import annotations

import gzip
import json
import struct
import sys

__all__ = ["load", "loads", "SCHEMA"]

_VARINT = 0
_FIXED64 = 1
_LEN = 2
_FIXED32 = 5

# field number -> (name, kind, repeated)
# kind is one of "bool", "double", "string", "uint32", "uint64", or a message
# name present in SCHEMA, or ("map", key_kind, value_kind).
SCHEMA = {
    "Parameter": {
        1: ("value", "double", False),
        2: ("error", "double", False),
    },
    "Axis": {
        1: ("kind", "string", False),
        2: ("low", "double", False),
        3: ("high", "double", False),
        4: ("n_bins", "uint32", False),
    },
    "Component": {
        1: ("name", "string", False),
        2: ("total", "double", False),
        3: ("bins", "double", True),
    },
    "Sample": {
        1: ("name", "string", False),
        2: ("components", "string", True),
        3: ("livetime", "double", False),
        4: ("total_bins", "uint64", False),
        5: ("axes", "Axis", True),
        6: ("data", "double", True),
        7: ("prediction", "double", True),
        8: ("data_total", "double", False),
        9: ("pred_total", "double", False),
        10: ("components_breakdown", "Component", True),
    },
    "FitResult": {
        1: ("converged", "bool", False),
        2: ("chi2", "double", False),
        3: ("edm", "double", False),
        4: ("fit_duration", "double", False),
        5: ("parameters", ("map", "string", "Parameter"), False),
        6: ("samples", "Sample", True),
        7: ("data_total", "double", False),
        8: ("pred_total", "double", False),
        9: ("likelihood", "string", False),
    },
}

_SCALAR_DEFAULTS = {
    "bool": False,
    "double": 0.0,
    "string": "",
    "uint32": 0,
    "uint64": 0,
}


def _read_varint(buf, pos):
    result = 0
    shift = 0
    while True:
        if pos >= len(buf):
            raise ValueError("truncated varint")
        byte = buf[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, pos
        shift += 7
        if shift > 63:
            raise ValueError("varint too long")


def _iter_fields(buf):
    """Yield (field_number, wire_type, payload) for every field in ``buf``.

    ``payload`` is the raw bytes for length-delimited/fixed fields and the
    decoded integer for varints.
    """
    pos = 0
    end = len(buf)
    while pos < end:
        key, pos = _read_varint(buf, pos)
        field, wire = key >> 3, key & 0x07
        if wire == _VARINT:
            value, pos = _read_varint(buf, pos)
        elif wire == _FIXED64:
            value = buf[pos:pos + 8]
            pos += 8
        elif wire == _LEN:
            length, pos = _read_varint(buf, pos)
            value = buf[pos:pos + length]
            pos += length
        elif wire == _FIXED32:
            value = buf[pos:pos + 4]
            pos += 4
        else:
            raise ValueError("unsupported wire type %d (field %d)" % (wire, field))
        yield field, wire, value


def _unpack_doubles(raw):
    if len(raw) % 8:
        raise ValueError("packed double block is not a multiple of 8 bytes")
    return list(struct.unpack("<%dd" % (len(raw) // 8), raw))


def _decode_scalar(kind, wire, value):
    if kind == "double":
        if wire == _FIXED64:
            return struct.unpack("<d", value)[0]
        raise ValueError("expected fixed64 for double, got wire %d" % wire)
    if kind == "bool":
        return bool(value)
    if kind in ("uint32", "uint64"):
        return value
    if kind == "string":
        return value.decode("utf-8")
    raise ValueError("unknown scalar kind %r" % (kind,))


def _decode_map_entry(key_kind, value_kind, raw):
    key = _SCALAR_DEFAULTS.get(key_kind)
    value = {} if value_kind in SCHEMA else _SCALAR_DEFAULTS.get(value_kind)
    for field, wire, payload in _iter_fields(raw):
        if field == 1:
            key = _decode_scalar(key_kind, wire, payload)
        elif field == 2:
            if value_kind in SCHEMA:
                value = _decode_message(value_kind, payload)
            else:
                value = _decode_scalar(value_kind, wire, payload)
    return key, value


def _decode_message(msg_name, buf):
    fields = SCHEMA[msg_name]

    # proto3 defaults: absent fields still show up, so downstream code never
    # has to guard with .get().
    out = {}
    for name, kind, repeated in fields.values():
        if repeated:
            out[name] = []
        elif isinstance(kind, tuple):
            out[name] = {}
        elif kind in SCHEMA:
            out[name] = None
        else:
            out[name] = _SCALAR_DEFAULTS[kind]

    for field, wire, payload in _iter_fields(buf):
        spec = fields.get(field)
        if spec is None:
            continue  # unknown field from a newer schema -- skip it
        name, kind, repeated = spec

        if isinstance(kind, tuple):  # map<key, value>
            key, value = _decode_map_entry(kind[1], kind[2], payload)
            out[name][key] = value
        elif kind in SCHEMA:
            sub = _decode_message(kind, payload)
            if repeated:
                out[name].append(sub)
            else:
                out[name] = sub
        elif repeated and kind == "double" and wire == _LEN:
            out[name].extend(_unpack_doubles(payload))  # packed
        elif repeated:
            out[name].append(_decode_scalar(kind, wire, payload))
        else:
            out[name] = _decode_scalar(kind, wire, payload)

    return out


def loads(data):
    """Decode raw bytes (gzipped or plain) of a FitResult message."""
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    return _decode_message("FitResult", data)


def load(path):
    """Read and decode a ``*.pb.gz`` (or uncompressed ``*.pb``) fit result."""
    with open(path, "rb") as handle:
        return loads(handle.read())


def _summary(res):
    lines = [
        "likelihood   : %s" % res["likelihood"],
        "converged    : %s" % res["converged"],
        "chi2         : %.8g" % res["chi2"],
        "edm          : %.8g" % res["edm"],
        "fit_duration : %.4g s" % res["fit_duration"],
        "data / pred  : %.6g / %.6g" % (res["data_total"], res["pred_total"]),
        "",
        "parameters (%d):" % len(res["parameters"]),
    ]
    for name in sorted(res["parameters"]):
        par = res["parameters"][name]
        lines.append("  %-28s %14.8g +- %-14.8g" % (name, par["value"], par["error"]))

    lines.append("")
    lines.append("samples (%d):" % len(res["samples"]))
    for sample in res["samples"]:
        axes = ", ".join(
            "%s[%g, %g, %d]" % (a["kind"], a["low"], a["high"], a["n_bins"])
            for a in sample["axes"]
        )
        lines.append("  %s  bins=%d  livetime=%g" % (sample["name"], sample["total_bins"], sample["livetime"]))
        lines.append("    axes: %s" % axes)
        lines.append("    data=%.6g  pred=%.6g" % (sample["data_total"], sample["pred_total"]))
        for comp in sample["components_breakdown"]:
            lines.append("      %-20s %.6g" % (comp["name"], comp["total"]))
    return "\n".join(lines)


def main(argv):
    args = [a for a in argv[1:] if not a.startswith("-")]
    if len(args) != 1:
        print(__doc__.strip().splitlines()[0])
        print("usage: ic_result.py [--json] <result.pb.gz>", file=sys.stderr)
        return 2
    res = load(args[0])
    if "--json" in argv[1:]:
        json.dump(res, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        print(_summary(res))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

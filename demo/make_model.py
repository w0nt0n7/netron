#!/usr/bin/env python3
"""Generate a tiny, valid TensorFlow GraphDef (.pb) for the Netron demo.

No TensorFlow dependency required - the GraphDef protobuf is hand-encoded.
GraphDef  { repeated NodeDef node = 1; }
NodeDef   { string name = 1; string op = 2; repeated string input = 3; }
"""

import os


def varint(value):
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def length_delimited(field, data):
    tag = (field << 3) | 2  # wire type 2 (length-delimited)
    return varint(tag) + varint(len(data)) + data


def node(name, op, inputs=()):
    payload = length_delimited(1, name.encode("utf-8"))
    payload += length_delimited(2, op.encode("utf-8"))
    for item in inputs:
        payload += length_delimited(3, item.encode("utf-8"))
    return payload


nodes = [
    node("input", "Placeholder"),
    node("conv1/weights", "Const"),
    node("conv1", "Conv2D", ["input", "conv1/weights"]),
    node("conv1/relu", "Relu", ["conv1"]),
    node("pool1", "MaxPool", ["conv1/relu"]),
    node("fc/weights", "Const"),
    node("fc/matmul", "MatMul", ["pool1", "fc/weights"]),
    node("fc/bias", "Const"),
    node("logits", "BiasAdd", ["fc/matmul", "fc/bias"]),
    node("output", "Softmax", ["logits"]),
]

graph_def = b"".join(length_delimited(1, n) for n in nodes)

target = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pb")
with open(target, "wb") as handle:
    handle.write(graph_def)

print(f"wrote {target} ({len(graph_def)} bytes, {len(nodes)} nodes)")

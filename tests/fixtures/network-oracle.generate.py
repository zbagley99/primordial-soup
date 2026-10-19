"""Independent double-precision equations; no application code or RNG imports."""
import json
import math
from pathlib import Path


def matrix(rows, cols, phase):
    return [[math.sin((i * cols + j + phase) * 0.71) * 0.7
             for j in range(cols)] for i in range(rows)]


def vector(n, phase):
    return [math.cos((i + phase) * 0.43) * 0.3 for i in range(n)]


def dot(w, x):
    return [sum(a * b for a, b in zip(row, x)) for row in w]


def affine(w, x, bias):
    return [a + b for a, b in zip(dot(w, x), bias)]


def flatten(*parts):
    return [v for part in parts for row in part
            for v in (row if isinstance(row, list) else [row])]


inputs = [[0.25, -0.75, 1.5, -0.125], [-1.25, 0.5, 0.125, 0.75],
          [0.0, 0.25, -0.5, 1.0], [2.0, -1.0, 0.5, -0.25]]
wz, uz, bz = matrix(4, 4, 1), matrix(4, 4, 19), vector(4, 2)
wr, ur, br = matrix(4, 4, 37), matrix(4, 4, 53), vector(4, 5)
wn, un, bn = matrix(4, 4, 71), matrix(4, 4, 89), vector(4, 8)
wo, bo = matrix(4, 4, 107), vector(4, 11)
h = [0.375, -0.625, 0.125, -0.25]
initial = h[:]
steps = []
for x in inputs:
    z = [1 / (1 + math.exp(-(a + b)))
         for a, b in zip(affine(wz, x, bz), dot(uz, h))]
    r = [1 / (1 + math.exp(-(a + b)))
         for a, b in zip(affine(wr, x, br), dot(ur, h))]
    candidate = [math.tanh(a + b) for a, b in
                 zip(affine(wn, x, bn), dot(un, [a * b for a, b in zip(r, h)]))]
    h = [a * b + (1 - a) * c for a, b, c in zip(z, h, candidate)]
    steps.append(dict(input=x, hidden=h[:], output=affine(wo, h, bo)))
gru = dict(mode="gru", weights=flatten(wz, uz, bz, wr, ur, br, wn, un, bn, wo, bo),
           initial_hidden=initial, steps=steps)
w1, b1, w2, b2 = matrix(8, 4, 3), vector(8, 4), matrix(4, 8, 41), vector(4, 13)
mlp = dict(mode="mlp", weights=flatten(w1, b1, w2, b2), initial_hidden=[0] * 4,
           steps=[dict(input=x, output=affine(w2, [math.tanh(v) for v in affine(w1, x, b1)], b2))
                  for x in inputs])
Path(__file__).with_name("network-oracle.fixture.json").write_text(
    json.dumps([mlp, gru], indent=2) + "\n")

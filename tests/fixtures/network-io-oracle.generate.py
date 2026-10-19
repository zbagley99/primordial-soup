"""Independent double precision revision-3 equations; does not import Rust."""
import json
import math
from pathlib import Path


def matrix(rows, cols, phase):
    return [[(((r * cols + c + phase) * 7) % 23 - 11) / 19
             for c in range(cols)] for r in range(rows)]


def bias(phase):
    return [((i * 3 + phase) % 7 - 3) / 11 for i in range(4)]


def affine(w, x, b):
    return [v + sum(a * c for a, c in zip(row, x))
            for row, v in zip(w, b)]


def flat(*parts):
    return [v for part in parts for row in part
            for v in (row if isinstance(row, list) else [row])]


inputs = [[.2, -.3, .1, .4, .5, .7, .2],
          [-.1, .2, .4, -.3, .7, .5, .3],
          [0, 0, 0, 0, .9, .9, 0], [0, 0, 0, 0, 0, 0, 0]]
w1, b1, w2, b2 = matrix(8, 7, 1), [i / 23 - .15 for i in range(8)], matrix(4, 8, 9), bias(2)
mlp = dict(mode="mlp", weights=flat(w1, b1, w2, b2), initial_hidden=[0]*4,
           steps=[dict(input=x, output=[2*math.tanh(v) for v in affine(
               w2, [math.tanh(a) for a in affine(w1, x, b1)], b2)])
                  for x in inputs])
wz, uz, bz = matrix(4, 7, 2), matrix(4, 4, 7), bias(1)
wr, ur, br = matrix(4, 7, 3), matrix(4, 4, 9), bias(2)
wn, un, bn = matrix(4, 7, 5), matrix(4, 4, 11), bias(3)
wo, bo = matrix(4, 4, 13), bias(4)
h = [.3, -.2, .5, -.1]
initial = h[:]
steps = []
for x in inputs:
    z = [1/(1+math.exp(-a-b)) for a, b in zip(affine(wz, x, bz), affine(uz, h, [0]*4))]
    r = [1/(1+math.exp(-a-b)) for a, b in zip(affine(wr, x, br), affine(ur, h, [0]*4))]
    n = [math.tanh(a+b) for a, b in zip(affine(wn, x, bn), affine(un, [a*b for a, b in zip(r, h)], [0]*4))]
    h = [a*b+(1-a)*c for a, b, c in zip(z, h, n)]
    steps.append(dict(input=x, hidden=h[:], output=[2*math.tanh(v) for v in affine(wo, h, bo)]))
gru = dict(mode="gru", weights=flat(wz, uz, bz, wr, ur, br, wn, un, bn, wo, bo), initial_hidden=initial, steps=steps)
Path(__file__).with_name("network-io-oracle.fixture.json").write_text(json.dumps([mlp, gru], indent=2)+"\n")

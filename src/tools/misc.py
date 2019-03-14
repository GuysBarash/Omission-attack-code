import json
import numpy as np


def to_json(sr, path):
    js = json.dumps(json.loads(sr.to_json()), indent=2)
    with open(path, 'w+') as ffile:
        ffile.write(js)

    return True


def dist(p1, p2):
    xs = (p1[0] - p2[0]) ** 2
    ys = (p1[1] - p2[1]) ** 2
    dist = np.sqrt(xs + ys)
    return dist


def safe_div(x, y, default=0.0):
    if y == 0:
        return default
    else:
        return float(x) / float(y)

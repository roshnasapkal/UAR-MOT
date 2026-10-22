"""Physical partial matchings and a dependency-free augmented Hungarian solver."""
from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True, order=True)
class Assignment:
    matches: tuple
    unmatched_tracks: tuple
    unmatched_detections: tuple

    @classmethod
    def from_matches(cls, n_tracks, n_detections, matches):
        pairs = tuple(sorted(tuple(pair) for pair in matches))
        if any(len(pair) != 2 or any(type(x) is not int for x in pair)
               for pair in pairs):
            raise ValueError("Match indices must be integer pairs")
        rows, cols = [i for i, _ in pairs], [j for _, j in pairs]
        if (len(set(rows)) != len(rows) or len(set(cols)) != len(cols)
                or any(not 0 <= i < n_tracks for i in rows)
                or any(not 0 <= j < n_detections for j in cols)):
            raise ValueError("Matches must be in range and one-to-one")
        return cls(pairs, tuple(i for i in range(n_tracks) if i not in rows),
                   tuple(j for j in range(n_detections) if j not in cols))


def finite_values(values, name):
    result = tuple(float(x) for x in values)
    if not all(isfinite(x) for x in result):
        raise ValueError(name + " must be finite")
    return result


def integer_scale(values):
    """Exactly embed finite binary floats (or integers) on one integer lattice.

    Float denominators are powers of two, so their maximum is a common
    denominator. Python integers do not round or overflow during optimization.
    """
    ratios = [x.as_integer_ratio() if isinstance(x, float) else (x, 1)
              for x in values]
    scale = max((q for _, q in ratios), default=1)
    return [p * (scale // q) for p, q in ratios], scale


def hungarian(costs):
    """Minimum square assignment; None edges are absent, never big-M costs.

    Returns row-to-column indices, or None if no perfect assignment exists.
    Integer costs are exact. Finite float costs are scaled exactly first.
    Strict comparisons and ascending scans resolve exact ties deterministically.
    """
    n = len(costs)
    if any(len(row) != n for row in costs):
        raise ValueError("Expected square costs")
    edges = [x for row in costs for x in row if x is not None]
    if any(type(x) not in (int, float) or
           (isinstance(x, float) and not isfinite(x)) for x in edges):
        raise ValueError("Hungarian costs must be finite floats or integers")
    scaled, _ = integer_scale(edges)
    edge_iter = iter(scaled)
    costs = [[next(edge_iter) if x is not None else None for x in row]
             for row in costs]
    u, v = [0] * (n + 1), [0] * (n + 1)
    p, way = [0] * (n + 1), [0] * (n + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        best, used = [None] * (n + 1), [False] * (n + 1)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], None, 0
            for j in range(1, n + 1):
                if used[j]:
                    continue
                edge = costs[i0 - 1][j - 1]
                if edge is not None:
                    reduced = edge - u[i0] - v[j]
                    if best[j] is None or reduced < best[j]:
                        best[j], way[j] = reduced, j0
                if best[j] is not None and (delta is None or best[j] < delta):
                    delta, j1 = best[j], j
            if delta is None:
                return None
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                elif best[j] is not None:
                    best[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    result = [0] * n
    for j in range(1, n + 1):
        result[p[j] - 1] = j - 1
    assert all(costs[i][j] is not None for i, j in enumerate(result))
    return tuple(result)


def solve_partial(costs, mask, track_penalties, detection_penalties, excluded=None):
    """Exclude one selected physical edge ('match', 'track', or 'detection')."""
    m, n = len(track_penalties), len(detection_penalties)
    matrix = [[None] * (m + n) for _ in range(m + n)]
    for i in range(m):
        for j in range(n):
            if mask[i][j] and excluded != ('match', i, j):
                matrix[i][j] = costs[i][j]
        if excluded != ('track', i):
            matrix[i][n + i] = track_penalties[i]
    for j in range(n):
        if excluded != ('detection', j):
            matrix[m + j][j] = detection_penalties[j]
        for i in range(m):
            matrix[m + j][n + i] = 0
    solution = hungarian(matrix)
    if solution is None:
        return None
    return Assignment.from_matches(m, n, ((i, solution[i]) for i in range(m)
                                           if solution[i] < n))

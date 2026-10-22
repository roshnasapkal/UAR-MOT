"""Global box-uncertainty certificate relative to a modeled feasible set."""
from dataclasses import dataclass
from fractions import Fraction
from .assignment import Assignment, finite_values, integer_scale
from .competitor import strongest_competitor
from .disagreement import Disagreement, disagreement


@dataclass(frozen=True)
class StabilityResult:
    nominal: Assignment
    competitor: Assignment | None
    gamma: float | None
    margin_status: str
    tolerance: float
    disagreement: Disagreement
    exact_gamma: Fraction | None
    exact_tolerance: Fraction


def analyze_stability(costs, feasible, rho, track_penalties, detection_penalties,
                      nominal, *, atol=1e-10, rtol=1e-12):
    """Analyze caller-supplied X*, including feasible but nonoptimal X*.

    Penalty lengths define the shape, including zero-row matrices. rho applies
    only to feasible real edges; fixed unmatched penalties have no uncertainty.
    Positive Gamma certifies modeled assignment stability, not tracking truth.
    This model does not establish compatibility with ByteTrack's matching policy.
    Input values are normalized to finite Python floats, then optimized exactly
    on their binary integer lattice. exact_gamma/exact_tolerance are Fractions;
    gamma/tolerance are float projections, never used for classification.
    """
    tp = finite_values(track_penalties, "track penalties")
    dp = finite_values(detection_penalties, "detection penalties")
    m, n = len(tp), len(dp)
    costs, feasible, rho = (tuple(tuple(row) for row in matrix)
                             for matrix in (costs, feasible, rho))
    if any(len(matrix) != m or any(len(row) != n for row in matrix)
           for matrix in (costs, feasible, rho)):
        raise ValueError("Matrices must have shape (tracks, detections)")
    if any(type(x) is not bool for row in feasible for x in row):
        raise ValueError("Feasible mask must contain booleans")
    rho = tuple(finite_values(row, "rho") for row in rho)
    if any(x < 0 for row in rho for x in row):
        raise ValueError("rho must be nonnegative")
    atol, rtol = finite_values((atol, rtol), "tolerances")
    if atol < 0 or rtol < 0:
        raise ValueError("Tolerances must be nonnegative")
    if not isinstance(nominal, Assignment):
        raise ValueError("nominal must be an Assignment")
    canonical = Assignment.from_matches(m, n, nominal.matches)
    if canonical != nominal:
        raise ValueError("Nominal decisions must be canonical and complete")
    if any(not feasible[i][j] for i, j in nominal.matches):
        raise ValueError("Nominal contains a forbidden edge")
    selected = set(nominal.matches)
    clean = [finite_values((costs[i][j] if feasible[i][j] else 0.0
                            for j in range(n)), "feasible costs") for i in range(m)]
    values = [x for row in clean for x in row] + [x for row in rho for x in row] + list(tp) + list(dp)
    integers, scale = integer_scale(values)
    it = iter(integers)
    clean = [[next(it) for _ in range(n)] for _ in range(m)]
    radii = [[next(it) for _ in range(n)] for _ in range(m)]
    tp, dp = tuple(next(it) for _ in range(m)), tuple(next(it) for _ in range(n))
    adjusted = [[clean[i][j] + (radii[i][j] if (i, j) in selected else -radii[i][j])
                 for j in range(n)] for i in range(m)]
    competitor = strongest_competitor(nominal, adjusted, feasible, tp, dp)
    diff = disagreement(nominal, competitor)
    if competitor is None:
        return StabilityResult(nominal, None, None, "no_competitor", atol, diff,
                               None, Fraction.from_float(atol))
    # Cancel shared decisions before summing to avoid needless cancellation.
    terms = ([clean[i][j] - radii[i][j] for i, j in diff.added_matches]
             + [-clean[i][j] - radii[i][j] for i, j in diff.removed_matches]
             + [tp[i] * (1 if i in competitor.unmatched_tracks else -1)
                for i in diff.changed_unmatched_tracks]
             + [dp[j] * (1 if j in competitor.unmatched_detections else -1)
                for j in diff.changed_unmatched_detections])
    exact_gamma = Fraction(sum(terms), scale)
    exact_tolerance = (Fraction.from_float(atol) + Fraction.from_float(rtol)
                       * Fraction(sum(abs(x) for x in terms), scale))
    status = ("stable" if exact_gamma > exact_tolerance else
              "unstable" if exact_gamma < -exact_tolerance else "tied")
    try:
        gamma, tolerance = float(exact_gamma), float(exact_tolerance)
    except OverflowError as exc:
        raise ValueError("Exact margin or tolerance exceeds public float range") from exc
    return StabilityResult(nominal, competitor, gamma, status, tolerance, diff,
                           exact_gamma, exact_tolerance)

"""Complete physical competitor search by excluding nominal decisions."""
from .assignment import solve_partial


def assignment_cost(assignment, costs, track_penalties, detection_penalties):
    # The analyzer supplies common-scale integers, including unmatched costs.
    return sum([costs[i][j] for i, j in assignment.matches]
                + [track_penalties[i] for i in assignment.unmatched_tracks]
                + [detection_penalties[j] for j in assignment.unmatched_detections])


def strongest_competitor(nominal, adjusted, mask, track_penalties, detection_penalties):
    # Any distinct physical assignment omits at least one of these decisions.
    exclusions = ([('match', i, j) for i, j in nominal.matches]
                  + [('track', i) for i in nominal.unmatched_tracks]
                  + [('detection', j) for j in nominal.unmatched_detections])
    candidates = set()
    for excluded in exclusions:
        candidate = solve_partial(adjusted, mask, track_penalties,
                                  detection_penalties, excluded)
        if candidate is not None:
            candidates.add(candidate)
    return min(candidates, key=lambda x: (
        assignment_cost(x, adjusted, track_penalties, detection_penalties), x),
        default=None)

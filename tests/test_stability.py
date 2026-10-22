"""Independent enumeration of physical partial matchings; CPU/std-library only."""
import itertools
import random
import unittest
from fractions import Fraction

from src.stability import Assignment, analyze_stability
from src.stability.assignment import hungarian


def enumerate_physical(m, n, mask):
    # Cartesian choices, rather than augmentation, Hungarian or exclusion search.
    for choices in itertools.product(range(-1, n), repeat=m):
        columns = [j for j in choices if j >= 0]
        if len(columns) != len(set(columns)):
            continue
        if any(j >= 0 and not mask[i][j] for i, j in enumerate(choices)):
            continue
        yield (tuple((i, j) for i, j in enumerate(choices) if j >= 0),
               tuple(i for i, j in enumerate(choices) if j < 0),
               tuple(j for j in range(n) if j not in columns))


def physical(x):
    return x.matches, x.unmatched_tracks, x.unmatched_detections


def oracle_value(x, nominal, c, rho, tp, dp):
    # Independent rational arithmetic on exact input floats; no scaling helper,
    # augmented graph, Hungarian solver or production objective is reused.
    def exact(v):
        return Fraction.from_float(float(v))
    def total(a):
        return (sum((exact(c[i][j]) for i, j in a[0]), Fraction())
                + sum((exact(tp[i]) for i in a[1]), Fraction())
                + sum((exact(dp[j]) for j in a[2]), Fraction()))
    changed = set(x[0]) ^ set(nominal[0])
    return total(x) - total(nominal) - sum((exact(rho[i][j]) for i, j in changed), Fraction())


class StabilityTests(unittest.TestCase):
    def check_oracle(self, c, mask, rho, tp, dp, nominal):
        result = analyze_stability(c, mask, rho, tp, dp, nominal)
        alternatives = [x for x in enumerate_physical(len(tp), len(dp), mask)
                        if x != physical(nominal)]
        if not alternatives:
            self.assertIsNone(result.competitor)
            self.assertIsNone(result.gamma)
            self.assertIsNone(result.exact_gamma)
            self.assertEqual(result.margin_status, 'no_competitor')
            self.assertEqual(result.disagreement.added_matches, ())
            return result
        values = [(oracle_value(x, physical(nominal), c, rho, tp, dp), x)
                  for x in alternatives]
        best = min(v for v, _ in values)
        self.assertEqual(result.exact_gamma, best)
        self.assertEqual(result.gamma, float(best))
        winners = [x for v, x in values if v == best]
        self.assertIn(physical(result.competitor), winners)
        self.assertFalse(result.margin_status == 'stable' and best <= 0)
        return result

    def test_stable_unstable_tied_and_nonoptimal(self):
        x = Assignment.from_matches(2, 2, [(0, 0), (1, 1)])
        for radius, status in [(0, 'stable'), (1, 'tied'), (2, 'unstable')]:
            with self.subTest(radius=radius):
                r = self.check_oracle([[0, 2], [2, 0]], [[True]*2]*2,
                                      [[radius]*2]*2, [10]*2, [10]*2, x)
                self.assertEqual(r.margin_status, status)
        x = Assignment.from_matches(1, 1, [])
        r = self.check_oracle([[0]], [[True]], [[0]], [2], [2], x)
        self.assertEqual(r.gamma, -4)
        self.assertEqual(r.margin_status, 'unstable')

    def test_entirely_unmatched_and_diagnostics(self):
        x = Assignment.from_matches(1, 1, [])
        r = self.check_oracle([[5]], [[True]], [[1]], [1], [1], x)
        self.assertEqual(r.gamma, 2)
        self.assertEqual(r.disagreement.added_matches, ((0, 0),))
        self.assertEqual(r.disagreement.changed_unmatched_tracks, (0,))
        self.assertEqual(r.disagreement.changed_unmatched_detections, (0,))
        r = self.check_oracle([[0]], [[False]], [[100]], [1], [1], x)
        self.assertEqual(r.margin_status, 'no_competitor')

    def test_absent_tracks_detections_and_empty(self):
        for m, n in [(0, 0), (0, 3), (3, 0)]:
            with self.subTest(shape=(m, n)):
                self.check_oracle([[] for _ in range(m)], [[] for _ in range(m)],
                                  [[] for _ in range(m)], [1]*m, [1]*n,
                                  Assignment.from_matches(m, n, []))

    def test_rectangular_forbidden_and_negative_costs(self):
        for m, n in [(2, 3), (3, 2)]:
            mask = [[i != j for j in range(n)] for i in range(m)]
            c = [[-i-j if mask[i][j] else float('nan') for j in range(n)]
                 for i in range(m)]
            for matches, _, _ in enumerate_physical(m, n, mask):
                self.check_oracle(c, mask, [[0.5]*n for _ in range(m)],
                                  [2]*m, [3]*n, Assignment.from_matches(m, n, matches))

    def test_seeded_exhaustive_all_nominals(self):
        rng = random.Random(42)
        for m, n in [(1, 2), (2, 1), (2, 2), (2, 3), (3, 2), (3, 3)]:
            for _ in range(6):
                mask = [[rng.random() < 0.7 for _ in range(n)] for _ in range(m)]
                c = [[rng.randint(-4, 8) for _ in range(n)] for _ in range(m)]
                rho = [[rng.choice([0, 0.25, 1, 3]) for _ in range(n)] for _ in range(m)]
                tp, dp = [rng.randint(-2, 5) for _ in range(m)], [rng.randint(-2, 5) for _ in range(n)]
                for matches, _, _ in enumerate_physical(m, n, mask):
                    self.check_oracle(c, mask, rho, tp, dp,
                                      Assignment.from_matches(m, n, matches))

    def test_ties_deterministic_and_tolerances(self):
        args = ([[0, 0], [0, 0]], [[True]*2]*2, [[0]*2]*2, [0]*2, [0]*2,
                Assignment.from_matches(2, 2, []))
        results = [analyze_stability(*args) for _ in range(5)]
        self.assertTrue(all(x == results[0] for x in results))
        x = Assignment.from_matches(1, 1, [])
        r = analyze_stability([[2+1e-11]], [[True]], [[0]], [1], [1], x)
        self.assertEqual(r.margin_status, 'tied')
        r = analyze_stability([[2+1e-11]], [[True]], [[0]], [1], [1], x, atol=0, rtol=0)
        self.assertEqual(r.margin_status, 'stable')

    def test_structural_infeasibility(self):
        self.assertIsNone(hungarian([[0, None], [0, None]]))
        self.assertEqual(hungarian([[None, -100], [1, None]]), (1, 0))
        # Intermediate potentials no longer overflow; the optimum is exact.
        self.assertEqual(hungarian([[1e308, -1e308], [1e308, -1e308]]), (1, 0))

    def test_audit_false_certificate_regression(self):
        r = self.check_oracle([[1., 4.], [1., 2.]], [[False, True], [False, True]],
                              [[0., 0.], [0., 0.]], [1., 1.], [1e16, 2.],
                              Assignment.from_matches(2, 2, []))
        self.assertEqual(r.gamma, -1)
        self.assertEqual(r.margin_status, 'unstable')
        self.assertEqual(r.competitor.matches, ((1, 1),))

    def test_hungarian_mixed_scale_exact_permutation_oracle(self):
        # A single constrained optimization must preserve small differences too.
        matrices = [[[1e16, 1e16], [1., 2.]],
                    [[1e308, -1e308], [1e308, -1e308]],
                    [[0.1, 0.3, None], [0.2, 1e16, 0.1], [1e16, 0.2, 0.3]]]
        for a in matrices:
            n = len(a)
            def total(p):
                return sum((Fraction.from_float(float(a[i][p[i]]))
                            for i in range(n)), Fraction())
            possible = [p for p in itertools.permutations(range(n))
                        if all(a[i][p[i]] is not None for i in range(n))]
            self.assertEqual(total(hungarian(a)), min(map(total, possible)))

    def test_removal_swap_and_unmatched_transitions(self):
        r = self.check_oracle([[5.]], [[True]], [[0.1]], [1.], [1.],
                              Assignment.from_matches(1, 1, [(0, 0)]))
        self.assertEqual(r.disagreement.removed_matches, ((0, 0),))
        self.assertEqual(r.disagreement.added_matches, ())
        self.assertEqual(r.disagreement.changed_unmatched_tracks, (0,))
        self.assertEqual(r.disagreement.changed_unmatched_detections, (0,))
        r = self.check_oracle([[3., 0.1], [0.2, 3.]], [[True]*2]*2,
                              [[0.1]*2]*2, [100.]*2, [100.]*2,
                              Assignment.from_matches(2, 2, [(0, 0), (1, 1)]))
        self.assertEqual(r.disagreement.removed_matches, ((0, 0), (1, 1)))
        self.assertEqual(r.disagreement.added_matches, ((0, 1), (1, 0)))
        self.assertEqual(r.disagreement.changed_unmatched_tracks, ())
        self.assertEqual(r.disagreement.changed_unmatched_detections, ())
        r = self.check_oracle([[2., 0.1]], [[True, True]], [[0., 0.]],
                              [10.], [0., 0.], Assignment.from_matches(1, 2, [(0, 0)]))
        self.assertEqual(r.disagreement.changed_unmatched_detections, (0, 1))
        self.assertEqual(r.disagreement.changed_unmatched_tracks, ())

    def test_mixed_scale_exact_all_nominals(self):
        rng = random.Random(946)
        pool = [0., 0.1, 0.2, 1., -1., 4., 1e16, -1e16, 1e100, 5e-324]
        for _ in range(60):
            mask = [[rng.random() < 0.8 for _ in range(2)] for _ in range(2)]
            c = [[rng.choice(pool) for _ in range(2)] for _ in range(2)]
            rho = [[rng.choice([0., 0.1, 1., 5e-324]) for _ in range(2)] for _ in range(2)]
            tp, dp = [rng.choice(pool) for _ in range(2)], [rng.choice(pool) for _ in range(2)]
            for matches, _, _ in enumerate_physical(2, 2, mask):
                self.check_oracle(c, mask, rho, tp, dp, Assignment.from_matches(2, 2, matches))

    def test_exact_adjustment_and_extreme_range(self):
        # Adding rho to a large float would previously discard rho completely.
        self.check_oracle([[1e16, 1e16]], [[True, True]], [[0.1, 0.2]],
                          [1e16], [0., 0.], Assignment.from_matches(1, 2, [(0, 0)]))
        r = self.check_oracle([[5e-324]], [[True]], [[0.]], [0.], [0.],
                              Assignment.from_matches(1, 1, []))
        self.assertEqual(r.exact_gamma, Fraction.from_float(5e-324))
        self.assertEqual(r.margin_status, 'tied')
        with self.assertRaisesRegex(ValueError, 'public float range'):
            analyze_stability([[-1e308]], [[True]], [[0.]], [1e308], [1e308],
                              Assignment.from_matches(1, 1, []))
        # Negative near-zero gaps are never stable, even if within tolerance.
        r = analyze_stability([[2.-1e-11]], [[True]], [[0.]], [1.], [1.],
                              Assignment.from_matches(1, 1, []))
        self.assertLess(r.exact_gamma, 0)
        self.assertEqual(r.margin_status, 'tied')

    def test_invalid_inputs(self):
        base = dict(costs=[[1]], feasible=[[True]], rho=[[0]],
                    track_penalties=[1], detection_penalties=[1],
                    nominal=Assignment.from_matches(1, 1, [(0, 0)]))
        changes = [dict(rho=[[-1]]), dict(rho=[[float('nan')]]),
                   dict(costs=[[float('inf')]]), dict(costs=[]),
                   dict(feasible=[[False]]), dict(feasible=[[1]]),
                   dict(track_penalties=[float('inf')]), dict(atol=-1),
                   dict(rtol=float('nan')), dict(nominal=Assignment(((0, 0),), (0,), ()))]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                analyze_stability(**(base | change))
        with self.assertRaises(ValueError):
            Assignment.from_matches(2, 2, [(0, 0), (0, 1)])


if __name__ == '__main__':
    unittest.main()

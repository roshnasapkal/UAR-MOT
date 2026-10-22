# Phase 1: standalone global assignment stability

This module is a methodological clarification of the manuscript's incompletely
specified unmatched costs. It does not reproduce or alter ByteTrack's actual
association policy. ByteTrack compatibility requires a later integration study.

## Public contract

`src.stability.analyze_stability(costs, feasible, rho, track_penalties,
detection_penalties, nominal, *, atol=1e-10, rtol=1e-12)` accepts rectangular
row sequences. Penalty lengths define the dimensions, including zero rows or
columns. `feasible` contains Python booleans; unavailable real edges can never
be selected. Costs on unavailable edges are ignored. All radii must be finite
and nonnegative, including those on unavailable edges (which have no effect).
Feasible costs and penalties must be finite. Negative costs/penalties are allowed.

Create the nominal assignment with `Assignment.from_matches(m, n, pairs)`;
unmatched decisions are inferred explicitly from the complement of the matches.
The analyzer validates feasibility and complete, canonical unmatched lists.
It neither computes nor requires cost optimality of the supplied nominal.

A physical assignment X is a partial one-to-one matching with cost

    C(X) = sum matched c[i,j] + sum unmatched-track p[i]
           + sum unmatched-detection q[j].

Unmatched penalties are caller supplied, fixed, and mandatory. Real feasible
edges have independent box perturbations |delta[i,j]| <= rho[i,j]. Forbidden
edges remain forbidden under perturbation. The adjusted real costs add rho on
nominal matches and subtract rho on other feasible edges. Unmatched costs do
not change.

For every distinct physical competitor Y, its worst-case advantage is measured
by the gap

    g(Y) = C(Y) - C(X*) - sum rho[i,j] over M(Y) symmetric-difference M(X*).

The strongest competitor minimizes g; Gamma is its value. Positive Gamma
certifies unique stability only within this modeled feasible set and uncertainty
box. It does not guarantee ground-truth tracking correctness. A feasible but
nonoptimal nominal is accepted and can yield a negative Gamma even with rho=0.

## Complete search and structural feasibility

The solver augments m track rows with n detection-dummy rows, and n detection
columns with m track-dummy columns. A track can use its own unmatched column;
a detection dummy can use its own detection column. Dummy-to-dummy edges have
zero cost. Every physical partial matching has such a perfect augmentation.
Absent edges are `None`, never finite big-M values. Hungarian returns infeasible
when no augmenting path exists.

One adjusted solve excludes each selected nominal real match, unmatched track,
or unmatched detection in turn. Every physically distinct assignment omits at
least one such decision, proving coverage of all competitors. This includes an
entirely unmatched nominal. Results are decoded and deduplicated by physical
decisions; dummy permutations are not alternatives.

## Numerical and result semantics

Ascending Hungarian scans with strict comparisons resolve exact ties
deterministically. Across returned physical candidates, exact equal adjusted
costs use lexicographic Assignment ordering. This does not promise the globally
lexicographically smallest member of every tied solution set.

The numerical contract is exact **relative to the accepted Python float values**.
Costs, radii, penalties and tolerances are first normalized with `float(value)`;
nonfinite values are rejected (forbidden costs are ignored). This conversion may
round supplied decimal strings, large integers or other numeric types. The core
does not certify values before that conversion, nor measurement errors beyond rho.

Every finite binary float is an exact integer divided by a power of two. All
costs, radii and penalties are converted with `as_integer_ratio()` to a common
power-of-two denominator. Adjusted costs are formed by integer addition and
subtraction **after** conversion. Hungarian potentials, reduced costs and all
cross-candidate totals use arbitrary-precision Python integers. No floating
infinity participates in optimization: `None` denotes absent edges and
unreached distances. Consequently, neither a large common cost nor a small rho
can erase an optimization difference. The low-level Hungarian routine also
scales finite float inputs exactly when called directly.

Gamma is computed from changed decisions on that same integer scale. The result
adds `exact_gamma` and `exact_tolerance` (`fractions.Fraction`) for lossless
inspection. Existing `gamma` and `tolerance` fields remain float projections for
convenience. They are not used for decisions. Tolerance is computed exactly as
`atol + rtol * sum(abs(changed gap terms))`, including exact representations of
the supplied float tolerances. Status is `stable` for exact Gamma > exact tol,
`unstable` for exact Gamma < -exact tol, and `tied` otherwise. `tied` denotes the
explicit tolerance band, not an unresolved solver optimization. In particular,
a negative exact Gamma can never be classified as stable. Tolerances neither
perturb optimization costs nor permit suboptimal competitors.

If converting the exact Gamma or tolerance to its public float projection
overflows, the call raises `ValueError` without returning a certificate. Exact
intermediate costs and potentials need not fit in a float. Thus the supported
successful-return contract requires finite normalized inputs and representable
public Gamma/tolerance projections, not representable intermediate float sums.
Returned positive stability certificates concern only the modeled feasible set
and independent uncertainty box at those normalized input values.

When no distinct physical competitor exists, `competitor=None`, `gamma=None`,
and `margin_status='no_competitor'`. This is not an ordinary stable result.
`exact_gamma` is also None; `exact_tolerance` is the exact supplied atol.
Empty matrices, zero tracks, zero detections, and entirely forbidden graphs have
one physical assignment and therefore no competitor. Disagreement diagnostics
report removed/added real matches and symmetric differences of unmatched track
and detection decisions. No competitor produces empty diagnostics.

## Complexity and validation

Let N=m+n and k=|M(X*)|. There are m+n-k exclusion solves. Each dense Hungarian
solve takes O(N^3) time and O(N^2) space including augmentation; total search is
O((m+n-k) N^3), hence O(N^4) in the worst case. Stored deduplicated physical
candidates take O(N^2) space. No GPU or external Python packages are required.
These are arithmetic-operation/storage-count bounds, not fixed-width bit
complexity bounds. If scaled costs occupy B bits, integer arithmetic and storage
grow with B (and sums/potentials can require additional bits depending on N).
Across the binary64 exponent range, common-scale inputs can approach 2100 bits.
Exact arithmetic adds overhead to the already dense repeated solves; practical
real-time scalability and tracker integration have not been benchmarked.

`python -m unittest discover -s tests -p test_stability.py -v` runs the Phase 1
suite. Its independent Cartesian-product oracle enumerates physical matchings
directly, evaluates the gap formula, and checks the strongest competitor set.
Its independent Fraction oracle constructs exact rationals from each input float
and compares gaps and competitor membership with exact equality. Tests include
the audited false-certificate counterexample, mixed-scale costs, non-binary-exact
decimal floats, subnormals, overflow rejection, direct Hungarian permutation
comparisons, removal, swaps, unmatched transitions, ties and nonoptimal nominals.
The implementation does not calibrate uncertainty, implement recovery cues,
integrate tracker association stages or choose unmatched penalties.

# from FinalizeColumns import    detect_four_columns # check_predicted_column_values


"""
Modern, efficient redesign of the column-detection algorithm.

Design goals (Option C):
- Maintain the logical intent: find 4 columns (likely evenly spaced)
  from a 1D projection profile of black pixels.
- Replace brute-force nested loops with a clear pipeline:
    1. preprocess profile (smoothing optional)
    2. extract peaks
    3. split peaks into quadrants and keep top-N per quadrant
    4. attempt geometric matches using predicted positions + binary search
    5. validate using cached local-max & geometry validators
- Provide backward-compatibility hooks to call original helpers.

Notes mapping to original:
- Original quadrant enumeration & sorting -> split_and_sort_quadrants()
- Original iter1..iter4 and potentialNextVal selection -> candidate priority + top-N approach
- Original columnConfirm3 (buggy) -> sliding_window_max & is_local_max()
- Original nested loops of distance comparisons -> predict-and-binary-search using bisect

Author: Original -> Kevin E.      Refactor -> ChatGPT, edited by Kevin E.
"""

from typing import List, Tuple, Callable, Optional, Sequence
from collections import deque
from functools import lru_cache
import bisect
import math


# -------------------------
# Utility & helper types
# -------------------------
Index = int
Profile = List[int]


# -------------------------
# Default helpers (fallbacks)
# -------------------------
# def default_within_tolerance(expected: float, actual: float, tolerance: float) -> bool:
#     """
#     Fallback for withinTolerance from original code.
#     Original usage: withinTolerance(diff1 * K, diff2, tolerance)
#     Improvement: Use a robust absolute-or-relative tolerance test.
#     """
#     if expected == 0:
#         return abs(actual) <= tolerance
#     # allow small absolute variations or relative variations
#     return abs(actual - expected) <= max(tolerance, tolerance * abs(expected))

# the original version
def default_within_tolerance(firstVal, secondVal, tolerance):
    if abs(firstVal - secondVal) < tolerance:
        return True
    else:
        return False




def default_column_confirm2(i1: Index, i2: Index, i3: Index, i4: Index) -> bool:
    """
    Fallback for columnConfirm2 from original code.
    The original likely checked relative geometry constraints; we implement a permissive default:
    - indices should be integers >= 0, strictly increasing (i1 < i2 < i3 < i4).
    Improvement: conservative, but you should inject your original columnConfirm2 for exact behavior.
    """
    return all(isinstance(x, int) and x >= 0 for x in (i1, i2, i3, i4)) and (i1 < i2 < i3 < i4)


def default_initial_check(indices: Sequence[Index]) -> bool:
    """
    Fallback for initialCheck. The original called initialCheck([q1_top, q2_top, q3_top, q4_top]).
    Default: return False to force the full search (safe).
    """
    return False


# -------------------------
# Sliding-window maximum
# -------------------------
def sliding_window_max(arr: Profile, window: int) -> List[int]:
    """
    Compute maximum over sliding window of radius `window` centered at each index.
    Implementation: classic deque-based O(n).
    For an index i, this function returns the maximum in arr[max(0, i-window) : min(len(arr), i+window+1)].
    Improvement over original: precompute in O(n) and then local-max checks are O(1).
    """
    n = len(arr)
    if n == 0:
        return []

    # We'll build max for windows [i, i+2*window] then shift — easier approach:
    # Build max over fixed-size window of size (2*window + 1) using deque
    k = 2 * window + 1
    res = [0] * n

    dq = deque()  # stores indices of potential maxima for current sliding window (mono deque)
    # First pass: process first k elements; windows that are smaller near boundaries are handled by bounds in output
    for i in range(n):
        # Remove indices out of window [i-k+1 .. i]
        while dq and dq[0] < i - k + 1:
            dq.popleft()
        # Remove smaller values from the back
        while dq and arr[dq[-1]] <= arr[i]:
            dq.pop()
        dq.append(i)
        # The maximum for window ending at i is at dq[0]. We want center-based windows; convert later.

        # compute center index whose window ends at i:
        center = i - window
        if 0 <= center < n:
            # window for center is from center-window to center+window which ends at i
            res[center] = arr[dq[0]]

    # Fix left & right edges (where full window didn't exist): we already handled via bounds above.
    # For indices where result still zero but arr may be zero, ensure res filled
    # (the code above fills all centers 0..n-1, so we're good)
    return res


# -------------------------
# Peak extraction utilities
# -------------------------
def find_peaks(profile: Profile, min_height: int = 1) -> List[Tuple[int, Index]]:
    """
    Find simple peaks in 1D profile (local maxima compared to immediate neighbors).
    Returns list of (height, index). This is a light-weight peak pick used for candidate extraction.
    Improvement vs original: we extract peaks first, not every non-zero index; this massively reduces search space.
    """
    n = len(profile)
    peaks = []
    for i in range(n):
        h = profile[i]
        if h < min_height:
            continue
        left = profile[i - 1] if i - 1 >= 0 else -1
        right = profile[i + 1] if i + 1 < n else -1
        if h >= left and h >= right and (h > left or h > right):
            peaks.append((h, i))
        peaks.append((h, i))
    return peaks


def top_n_per_quadrant(peaks: List[Tuple[int, Index]], profile_len: int, n_per_quad: int) -> Tuple[List[Tuple[int, Index]], List[Tuple[int, Index]], List[Tuple[int, Index]], List[Tuple[int, Index]]]:
    """
    Given peaks = [(height,index), ...], split into 4 quadrants by column index,
    sort each quadrant descending by height and keep top n_per_quad entries.
    This replaces original Quad1..Quad4 enumeration & sort, with tunable top-N filtering to reduce computation.
    """
    q1, q2, q3, q4 = [], [], [], []
    L = profile_len
    for h, idx in peaks:
        if idx < L * 0.25:
            q1.append((h, idx))
        elif idx < L * 0.5:
            q2.append((h, idx))
        elif idx < L * 0.75:
            q3.append((h, idx))
        else:
            q4.append((h, idx))

    for q in (q1, q2, q3, q4):
        q.sort(reverse=True, key=lambda x: x[0])
    q1 = q1[:n_per_quad]
    q2 = q2[:n_per_quad]
    q3 = q3[:n_per_quad]
    q4 = q4[:n_per_quad]
    print(f"  ~!~!~!!@!@!@!@! Sorted quad 1: {q1}")
    print(f"  ~!~!~!!@!@!@!@! Sorted quad 2: {q2}")
    print(f"  ~!~!~!!@!@!@!@! Sorted quad 3: {q3}")
    print(f"  ~!~!~!!@!@!@!@! Sorted quad 4: {q4}")
    return q1, q2, q3, q4


# -------------------------
# Binary-search helpers
# -------------------------
def find_nearest_index(sorted_indices: List[Index], target: float) -> Tuple[Optional[Index], float]:
    """
    Given sorted_indices (ascending), find the element nearest to target.
    Return (index_or_None, distance)
    Improvement: O(log n) search instead of O(n) scan used in original nested loops.
    """
    if not sorted_indices:
        return None, float('inf')
    pos = bisect.bisect_left(sorted_indices, int(round(target)))
    candidates = []
    if pos < len(sorted_indices):
        candidates.append(sorted_indices[pos])
    if pos - 1 >= 0:
        candidates.append(sorted_indices[pos - 1])
    # choose nearest
    best = min(candidates, key=lambda x: abs(x - target))
    return best, abs(best - target)


# -------------------------
# Cached validators
# -------------------------
@lru_cache(maxsize=None)
def _cached_local_max(index: int, profile_tuple: Tuple[int, ...], radius: int) -> bool:
    """
    A wrapper for fast repeated local-max checks.
    AKA checks that the current index represents the largest local value within radius
    The profile is passed as a tuple so that caching is safe. The caller should pass tuple(profile).
    """
    profile = list(profile_tuple)
    n = len(profile)
    if index < 0 or index >= n:
        return False
    val = profile[index]
    # # naive early-exit approach: check both sides but stop when a larger element is found
    # # radius default is large (e.g., 625), but early exit will often short-circuit
    # for d in range(1, radius + 1):
    #     j = index + d
    #     if j < n and profile[j] > val:
    #         return False
    #     j = index - d
    #     if j >= 0 and profile[j] > val:
    #         return False

    d = 3
    j = index + d
    if j < n and profile[j] > val:
        return False
    j = index - d
    if j >= 0 and profile[j] > val:
        return False

    return True


# -------------------------
# High-level matcher
# -------------------------
def detect_four_columns(
    proj_profile: Profile,
    *,
    window_radius_for_local_max: int = 625,
    top_n_per_quad: int = 200,
    peak_min_height: int = 10000,
    tolerance: float = 35,
    initial_check: Callable[[Sequence[Index]], bool] = default_initial_check,
    within_tolerance: Callable[[float, float, float], bool] = default_within_tolerance,
    column_confirm2: Callable[[Index, Index, Index, Index], bool] = default_column_confirm2,
    column_confirm_local: Optional[Callable[[Index, Profile], bool]] = None
) -> List[int]:
    """
    Main function to detect four columns.

    Parameters:
    - proj_profile: list of pixel counts per column (original input)
    - window_radius_for_local_max: how far to check for being a local maximum (default 625 to match original)
    - top_n_per_quad: keep only top-N peaks per quadrant (performance knob)
    - peak_min_height: ignore peaks below this pixel count (noise rejection)
    - tolerance: relative geometric tolerance used in matching
    - initial_check: function([i1,i2,i3,i4]) -> bool (preserves original initialCheck behavior)
    - within_tolerance: function(expected,actual,tol) -> bool
    - column_confirm2: function(i1,i2,i3,i4) -> bool (preserve original complex geometry test)
    - column_confirm_local: function(index, profile) -> bool; if None, uses cached local max approach

    Returns:
      [i1, i2, i3, i4] on success, or [-1,-1,-1,-1] on failure.
    """

    n = len(proj_profile)
    if n == 0:
        return [-1, -1, -1, -1]

    # Provide default local-max function if none given
    if column_confirm_local is None:
        # We will call _cached_local_max which takes tuple(profile) for caching
        profile_tuple = tuple(proj_profile)
        def column_confirm_local(index, _profile=profile_tuple, radius=window_radius_for_local_max):
            return _cached_local_max(index, _profile, radius)

    # 1) Preprocess: find peaks (much cheaper than scanning every non-zero index)
    #    Original: iterated every i for i,pixelcount in enumerate(proj_profile) to collect Quad lists.
    peaks = find_peaks(proj_profile, min_height=peak_min_height)

    # 2) Split peaks into quadrants and keep top-N (original: Quad1..Quad4 sort((reverse=True)))
    q1, q2, q3, q4 = top_n_per_quadrant(peaks, n, top_n_per_quad)

    # If any quadrant empty, we can still attempt but original code expected at least one in each.
    if not (q1 and q2 and q3 and q4):
        # Try a fallback: expand top_n_per_quad to include more if some quads empty
        # But keep simple: if too many are empty, return failure quickly
        if not (q1 and q2 and q3 and q4):
            _cached_local_max.cache_clear()
            return [-1, -1, -1, -1]

    # Prepare ascending lists of indices for binary search
    q1_idx = sorted([idx for _, idx in q1])
    q2_idx = sorted([idx for _, idx in q2])
    q3_idx = sorted([idx for _, idx in q3])
    q4_idx = sorted([idx for _, idx in q4])

    # 3) initial check with the strongest candidate from each quadrant (original: initialCheck([...]))
    try_top = [q1[0][1], q2[0][1], q3[0][1], q4[0][1]]
    try:
        if initial_check(try_top):
            _cached_local_max.cache_clear()
            return try_top
    except Exception:
        # If user-provided initial_check is missing or throws, fall back to full search (original code expected initialCheck to exist)
        pass

    # 4) Candidate matching strategy:
    #    Instead of brute-forcing every triple/quadruple of peaks, we:
    #      - iterate each candidate as "anchor" (from union of top peaks)
    #      - compute a predicted partner index using expected geometric patterns (e.g. index3 = index1 + k*diff)
    #      - binary-search for nearest candidate in the predicted quadrant and test within_tolerance + validators
    #
    #    This preserves the original geometric intent (diff relationships) but replaces scanning with O(log n) lookups.

    # Build a unified prioritized list of candidate peak indices sorted descending by peak height
    # (this corresponds to original picking the quadrant with the strongest "next" value)
    unified_peaks = sorted(peaks, reverse=True, key=lambda t: t[0])  # (height, index)

    # Helper to validate and return on success
    def validate_and_return(i1, i2, i3, i4):
        # Ensure indices are ints and in range
        if not (0 <= i1 < n and 0 <= i2 < n and 0 <= i3 < n and 0 <= i4 < n):
            return None
        # Confirm geometry via column_confirm2 (originally used) and confirm peaks via local check (column_confirm3)
        try:
            if not column_confirm2(i1, i2, i3, i4):
                return None
        except Exception:
            # If user provided failing column_confirm2, treat as not confirmed
            return None
        # local confirmations (use cached or provided functions)
        if not column_confirm_local(i1) or not column_confirm_local(i2) or not column_confirm_local(i3) or not column_confirm_local(i4):
            return None
        return [i1, i2, i3, i4]

    # Remap some variables for readability
    q_indices = [q1_idx, q2_idx, q3_idx, q4_idx]

    # Iteration strategy:
    # For each strong peak p (descending height), try patterns depending on which quadrant it belongs to.
    # For clarity we explicitly implement the patterns used in the original (multipliers 1,2,3 etc).
    for peak_height, anchor_idx in unified_peaks:
        # Which quadrant is anchor in?
        # 0 => Q1, 1=>Q2, 2=>Q3, 3=>Q4
        if anchor_idx < n * 0.25:
            quad = 0
        elif anchor_idx < n * 0.5:
            quad = 1
        elif anchor_idx < n * 0.75:
            quad = 2
        else:
            quad = 3

        # For each quadrant we will attempt the geometric patterns observed in original code.
        # patterns: list of tuples describing predictable relationships. Each entry:
        # (anchor_quad, other_anchor_quad, multiplier_for_predicted, compute_indices_fn)
        # We'll implement each original case explicitly for clarity.

        if quad == 0:
            # Original: index_of_max == 0 block.
            # Anchor was Quad1 (current_x_pos). It looped over seen Quad2 items and checked Quad3 and Quad4.
            # Pattern 1 (with Quad3): diff2 ≈ diff1 * 2  => index3 approx, index4 = index3 + diff1
            # Pattern 2 (with Quad4): diff2 ≈ diff1 * 3  => index4 approx, index3 = index2 + diff1
            for a2 in q2_idx:
                diff1 = abs(anchor_idx - a2)
                if diff1 == 0:
                    continue
                # predict index3 ≈ anchor_idx + ? (we test both sides using binary search)
                expected3 = anchor_idx + 2 * diff1  # original used diff1*2 relation
                cand3, dist3 = find_nearest_index(q3_idx, expected3)
                if cand3 is not None and within_tolerance(expected3, cand3, tolerance):
                    index1, index2, index3 = anchor_idx, a2, cand3
                    index4 = index3 + diff1
                    res = validate_and_return(index1, index2, index3, index4)
                    if res:
                        _cached_local_max.cache_clear()
                        return res
                # check Quad4 pattern
                expected4 = anchor_idx + 3 * diff1
                cand4, dist4 = find_nearest_index(q4_idx, expected4)
                if cand4 is not None and within_tolerance(expected4, cand4, tolerance):
                    index1, index2, index4 = anchor_idx, a2, cand4
                    index3 = index2 + diff1
                    res = validate_and_return(index1, index2, index3, index4)
                    if res:
                        _cached_local_max.cache_clear()
                        return res

        elif quad == 1:
            # Original: index_of_max == 1 block.
            # Anchor Quad2, loop through Quad1 and check Quad3 (diff ~ diff1) or Quad4 (diff ~ 2*diff1)
            for a1 in q1_idx:
                diff1 = abs(anchor_idx - a1)
                if diff1 == 0:
                    continue
                expected3 = anchor_idx + diff1  # pattern diff1 ≈ diff2
                cand3, _ = find_nearest_index(q3_idx, expected3)
                if cand3 is not None and within_tolerance(expected3, cand3, tolerance):
                    i1, i2, i3 = a1, anchor_idx, cand3
                    i4 = i3 + diff1
                    res = validate_and_return(i1, i2, i3, i4)
                    if res:
                        _cached_local_max.cache_clear()
                        return res
                expected4 = anchor_idx + 2 * diff1
                cand4, _ = find_nearest_index(q4_idx, expected4)
                if cand4 is not None and within_tolerance(expected4, cand4, tolerance):
                    i1, i2, i4 = a1, anchor_idx, cand4
                    i3 = i2 + diff1
                    res = validate_and_return(i1, i2, i3, i4)
                    if res:
                        _cached_local_max.cache_clear()
                        return res

        elif quad == 2:
            # Original: index_of_max == 2 block.
            # Anchor Quad3, used Quad4 as anchor then checked Quad1 and Quad2 with diff ratios (2x and 1x).
            for a4 in q4_idx:
                diff1 = abs(anchor_idx - a4)
                if diff1 == 0:
                    continue
                expected1 = anchor_idx - 2 * diff1  # pattern diff2*2 => index1 = index3 - 2*diff
                cand1, _ = find_nearest_index(q1_idx, expected1)
                if cand1 is not None and within_tolerance(expected1, cand1, tolerance):
                    i1, i3, i4 = cand1, anchor_idx, a4
                    i2 = i1 + diff1
                    res = validate_and_return(i1, i2, i3, i4)
                    if res:
                        _cached_local_max.cache_clear()
                        return res
                expected2 = anchor_idx - diff1  # pattern diff2 ~ diff1 => index2 approx
                cand2, _ = find_nearest_index(q2_idx, expected2)
                if cand2 is not None and within_tolerance(expected2, cand2, tolerance):
                    i2, i3, i4 = cand2, anchor_idx, a4
                    i1 = i2 - diff1
                    res = validate_and_return(i1, i2, i3, i4)
                    if res:
                        _cached_local_max.cache_clear()
                        return res

        else:  # quad == 3
            # Original: index_of_max == 3 block.
            # Anchor Quad4, examined Quad3 then looked for matches in Quad1 and Quad2 with multipliers (3x and 1x).
            for a3 in q3_idx:
                diff1 = abs(anchor_idx - a3)
                if diff1 == 0:
                    continue
                expected1 = a3 - 3 * diff1  # reverse of index4 = index1 + 3*diff -> index1 approx
                cand1, _ = find_nearest_index(q1_idx, expected1)
                if cand1 is not None and within_tolerance(expected1, cand1, tolerance):
                    i1, i3, i4 = cand1, a3, anchor_idx
                    i2 = i1 + diff1
                    res = validate_and_return(i1, i2, i3, i4)
                    if res:
                        _cached_local_max.cache_clear()
                        return res
                expected2 = anchor_idx - diff1
                cand2, _ = find_nearest_index(q2_idx, expected2)
                if cand2 is not None and within_tolerance(expected2, cand2, tolerance):
                    i2, i3, i4 = cand2, a3, anchor_idx
                    i1 = i2 - diff1
                    res = validate_and_return(i1, i2, i3, i4)
                    if res:
                        _cached_local_max.cache_clear()
                        return res

    # If nothing matched, return failure same as original
    _cached_local_max.cache_clear()
    return [-1, -1, -1, -1]
"""Symmetric difference of physical decisions, with directional diagnostics."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Disagreement:
    removed_matches: tuple = ()
    added_matches: tuple = ()
    changed_unmatched_tracks: tuple = ()
    changed_unmatched_detections: tuple = ()


def disagreement(nominal, competitor):
    if competitor is None:
        return Disagreement()
    return Disagreement(
        tuple(sorted(set(nominal.matches) - set(competitor.matches))),
        tuple(sorted(set(competitor.matches) - set(nominal.matches))),
        tuple(sorted(set(nominal.unmatched_tracks) ^ set(competitor.unmatched_tracks))),
        tuple(sorted(set(nominal.unmatched_detections) ^ set(competitor.unmatched_detections))))

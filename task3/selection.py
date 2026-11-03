"""Quality scores, VisConf calibration, highest-score selection, self-consistency and rank-weighted consensus.

Every function sees only the current candidate prefix (rollout IDs in reveal order). Ties go to the earliest
supporting rollout in that order; RWC uses average ranks for tied scores.
"""

import numpy as np
from scipy.stats import rankdata

from task3.config import CALIBRATION_FLOOR, CALIBRATION_QUANTILES, NO_ANSWER


def self_consistency(groups):
    """Plurality answer; ties broken by the earliest supporting candidate. Returns (answer, counts, first)."""
    counts, first = {}, {}
    for position, group in enumerate(groups):
        counts[group] = counts.get(group, 0) + 1
        first.setdefault(group, position)
    winner = max(counts, key=lambda group: (counts[group], -first[group]))
    return winner, counts, first


def highest_score(scores):
    """Position of the highest score; the earliest candidate wins ties."""
    best = np.max(scores)
    return int(np.flatnonzero(scores == best)[0])


def rwc_votes(groups, scores):
    """Rank-weighted votes w_i = (N - r_i) / (N - 1) with descending average ranks (rank 1 = best)."""
    count = len(groups)
    if count == 1:
        return {groups[0]: 1.0}
    ranks = rankdata(-np.asarray(scores, dtype=np.float64), method="average")
    weights = (count - ranks) / (count - 1)
    votes = {}
    for group, weight in zip(groups, weights):
        votes[group] = votes.get(group, 0.0) + float(weight)
    return votes


def rwc(groups, scores):
    votes = rwc_votes(groups, scores)
    first = {}
    for position, group in enumerate(groups):
        first.setdefault(group, position)
    winner = max(votes, key=lambda group: (votes[group], -first[group]))
    return winner, votes


def rwc_margin(votes):
    ordered = sorted(votes.values(), reverse=True)
    total = sum(ordered)
    if total <= 0:
        return 0.0
    runner_up = ordered[1] if len(ordered) > 1 else 0.0
    return (ordered[0] - runner_up) / total


def visconf_scores(record, candidates):
    """Calibrate token KL on the pooled candidate set (5th/95th linear percentiles); return Z, Z_token, C."""
    token_kl = [record["token_kl"][candidate] for candidate in candidates]
    token_vis = [record["token_vis"][candidate] for candidate in candidates]
    pooled = np.concatenate(token_kl) if any(len(values) for values in token_kl) else np.zeros(1)
    low, high = np.quantile(pooled, CALIBRATION_QUANTILES, method="linear")
    spread = max(high - low, CALIBRATION_FLOOR)
    confidence, fused, tokenwise = [], [], []
    for kl_values, vis_values in zip(token_kl, token_vis):
        if len(kl_values) == 0:
            confidence.append(0.0)
            tokenwise.append(0.0)
            continue
        calibrated = np.clip((kl_values - low) / spread, 0.0, 1.0)
        confidence.append(float(calibrated.mean()))
        tokenwise.append(float((calibrated * vis_values).mean()))
    confidence = np.array(confidence)
    visual = record["V"][candidates]
    return visual * confidence, np.array(tokenwise), confidence


def method_names(vision):
    names = ["first", "sc", "max_logp", "max_selfcert", "selfcert_rwc", "shortest"]
    if vision:
        names += ["max_vision", "vision_rwc", "max_visconf", "visconf_rwc", "max_tokenfusion", "tokenfusion_rwc"]
    return names


def select_all(record, candidates, vision):
    """Selected answer group for every method on this candidate prefix."""
    groups = [record["groups"][candidate] for candidate in candidates]
    if len(candidates) == 1:
        return {name: groups[0] for name in method_names(vision)}
    selected = {"first": groups[0], "sc": self_consistency(groups)[0]}
    logp = record["L"][candidates]
    selfcert = record["K"][candidates]
    selected["max_logp"] = groups[highest_score(logp)]
    selected["max_selfcert"] = groups[highest_score(selfcert)]
    selected["selfcert_rwc"] = rwc(groups, selfcert)[0]
    selected["shortest"] = groups[highest_score(-record["tokens"][candidates].astype(np.float64))]
    if vision:
        visual = record["V"][candidates]
        visconf, tokenfusion, _ = visconf_scores(record, candidates)
        selected["max_vision"] = groups[highest_score(visual)]
        selected["vision_rwc"] = rwc(groups, visual)[0]
        selected["max_visconf"] = groups[highest_score(visconf)]
        selected["visconf_rwc"] = rwc(groups, visconf)[0]
        selected["max_tokenfusion"] = groups[highest_score(tokenfusion)]
        selected["tokenfusion_rwc"] = rwc(groups, tokenfusion)[0]
    return selected


def stopping_features(record, candidates, vision):
    """Evidence observed after revealing these candidates (used by both stopping controllers)."""
    groups = [record["groups"][candidate] for candidate in candidates]
    winner, counts, _ = self_consistency(groups)
    top = max(counts.values())
    unique_leader = sum(1 for value in counts.values() if value == top) == 1
    if vision:
        quality = visconf_scores(record, candidates)[0]
    else:
        quality = record["K"][candidates]
    rwc_winner, votes = rwc(groups, quality)
    return {"sc": winner, "unique": unique_leader and winner != NO_ANSWER, "support": top / len(groups),
            "rwc_agrees": rwc_winner == winner, "margin": rwc_margin(votes)}

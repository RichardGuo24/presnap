"""Guards for the temporal split — the anti-leakage core of Phase 3.

Hermetic: exercises the pure split_frame() on synthetic game_ids, no DB.
"""

import polars as pl

from presnap import contract
from presnap.split import SPLIT_NAMES, split_frame


def _synthetic() -> pl.DataFrame:
    # a few games per season across the whole range
    rows = []
    for year in range(2010, 2025):
        for g in range(4):
            rows.append({"game_id": f"{year}_{g:02d}_AAA_BBB", "play_id": g})
    return pl.DataFrame(rows)


def test_no_game_appears_in_two_splits():
    splits = split_frame(_synthetic())
    sets = {s: set(v["game_id"].to_list()) for s, v in splits.items()}
    names = list(sets)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap = sets[names[i]] & sets[names[j]]
            assert not overlap, f"{names[i]} and {names[j]} share games: {overlap}"


def test_splits_are_temporally_ordered():
    splits = split_frame(_synthetic())
    seasons = {s: set(v["season"].to_list()) for s, v in splits.items()}
    assert max(seasons["train"]) < min(seasons["val"])
    assert max(seasons["val"]) < min(seasons["cal"])
    assert max(seasons["cal"]) < min(seasons["test"])


def test_split_seasons_match_the_contract():
    splits = split_frame(_synthetic())
    for name in SPLIT_NAMES:
        got = set(splits[name]["season"].to_list())
        assert got == set(contract.SPLIT_SEASONS[name])


def test_cal_is_isolated_from_train_and_test():
    splits = split_frame(_synthetic())
    cal = set(splits["cal"]["game_id"].to_list())
    assert cal and not (cal & set(splits["train"]["game_id"].to_list()))
    assert not (cal & set(splits["test"]["game_id"].to_list()))

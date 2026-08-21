from recurrent.research.trajectory_seeding import build_trajectory_seed_records, derive_turn_request_seeds, stable_training_group_id


def test_independent_seeds_are_distinct_and_reproducible():
    kwargs = dict(base_seed=2026, global_step=7, batch_size=8, rollout_n=4, mode="independent")
    first = build_trajectory_seed_records(**kwargs)
    assert first == build_trajectory_seed_records(**kwargs)
    assert len({record["trajectory_seed"] for record in first}) == 8


def test_matched_mode_shares_only_within_source_group():
    records = build_trajectory_seed_records(base_seed=2026, global_step=7, batch_size=8, rollout_n=4, mode="matched")
    seeds = [record["trajectory_seed"] for record in records]
    assert len(set(seeds[:4])) == len(set(seeds[4:])) == 1
    assert seeds[0] != seeds[4]


def test_turn_seeds_follow_active_sample_indices():
    records = build_trajectory_seed_records(base_seed=2026, global_step=7, batch_size=8, rollout_n=4, mode="independent")
    base = [record["trajectory_seed"] for record in records]
    turn0 = derive_turn_request_seeds(base, [0, 2, 7], 0)
    assert turn0 == derive_turn_request_seeds(base, [0, 2, 7], 0)
    assert len(set(turn0)) == 3
    assert all(a != b for a, b in zip(turn0, derive_turn_request_seeds(base, [0, 2, 7], 1)))


def test_training_group_identity_is_deterministic_and_step_bound():
    first = stable_training_group_id(base_seed=2026, global_step=1, dataset_index=42)
    assert first == stable_training_group_id(base_seed=2026, global_step=1, dataset_index=42)
    assert first != stable_training_group_id(base_seed=2026, global_step=2, dataset_index=42)
    assert len(first) == 64

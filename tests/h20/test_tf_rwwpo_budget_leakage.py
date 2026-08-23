from tools.h20.audit_tf_rwwpo_budget_leakage import overlap_counts, source_identity


def row(index: int, question: str, context: str, answer: str):
    return {
        "prompt": [{"role": "user", "content": question}],
        "context": context,
        "reward_model": {"ground_truth": {"target": answer}},
        "extra_info": {"index": index},
    }


def public(identity):
    return {key: value for key, value in identity.items() if key != "prompt"}


def test_content_and_root_intersections_are_not_dataset_local_ids():
    train = [public(source_identity(row(7, "q", "c", "a")))]
    same = [public(source_identity(row(99, "q", "c", "a")))]
    counts = overlap_counts(train, same)
    assert counts["train_intersect_s128_content"] == 1
    assert counts["train_intersect_s128_root"] == 1
    assert counts["train_intersect_s128_dataset_local_semantic_id_diagnostic"] == 0


def test_root_overlap_detects_label_drift_without_exposing_answer():
    train = [public(source_identity(row(7, "q", "c", "a")))]
    changed_label = [public(source_identity(row(8, "q", "c", "different")))]
    counts = overlap_counts(train, changed_label)
    assert counts["train_intersect_s128_content"] == 0
    assert counts["train_intersect_s128_root"] == 1
    assert set(train[0]) == {
        "semantic_dataset_index", "question_hash", "context_hash",
        "ground_truth_hash", "root_key", "content_key",
    }


def test_disjoint_content_has_zero_intersection():
    train = [public(source_identity(row(1, "q1", "c1", "a1")))]
    test = [public(source_identity(row(1, "q2", "c2", "a2")))]
    counts = overlap_counts(train, test)
    assert counts["train_intersect_s128_content"] == 0
    assert counts["train_intersect_s128_root"] == 0
    assert counts["train_intersect_s128_dataset_local_semantic_id_diagnostic"] == 1

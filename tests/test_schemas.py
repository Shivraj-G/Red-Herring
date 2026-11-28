import copy
import json

import pytest
from pydantic import ValidationError

from pipeline.schemas import (
    LABELS,
    DatasetManifest,
    LabelEvent,
    ModelManifest,
    RunRecord,
    SegmentRecord,
    make_text_key,
)

# ---------- valid examples (modelled on the PRD's section 12 JSON) ----------

SEGMENT = {
    "segment_id": "seg_3fa91c07b2de",
    "text": "Hurry! Offer ends in 02:14:59",
    "text_key": "hurry offer ends in 00 00 00",
    "domain": "example-shop.in",
    "site_category": "shopping",
    "country": "IN",
    "page_kind": "product",
    "context": {
        "tag": "div",
        "role": None,
        "in_modal": False,
        "in_button_or_link": False,
        "prev_text": "Add to cart",
    },
    "source": "crawl",
    "sampling_source": "lexical",
    "dup_count": 14,
    "label": "URGENCY",
    "label_secondary": None,
    "multi": False,
    "needs_review": False,
    "label_status": "adjudicated",
    "split": "test_gold",
    "crawl_ts": "2026-11-24T09:12:44Z",
    "guideline_version": "1.1",
    "dataset_version": "0.1",
}

LABEL_EVENT = {
    "event_id": "evt_0001",
    "segment_id": "seg_3fa91c07b2de",
    "annotator": "me",
    "pass": 1,
    "label": "URGENCY",
    "label_secondary": None,
    "multi": False,
    "needs_review": False,
    "duration_ms": 4200,
    "guideline_version": "1.1",
    "ts": "2026-11-24T10:00:00Z",
    "note": "",
}

DATASET_MANIFEST = {
    "version": "0.1",
    "parent_version": None,
    "created_at": "2026-11-26T12:00:00Z",
    "git_commit": "abc1234",
    "guideline_version": "1.1",
    "files": [{"name": "train.jsonl", "sha256": "a" * 64, "rows": 100}],
    "class_counts": {"train": {"NONE": 60, "URGENCY": 40}},
    "domains_per_split": {"train": 30, "val": 5},
    "source_counts": {"crawl": 100},
    "agreement": {"intra_kappa": 0.8, "inter_kappa": None},
    "notes": "",
}

RUN_RECORD = {
    "run_id": "run_2026-12-01_m4_r8_s1",
    "git_commit": "abc1234",
    "config_path": "training/configs/m4_lora_r8.yaml",
    "dataset_version": "0.1",
    "dataset_hash": "b" * 64,
    "method": "lora",
    "base_model": "some-base-model",
    "lora": {"r": 8, "alpha": 16, "dropout": 0.1, "targets": ["query", "value"]},
    "seed": 1,
    "hardware": "Kaggle T4",
    "train_time_s": 120.5,
    "best_epoch": 3,
    "trainable_params": 1000,
    "total_params": 100000,
    "metrics": {"val": {"macro_f1": 0.7, "per_class_f1": {"NONE": 0.9}}},
    "test_evaluated": False,
    "artifacts": {"model": "models/m4", "val_logits": "models/m4/val.npy"},
    "created_at": "2026-12-01T12:30:00Z",
    "notes": "",
}

MODEL_MANIFEST = {
    "model_id": "red-herring",
    "version": "0.1.0",
    "labels": list(LABELS),
    "max_len": 64,
    "input_format": "text",
    "precision": "int8-dynamic",
    "tokenizer_file": "tokenizer.json",
    "files": [{"name": "model.onnx", "sha256": "c" * 64, "bytes": 1000}],
    "temperature": 1.0,
    "thresholds": {"careful": 0.9, "balanced": 0.7, "thorough": 0.5},
    "dataset_version": "0.1",
    "run_id": "run_2026-12-01_m4_r8_s1",
    "eval_summary": {
        "macro_f1": 0.7,
        "binary_precision_balanced": 0.8,
        "binary_recall_balanced": 0.75,
    },
    "runtime": {"onnxruntime_web": "1.20.0"},
}


def changed(base, **updates):
    """Copy of a valid example with some top-level fields replaced."""
    d = copy.deepcopy(base)
    d.update(updates)
    return d


# ---------- make_text_key ----------


def test_text_key_matches_prd_example():
    assert make_text_key("Hurry! Offer ends in 02:14:59") == "hurry offer ends in 00 00 00"


def test_text_key_collapses_whitespace():
    assert make_text_key("  Only   3 left!!  ") == "only 0 left"


# ---------- round trips ----------


@pytest.mark.parametrize(
    "model, data",
    [
        (SegmentRecord, SEGMENT),
        (LabelEvent, LABEL_EVENT),
        (DatasetManifest, DATASET_MANIFEST),
        (RunRecord, RUN_RECORD),
        (ModelManifest, MODEL_MANIFEST),
    ],
)
def test_round_trip(model, data):
    obj = model.model_validate(data)
    text = obj.model_dump_json(by_alias=True)
    assert model.model_validate_json(text) == obj


# ---------- SegmentRecord ----------


def test_segment_valid_normalises_domain():
    seg = SegmentRecord.model_validate(
        changed(SEGMENT, domain="https://WWW.Example-Shop.in/cart")
    )
    assert seg.domain == "example-shop.in"


@pytest.mark.parametrize(
    "updates",
    [
        {"label": "URGENT"},  # not one of the 7 labels
        {"text": "ab", "text_key": "ab"},  # too short
        {"text": "x" * 301, "text_key": "x" * 301},  # too long
        {"text_key": "wrong key"},  # does not match text
        {"multi": True},  # multi without a secondary label
        {"label_secondary": "SCARCITY"},  # secondary label without multi
        {"country": "india"},  # not alpha-2 or GLOBAL
        {"label_status": "unlabeled"},  # not in the doc's enum
        {"segment_id": "seg_xyz"},  # wrong format
        {"dup_count": 0},
        {"unexpected_field": 1},  # extra="forbid"
    ],
)
def test_segment_rejects_bad_values(updates):
    with pytest.raises(ValidationError):
        SegmentRecord.model_validate(changed(SEGMENT, **updates))


@pytest.mark.parametrize("missing", ["label", "split", "dataset_version", "domain"])
def test_segment_rejects_missing_required_field(missing):
    data = copy.deepcopy(SEGMENT)
    del data[missing]
    with pytest.raises(ValidationError):
        SegmentRecord.model_validate(data)


def test_segment_accepts_global_country():
    assert SegmentRecord.model_validate(changed(SEGMENT, country="GLOBAL")).country == "GLOBAL"


def test_segment_multi_with_secondary_is_valid():
    seg = SegmentRecord.model_validate(
        changed(SEGMENT, multi=True, label_secondary="SCARCITY")
    )
    assert seg.label_secondary == "SCARCITY"


# ---------- LabelEvent ----------


def test_label_event_parses_pass_key_from_json():
    ev = LabelEvent.model_validate_json(json.dumps(LABEL_EVENT))
    assert ev.pass_ == 1


def test_label_event_dumps_pass_key_with_alias():
    ev = LabelEvent.model_validate(LABEL_EVENT)
    dumped = json.loads(ev.model_dump_json(by_alias=True))
    assert "pass" in dumped
    assert "pass_" not in dumped


def test_label_event_rejects_pass_3():
    with pytest.raises(ValidationError):
        LabelEvent.model_validate(changed(LABEL_EVENT, **{"pass": 3}))


def test_label_event_multi_rule():
    with pytest.raises(ValidationError):
        LabelEvent.model_validate(changed(LABEL_EVENT, multi=True))


# ---------- DatasetManifest ----------


@pytest.mark.parametrize("version", ["1", "v0.1", "0.1.0"])
def test_dataset_manifest_rejects_bad_version(version):
    with pytest.raises(ValidationError):
        DatasetManifest.model_validate(changed(DATASET_MANIFEST, version=version))


def test_dataset_manifest_rejects_short_sha():
    bad = changed(DATASET_MANIFEST, files=[{"name": "t.jsonl", "sha256": "abc", "rows": 1}])
    with pytest.raises(ValidationError):
        DatasetManifest.model_validate(bad)


def test_dataset_manifest_rejects_unknown_split_name():
    with pytest.raises(ValidationError):
        DatasetManifest.model_validate(
            changed(DATASET_MANIFEST, domains_per_split={"training": 3})
        )


# ---------- RunRecord ----------


def test_run_record_lora_method_needs_lora_config():
    data = changed(RUN_RECORD)
    del data["lora"]
    with pytest.raises(ValidationError):
        RunRecord.model_validate(data)


def test_run_record_baseline_must_not_have_lora_config():
    with pytest.raises(ValidationError):
        RunRecord.model_validate(changed(RUN_RECORD, method="tfidf"))


def test_run_record_baseline_without_lora_is_valid():
    data = changed(RUN_RECORD, method="tfidf", base_model=None)
    del data["lora"]
    assert RunRecord.model_validate(data).method == "tfidf"


def test_run_record_rejects_unknown_method():
    with pytest.raises(ValidationError):
        RunRecord.model_validate(changed(RUN_RECORD, method="magic"))


def test_run_record_rejects_metrics_other_than_val():
    bad = changed(
        RUN_RECORD, metrics={"test": {"macro_f1": 0.7, "per_class_f1": {}}}
    )
    with pytest.raises(ValidationError):
        RunRecord.model_validate(bad)


# ---------- ModelManifest ----------


def test_model_manifest_rejects_labels_in_wrong_order():
    swapped = list(LABELS)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    with pytest.raises(ValidationError):
        ModelManifest.model_validate(changed(MODEL_MANIFEST, labels=swapped))


def test_model_manifest_rejects_missing_label():
    with pytest.raises(ValidationError):
        ModelManifest.model_validate(changed(MODEL_MANIFEST, labels=list(LABELS)[:-1]))


def test_model_manifest_rejects_careful_below_thorough():
    bad = changed(
        MODEL_MANIFEST, thresholds={"careful": 0.3, "balanced": 0.5, "thorough": 0.9}
    )
    with pytest.raises(ValidationError):
        ModelManifest.model_validate(bad)


def test_model_manifest_rejects_unknown_precision():
    with pytest.raises(ValidationError):
        ModelManifest.model_validate(changed(MODEL_MANIFEST, precision="fp16"))


def test_model_manifest_needs_at_least_one_file():
    with pytest.raises(ValidationError):
        ModelManifest.model_validate(changed(MODEL_MANIFEST, files=[]))
import re
from datetime import datetime
from enum import Enum
from typing import Annotated, Literal

import tldextract
from pydantic import (AfterValidator, BaseModel, ConfigDict, Field,
                      model_validator)

_tld = tldextract.TLDExtract(suffix_list_urls=())  # bundled list, no network


def make_text_key(text: str) -> str:
    text = re.sub(r"\d", "0", text.lower())
    return re.sub(r"[\W_]+", " ", text).strip()


def _registrable(v: str) -> str:
    return (_tld(v).registered_domain or v).lower()


Domain = Annotated[str, AfterValidator(_registrable)]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Country = Annotated[str, Field(pattern=r"^[A-Z]{2}$")]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class Label(str, Enum):
    NONE = "NONE"
    URGENCY = "URGENCY"
    SCARCITY = "SCARCITY"
    SOCIAL_PROOF = "SOCIAL_PROOF"
    CONFIRMSHAMING = "CONFIRMSHAMING"
    SNEAKING = "SNEAKING"
    OBSTRUCTION = "OBSTRUCTION"


LABELS = [l.value for l in Label]  # order = class ids 0..6


class SiteCategory(str, Enum):
    SHOP = "shopping"
    TRAVEL = "travel"
    FOOD = "food"
    TICKET = "ticketing"
    SAAS = "saas"
    OTHER = "other"


class PageKind(str, Enum):  # CHECK against 12.1
    PRODUCT = "product"
    LISTING = "listing"
    CART = "cart"
    OTHER = "other"


class Source(str, Enum):  # CHECK against 12.1
    CRAWL = "crawl"
    MATHUR = "mathur"
    YADA = "yada"
    SYNTHETIC = "synthetic"


class SamplingSource(str, Enum):
    LEXICAL = "lexical"
    RANDOM = "random"
    UNCERTAIN = "uncertain"


class LabelStatus(str, Enum):  # CHECK against 12.1
    UNLABELED = "unlabeled"
    LABELED = "labeled"
    ADJUDICATED = "adjudicated"


class Split(str, Enum):
    TRAIN = "train"
    VAL = "val"
    TEST_GOLD = "test_gold"
    TEST_NATURAL = "test_natural"


class SegmentContext(Strict):
    tag: str
    role: str | None = None
    in_modal: bool
    in_button_or_link: bool
    prev_text: str | None = Field(default=None, max_length=80)


class SegmentRecord(Strict):
    segment_id: str = Field(pattern=r"^seg_[0-9a-f]{12}$")
    text: str = Field(min_length=3, max_length=300)
    text_key: str
    domain: Domain
    site_category: SiteCategory
    country: Country
    page_kind: PageKind
    context: SegmentContext
    source: Source
    sampling_source: SamplingSource
    dup_count: int = Field(ge=1)
    label: Label | None = None
    label_secondary: Label | None = None
    multi: bool = False
    needs_review: bool = False
    label_status: LabelStatus = LabelStatus.UNLABELED
    split: Split | None = None
    crawl_ts: datetime
    guideline_version: str
    dataset_version: str | None = None

    @model_validator(mode="after")
    def _consistent(self):
        if self.text_key != make_text_key(self.text):
            raise ValueError("text_key does not match text")
        if self.multi != (self.label_secondary is not None):
            raise ValueError("multi must be true exactly when label_secondary is set")
        return self


class LabelEvent(Strict):
    event_id: str
    segment_id: str = Field(pattern=r"^seg_[0-9a-f]{12}$")
    annotator: Literal["me", "friend", "adjudicated"]
    pass_: Literal[1, 2] = Field(alias="pass")  # `pass` is a Python keyword
    label: Label
    label_secondary: Label | None = None
    multi: bool = False
    needs_review: bool = False
    duration_ms: int = Field(ge=0)
    guideline_version: str
    ts: datetime
    note: str = ""

    @model_validator(mode="after")
    def _consistent(self):
        if self.multi != (self.label_secondary is not None):
            raise ValueError("multi must be true exactly when label_secondary is set")
        return self


class FileEntry(Strict):
    name: str
    sha256: Sha256
    rows: int | None = Field(default=None, ge=0)   # datasets
    bytes: int | None = Field(default=None, ge=0)  # model files


class Agreement(Strict):
    intra_kappa: float | None = Field(default=None, ge=-1, le=1)
    inter_kappa: float | None = Field(default=None, ge=-1, le=1)


class DatasetManifest(Strict):
    version: str = Field(pattern=r"^\d+\.\d+$")
    parent_version: str | None = None
    created_at: datetime
    git_commit: str
    guideline_version: str
    files: list[FileEntry]
    class_counts: dict[Split, dict[str, int]]
    domains_per_split: dict[Split, int]
    source_counts: dict[str, int]
    agreement: Agreement
    notes: str = ""


class LoraConfig(Strict):
    r: int = Field(gt=0)
    alpha: int = Field(gt=0)
    dropout: float = Field(ge=0, le=1)
    targets: list[str] = Field(min_length=1)


class ValMetrics(Strict):
    macro_f1: float = Field(ge=0, le=1)
    per_class_f1: dict[str, float]


class RunArtifacts(Strict):
    model: str
    val_logits: str


class RunRecord(Strict):
    run_id: str
    git_commit: str
    config_path: str
    dataset_version: str
    dataset_hash: str
    method: Literal["majority", "rules", "tfidf", "full", "lora"]  # CHECK
    base_model: str | None = None  # baselines have none
    lora: LoraConfig | None = None
    seed: int
    hardware: str
    train_time_s: float = Field(ge=0)
    best_epoch: int = Field(ge=0)
    trainable_params: int = Field(ge=0)
    total_params: int = Field(ge=0)
    metrics: dict[Literal["val"], ValMetrics]
    test_evaluated: bool = False
    artifacts: RunArtifacts
    created_at: datetime
    notes: str = ""

    @model_validator(mode="after")
    def _lora_matches_method(self):
        if (self.method == "lora") != (self.lora is not None):
            raise ValueError("lora config is required for method=lora and only then")
        return self


class Thresholds(Strict):
    careful: float = Field(ge=0, le=1)
    balanced: float = Field(ge=0, le=1)
    thorough: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def _ordered(self):
        if not (self.careful >= self.balanced >= self.thorough):
            raise ValueError("careful must flag least, thorough most")
        return self


class EvalSummary(Strict):
    macro_f1: float = Field(ge=0, le=1)
    binary_precision_balanced: float = Field(ge=0, le=1)
    binary_recall_balanced: float = Field(ge=0, le=1)


class RuntimeInfo(Strict):
    onnxruntime_web: str


class ModelManifest(Strict):
    model_id: str
    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    labels: list[str]
    max_len: int = Field(gt=0)
    input_format: Literal["text", "context"]
    precision: Literal["fp32", "int8-dynamic"]
    tokenizer_file: str
    files: list[FileEntry] = Field(min_length=1)
    temperature: float = Field(gt=0)
    thresholds: Thresholds
    dataset_version: str
    run_id: str
    eval_summary: EvalSummary
    runtime: RuntimeInfo

    @model_validator(mode="after")
    def _labels_in_order(self):
        if self.labels != LABELS:
            raise ValueError(f"labels must be exactly {LABELS}, in that order")
        return self
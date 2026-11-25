from pydantic import (AfterValidator, BaseModel, ConfigDict, Field, model_validator)
from enum import Enum
import re
from typing import Annotated

import tldextract
_tld = tldextract.TLDExtract(suffix_list_urls=())

class SiteCategories(Enum):
    SHOP = "shopping"
    TRAVEL = "travel" 
    FOOD = "food"
    TICKET = "ticketing"
    SAAS = "saas"
    OTHER = "other"

def make_text_key(text: str) -> str:
    text = re.sub(r"\d", "0", text.lower())
    return re.sub(r"[\W_]+", " ", text).strip()

def _registrable(v: str) -> str:
    return (_tld(v).registered_domain or v).lower()

Domain = Annotated[str, AfterValidator(_registrable)]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Country = Annotated[str, Field(pattern=r"^[A-Z]{2}$")]

class SegmentRecord(BaseModel):
    segment_id: str
    text: str = Field(min_length=3,max_length=300)
    text_key: DedupeForm
    domain: RegistrableDomain
    site_category: SiteCategories
    country: str
    page_kind: 
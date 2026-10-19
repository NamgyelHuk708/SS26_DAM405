"""Pydantic request/response models for the prediction API.

FastAPI uses these to validate incoming JSON (a malformed request becomes a
422 response automatically) and to generate the OpenAPI docs at ``/docs``.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

EXAMPLE_REQUEST = {
    "age": 39,
    "workclass": "State-gov",
    "fnlwgt": 77516,
    "education": "Bachelors",
    "education-num": 13,
    "marital-status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital-gain": 2174,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}


class PredictionRequest(BaseModel):
    """One Adult-census person, with the RAW (untransformed) feature values.

    JSON keys use the original Adult column names (e.g. ``education-num``);
    hyphens are not valid in Python names, so attributes use snake_case + alias.
    Categorical fields are required but may be ``null`` (missing, as in training).
    """

    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
        json_schema_extra={"examples": [EXAMPLE_REQUEST]},
    )

    age: int = Field(..., ge=17, le=120)
    workclass: Optional[str] = Field(...)
    fnlwgt: int = Field(..., ge=0)
    education: Optional[str] = Field(...)
    education_num: int = Field(..., alias="education-num", ge=1, le=16)
    marital_status: Optional[str] = Field(..., alias="marital-status")
    occupation: Optional[str] = Field(...)
    relationship: Optional[str] = Field(...)
    race: Optional[str] = Field(...)
    sex: Optional[str] = Field(...)
    capital_gain: int = Field(..., alias="capital-gain", ge=0)
    capital_loss: int = Field(..., alias="capital-loss", ge=0)
    hours_per_week: int = Field(..., alias="hours-per-week", ge=1, le=168)
    native_country: Optional[str] = Field(..., alias="native-country")

    def to_row(self) -> dict:
        """Feature dict keyed by the ORIGINAL column names."""
        return self.model_dump(by_alias=True)

    def to_frame(self) -> pd.DataFrame:
        """One-row DataFrame in the format the sklearn pipeline was trained on."""
        return pd.DataFrame([self.to_row()])


class PredictionResponse(BaseModel):
    """What /predict returns."""

    # "model_version" would clash with pydantic's protected "model_" namespace.
    model_config = ConfigDict(protected_namespaces=())

    prediction: int = Field(..., description="0 = <=50K, 1 = >50K")
    label: str = Field(..., description="Human-readable class")
    probability: float = Field(..., ge=0, le=1, description="P(income > 50K)")
    model_version: str = Field(..., description="Registry version of the champion")
    request_id: str = Field(..., description="Same id as in the service logs")

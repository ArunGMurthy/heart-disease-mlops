"""Pydantic request/response schemas for the /predict endpoint."""
from __future__ import annotations

from pydantic import BaseModel, Field


class PatientFeatures(BaseModel):
    age: float = Field(..., json_schema_extra={"example": 63})
    sex: int = Field(..., ge=0, le=1, json_schema_extra={"example": 1})
    cp: int = Field(
        ..., ge=1, le=4, description="Chest pain type (1-4)", json_schema_extra={"example": 3}
    )
    trestbps: float = Field(..., json_schema_extra={"example": 145})
    chol: float = Field(..., json_schema_extra={"example": 233})
    fbs: int = Field(..., ge=0, le=1, json_schema_extra={"example": 1})
    restecg: int = Field(..., ge=0, le=2, json_schema_extra={"example": 0})
    thalach: float = Field(..., json_schema_extra={"example": 150})
    exang: int = Field(..., ge=0, le=1, json_schema_extra={"example": 0})
    oldpeak: float = Field(..., json_schema_extra={"example": 2.3})
    slope: int = Field(
        ...,
        ge=1,
        le=3,
        description="Slope of peak exercise ST segment (1-3)",
        json_schema_extra={"example": 1},
    )
    ca: float = Field(
        ...,
        ge=0,
        le=3,
        description="Number of major vessels (0-3)",
        json_schema_extra={"example": 0},
    )
    thal: float = Field(
        ...,
        description="Thalassemia: 3=normal, 6=fixed defect, 7=reversible defect",
        json_schema_extra={"example": 3},
    )


class PredictionResponse(BaseModel):
    prediction: int
    label: str
    confidence: float
    model_version: str

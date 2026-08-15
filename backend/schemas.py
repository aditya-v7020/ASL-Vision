from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    classes_count: int
    framework: str

class TopPrediction(BaseModel):
    class_name: str = Field(..., alias="class")
    confidence: float

    class Config:
        populate_by_name = True

class QualityMetrics(BaseModel):
    brightness: float
    contrast: float
    sharpness: float
    is_quality_acceptable: bool

class HandPresenceInfo(BaseModel):
    has_hand: bool
    status: str
    hand_score: Optional[float] = None
    fg_ratio: Optional[float] = None
    cnt_ratio: Optional[float] = None
    edge_density: Optional[float] = None
    reason: Optional[str] = None

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    top_predictions: List[TopPrediction]
    is_uncertain: bool = False
    uncertainty_reason: Optional[str] = None
    quality_metrics: Optional[QualityMetrics] = None
    hand_presence: Optional[HandPresenceInfo] = None
    prototype_match: Optional[str] = None
    prototype_similarity: Optional[float] = None
    reference_sample: Optional[str] = None

class ClassesResponse(BaseModel):
    classes: List[str]
    total: int


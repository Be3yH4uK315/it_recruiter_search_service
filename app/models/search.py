from typing import Optional, List
from pydantic import BaseModel, Field, field_validator

# --- SEARCH ---
class SearchFilters(BaseModel):
    role: Optional[str] = None
    must_skills: Optional[List[str]] = Field(default_factory=list)
    nice_skills: Optional[List[str]] = Field(default_factory=list)
    experience_min: Optional[float] = None
    experience_max: Optional[float] = None
    location: Optional[str] = None
    work_modes: Optional[List[str]] = Field(default_factory=list)
    exclude_ids: Optional[List[str]] = Field(default_factory=list)

    @field_validator('must_skills', 'nice_skills', mode='before')
    @classmethod
    def normalize_skills(cls, v: List[str]) -> List[str]:
        return [skill.strip().lower() for skill in v if skill.strip()]

    @field_validator('experience_min', 'experience_max')
    @classmethod
    def validate_experience(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and v < 0:
            raise ValueError("Experience must be non-negative")
        return v
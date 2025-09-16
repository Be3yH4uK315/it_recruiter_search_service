from typing import Optional, List
from pydantic import BaseModel, Field

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
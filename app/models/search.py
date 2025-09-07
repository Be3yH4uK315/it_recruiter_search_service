from typing import Optional, List
from pydantic import BaseModel


class SearchFilters(BaseModel):
    role: Optional[str] = None
    must_skills: Optional[List[str]] = []
    nice_skills: Optional[List[str]] = []
    experience_min: Optional[float] = None
    experience_max: Optional[float] = None
    location: Optional[str] = None
    work_modes: Optional[List[str]] = []
    exclude_ids: Optional[List[str]] = []
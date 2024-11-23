from pydantic import BaseModel, Field
from typing import List, Optional, Literal


class TimeCurrentRole(BaseModel):
    qualifier: Literal['greater_than', 'less_than', 'equal_to']
    duration: int
    unit: Literal['year', 'month']


class SearchTemplate(BaseModel):
    person_title: str
    job_title_keywords: Optional[List[str]] = None

    about_job: str
    job_description_keywords: Optional[List[str]] = None

    time_current_role: Optional[TimeCurrentRole] = None

    location: Optional[str] = None
    previous_locations: Optional[List[str]] = None

    companies: Optional[List[str]] = None
    education: Optional[List[str]] = None


class SearchCompaniesInput(BaseModel):
    organization_num_employees_ranges: Optional[List[str]] = Field(None, description="An array of intervals to include organizations having a number of employees in a range. e.g., ['1,100', '1,1000']")
    organization_locations: Optional[List[str]] = Field(None, description="An array of strings denoting allowed locations of organization headquarters. e.g., ['United States']")
    organization_not_locations: Optional[List[str]] = Field(None, description="An array of strings denoting un-allowed locations of organization headquarters. e.g., ['India']")
    q_organization_keyword_tags: Optional[List[str]] = Field(None, description="An array of strings denoting the keywords an organization should be associated with. e.g., ['sales strategy', 'lead']")
    q_organization_name: Optional[str] = Field(None, description="A string representing the name of the organization we want to filter. e.g., 'Apollo.io'")


class SearchCompaniesInputs(BaseModel):
    """List of SearchCompaniesInput."""
    search_inputs: List[SearchCompaniesInput] = Field(description="List of SearchCompaniesInput")

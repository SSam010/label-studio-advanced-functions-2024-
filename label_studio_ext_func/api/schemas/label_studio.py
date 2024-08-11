from typing import Optional

from pydantic import BaseModel
from pydantic.v1 import validator


class RawProjects(BaseModel):
    projects: list[int]


class Projects(BaseModel):
    projects: list[int]
    drop_space: bool = True

    @validator('projects')
    def check_projects(self, projects):
        if len(projects) > 2:
            return projects[:2]
        return projects


class ProjectsMerge(BaseModel):
    projects: list[int]
    title: Optional[str] = None
    drop_space: bool = True
    only_diff: bool = False

    @validator('projects')
    def check_projects(self, projects):
        if len(projects) > 2:
            return projects[:2]
        return projects


class ProjectsMergeRelations(BaseModel):
    projects: list[int]
    title: Optional[str] = None

    @validator('projects')
    def check_projects(self, projects):
        if len(projects) > 2:
            return projects[:2]
        return projects


class DataForNewProject(BaseModel):
    data: list
    relation_labels: list
    title: Optional[str] = 'title'

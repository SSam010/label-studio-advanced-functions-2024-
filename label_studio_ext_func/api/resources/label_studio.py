import logging
from typing import Optional

from fastapi import APIRouter, Request, Response
from fastapi.templating import Jinja2Templates

from api.core.label_studio import LabelWork
from api.schemas.label_studio import (
    Projects,
    ProjectsMerge,
    RawProjects,
    DataForNewProject,
    ProjectsMergeRelations,
)
from config import labels


logger = logging.getLogger()

router = APIRouter(prefix="", tags=["label_studio_ext_func"])
templates = Jinja2Templates(directory="front/templates")


@router.get("/get_project/{pr_id}")
async def get_projects(request: Request, pr_id: int):
    """
    Retrieves project data based on the provided project ID.
    Args:
        pr_id : int
            The project ID for which the data is to be fetched.

    Returns:
        dict: A dictionary containing the result of the fetched project data.
    """
    label = LabelWork(labels=labels)
    return {"result": await label.fetch_data(pr_id)}


@router.post("/get_projects")
async def get_projects(request: Request, data: RawProjects):
    """
    Retrieves data for multiple projects based on the provided project IDs.
    Args:
        data : RawProjects
            The request body containing the project IDs. It should be an instance of RawProjects, which includes
            a list of project IDs.

    Returns:
        dict: A dictionary containing the results of the fetched project data. Each key is a project ID and
        each value is the corresponding project data.
    """
    label = LabelWork(labels=labels)
    return {"result": {pr_id: await label.fetch_data(pr_id) for pr_id in data.projects}}


@router.post("/merge_projects")
async def merge_projects(request: Request, data: ProjectsMerge):
    """
    Merges multiple projects into a new project.
    Args:
        data : ProjectsMerge
            The request body containing the projects to be merged and additional parameters. It should be an
            instance of ProjectsMerge, which includes:
                - projects: List of project IDs to be merged.
                - title: Title of the new merged project.
                - drop_space: Boolean flag indicating whether to drop space in the merged project.
                - only_diff: Boolean flag indicating whether to include only differences in the merged project.

    Returns:
        dict: A dictionary containing the result of the merge operation. The key "new_project_id" holds the ID
        of the newly created merged project.
    """
    label = LabelWork(labels=labels)
    new_project_id = await label.merge_projects(
        projects=data.projects,
        title=data.title,
        drop_space=data.drop_space,
        only_diff=data.only_diff,
    )
    return {"result": {"new_project_id": new_project_id}}


@router.post("/merge_relations")
async def merge_relations(request: Request, data: ProjectsMergeRelations):
    """
    Merges relations from multiple projects into a new project.
    Args:
        data : ProjectsMergeRelations
            The request body containing the projects to be merged and the title for the new project.
            It should be an instance of ProjectsMergeRelations, which includes:
                - projects: List of project IDs whose relations are to be merged.
                - title: Title of the new merged project.

    Returns:
        dict: A dictionary containing the result of the merge operation. The key "new_project_id" holds the ID
        of the newly created merged project.
    """

    label = LabelWork(labels=labels)
    new_project_id = await label.merge_relations(
        projects=data.projects,
        title=data.title,
    )
    return {"result": {"new_project_id": new_project_id}}


@router.post("/find_difference")
async def find_difference(request: Request, data: Projects):
    """
    Finds differences between multiple projects.

    Args:
        data : Projects
            The request body containing the projects to be compared and additional parameters. It should be
            an instance of Projects, which includes:
                - projects: List of project IDs to be compared.
                - drop_space: Boolean flag indicating whether to ignore spaces in the comparison.

    Returns:
        dict: A dictionary containing the result of the difference operation.
    """
    label = LabelWork(labels=labels)
    return {
        "result": await label.find_difference(
            projects=data.projects, drop_space=data.drop_space
        )
    }


@router.post("/count_statistics")
async def count_statistics(request: Request, data: Projects):
    """
    This method counts statistics for multiple projects.

    Args:
        data : Projects
            The request body containing the projects for which statistics are to be counted and additional
            parameters. It should be an instance of Projects, which includes:
                - projects: List of project IDs for which statistics are to be counted.
                - drop_space: Boolean flag indicating whether to ignore spaces in the statistical analysis.

    Returns:
        dict: A dictionary containing the result of the statistical analysis.
    """
    label = LabelWork(labels=labels)
    return {
        "result": await label.count_statistics(
            projects=data.projects, drop_space=data.drop_space
        )
    }


@router.get("/visual/{projects}")
async def visual(request: Request, projects, drop_space: Optional[bool] = True):
    """
    This method generates a visual representation (histogram) of statistics for multiple projects.

    Args:
        projects : str
            A string containing project IDs separated by hyphens (e.g., "1-2-3").
        drop_space : Optional[bool]
            An optional boolean flag indicating whether to ignore spaces in the statistical analysis. Defaults
            to True.

    Returns:
        TemplateResponse: An HTML response containing the rendered histogram with the statistical data.
    """

    projects_list = projects.split("-")
    label = LabelWork(labels=labels)
    data = await label.count_statistics(projects_list, drop_space=drop_space)
    return templates.TemplateResponse(
        "histogram.html", {"request": request, "data": data}
    )


@router.post("/remove_outer_spaces")
async def remove_outer_spaces(request: Request, project_id: int):
    """
    Removes unnecessary problems from selected annotations

    Args:
        project_id : int
            The ID of the project from which outer spaces are to be removed.

    Returns:
        dict: A dictionary containing the result message if the operation is successful.
        Response: An error response with status code 500 if there is an error during the operation.
    """
    label = LabelWork(labels=labels)
    data = await label.fetch_data(project_id)
    if await label.drop_space_in_start_and_end(data, with_update=True):
        return {"result": "Spaces were deleted"}
    else:
        return Response(content="There was error", status_code=500)


@router.post("/make_sentences_as_tasks_with_possible_relations")
async def make_sentences_as_tasks_with_possible_relations(
    request: Request, project_id: int
):
    entity_types = {
        "car",
        "ferry",
        "airplane",
    }
    label = LabelWork(labels=labels)
    data = await label.fetch_data(project_id)

    split_data = await label.make_sentences_as_tasks_with_possible_relations(
        project_data=data, entity_types=entity_types
    )
    new_project_id = await label.create_project_with_data(
        data=split_data, title=f"{project_id}_Sentences_as_task_from_project"
    )
    return {"new_project_id": new_project_id}

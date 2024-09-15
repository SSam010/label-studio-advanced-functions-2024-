import json
import logging
import random
import xml.etree.ElementTree as ET
from typing import Optional, Union, Iterable

import aiohttp
from nltk import sent_tokenize

from config import LB
from utils.utils import find_first

logger = logging.getLogger()


class LabelWork:
    """
    The class allows you to expand the basic functionality of the free version of Label Studio.
    The methods are designed to simplify the validation of the work of annotators:
        - find differing annotations, correct the markup associated with capturing extra spaces,
          create a combined version of the projects (with the ability to choose: keep the original markup
          or leave only the different ones) and more.
    """

    def __init__(self, labels: Union[list, tuple], url: str = None, token: str = None):
        """
        Constructs all the necessary attributes for the LabelWork object.

        Args:
            labels : Union[list, tuple]
                The labels which are in the project.
            url : str, optional
                The URL. By default it is taken from the environment variable.
            token : str, optional
                The label studio access token. By default it is taken from the environment variable.
        """
        self.labels = labels
        self.url = url if url else f"{LB.LB_SERVICE_NAME}:{LB.LABEL_STUDIO_PORT}"
        self.token = token if token else LB.LABEL_STUDIO_API
        self.headers = {"Authorization": f"Token {self.token}"}

    async def count_statistics(self, projects: list[int], drop_space: bool = True):
        """
        Count statistics between given projects.

        Args:
            projects : list[int]
                The list of project IDs to compare.
            drop_space : bool, optional
                If True, leading and trailing spaces are removed from the project data. Defaults to True.

        Returns:
            dict: Statistics of comparing between the projects.
        """
        projects_data = {pr_id: await self.fetch_data(pr_id) for pr_id in projects}

        for project in projects_data.values():
            if drop_space:
                await self.drop_space_in_start_and_end(project, with_update=False)
            self._split_anns_and_rels(project)
            self._update_relation_information(project)

        return {
            "amount_anns_per_project": self._count_annotations_per_label(projects_data),
            "difference_between_projects": self._find_difference_between_projects(
                projects_data, only_amount=True
            ),
        }

    async def find_difference(
        self, projects: list[int], drop_space: bool = True
    ) -> dict:
        """
        Finds the differences between given projects.

        Args:
            projects : list[int]
                The list of project IDs to compare.
            drop_space : bool, optional
                If True, leading and trailing spaces are removed from the project data. Defaults to True.

        Returns:
            dict: The differences between the projects.
        """
        projects_data = {pr_id: await self.fetch_data(pr_id) for pr_id in projects}
        for project in projects_data.values():
            if drop_space:
                await self.drop_space_in_start_and_end(project, with_update=False)
            self._split_anns_and_rels(project)

        return self._find_difference_between_projects(projects_data)

    async def merge_projects(
        self,
        projects: list[int],
        label_config: Optional[str] = None,
        title: Optional[str] = None,
        drop_space: bool = True,
        only_diff: bool = False,
    ):
        """
        Merges multiple projects into one. All of relations will be dropped

        This method fetches the data for each project in the given list and optionally removes leading and trailing
        spaces from the project data. It then merges the projects by combining their tasks.

        If `only_diff` is set to True, it only keeps the tasks that are different across projects.

        The method then creates a new project with the merged tasks. The label configuration and title of the new
        project can be specified. If not, default values are used.

        Args:
            projects : list[int]
                The list of project IDs to merge.
            label_config : str, optional
                The label configuration for the new project. If not provided, a default configuration is used.
            title : str, optional
                The title for the new project. If not provided, a default title is used.
            drop_space : bool, optional
                If True, leading and trailing spaces are removed from the project data. Defaults to True.
            only_diff : bool, optional
                If True, only the tasks that are different across projects are kept in the new project. Defaults to False.

        Returns:
            int: The ID of the new project.
        """
        projects_data = {pr_id: await self.fetch_data(pr_id) for pr_id in projects}
        for project in projects_data.values():
            if drop_space:
                await self.drop_space_in_start_and_end(project, with_update=False)
            # need for avoiding error, if there are any relations
            self._split_anns_and_rels(project)

        if only_diff:
            projects_data = self.drop_simular_annotations(projects_data)
        result = self._merging_projects(projects_data)

        title = title or f"Union_project " + ", ".join(map(str, projects))

        label_config = label_config or self.make_default_views_layout(projects)

        new_project_id = await self._create_new_project(
            label_config=label_config, title=title
        )
        if new_project_id is None:
            raise RuntimeError("project wasn't created")
        await self.upload_data(new_project_id, result)
        return new_project_id

    async def get_current_relations_from_projects(
        self, projects: list[int]
    ) -> set[set]:
        relations = set()
        for project_id in projects:
            project_meta_info = await self.get_meta_project_info(project_id)
            root = ET.fromstring(project_meta_info["label_config"])
            relations.update(
                {relation.get("value") for relation in root.findall(".//Relation")}
            )

        return relations

    async def merge_relations(self, projects: list[int], title: Optional[str] = None):
        """
        Merges two projects relations. All the annotations have to be simular.
        Only different relations will be left.

        The method then creates a new project with the merged tasks(only relations). The label configuration
         and title of the new project can be specified. If not, default values are used.

        Args:
            projects : list[int]
                The list of project IDs to merge.
            title : str, optional
                The title for the new project. If not provided, a default title is used.
        Returns:
            int: The ID of the new project.
        """
        projects_data = {pr_id: await self.fetch_data(pr_id) for pr_id in projects}
        relations = await self.get_current_relations_from_projects(projects)
        relations.update((*projects, "dif_direction", "dif_labels", "only_one_project"))

        for project in projects_data.values():
            self._split_anns_and_rels(project)
            self._update_relation_information(project)
        result = self._merge_relations_process(projects_data)

        title = title or f"Union_project " + ", ".join(map(str, projects))

        return await self.create_project_with_data(
            data=result, title=title, relation_labels=relations
        )

    def _find_difference_between_projects(
        self, projects_data: dict, only_amount: bool = False
    ) -> dict:
        """
        Finds the difference between projects.

        This method iterates over the projects data and compares each pair of projects. For each pair, it finds the
        differences between their tasks. The differences are categorized into label differences and coordinate differences.

        If `only_amount` is set to True, the method only counts the number of differences and returns these counts.
        Otherwise, it returns the actual differences.

        Args:
            projects_data : dict
                The projects data. Each key is a project ID and each value is a list of tasks for that project.
            only_amount : bool, optional
                If True, only the amount of differences is returned. If False, the actual differences are returned.
                Defaults to False.
        Returns:
            dict: The differences between projects. Each key is a pair of project IDs and each value is a dictionary
                containing the differences between the tasks of those projects. If `only_amount` is True, the values are
                the counts of differences.
        """

        differ = {}
        data_list = list(projects_data.items())

        for i in range(len(data_list)):
            _first_pr = data_list[i][1]

            for j in range(i + 1, len(data_list)):
                task_differ = {}
                _second_pr = data_list[j][1]

                for fp_task in _first_pr:
                    sp_task = self._find_same_task(
                        fp_task,
                        second_pr=_second_pr,
                        fp_project_id=data_list[i][0],
                        sec_project_id=data_list[j][0],
                    )
                    if sp_task is None:
                        continue

                    _dif = self._comparing_tasks(fp_task, sp_task, only_amount)
                    task_differ[sp_task["data"]["text"]] = _dif

                differ[f"{data_list[i][0]}-{data_list[j][0]}"] = task_differ

        return differ

    @staticmethod
    def _sort_task(task: dict):
        return sorted(
            task["annotations"][0]["result"],
            key=lambda x: x["value"]["start"],
        )

    @classmethod
    def _comparing_tasks(
        cls, fp_task: dict, sp_task: dict, only_amount: bool = False
    ) -> dict:
        """
        Compares two tasks and finds the differences.

        This method compares the annotations of two tasks. It checks for differences in labels and coordinates.
        Differences are categorized into label differences, coordinate differences, and full matches.

        If `only_amount` is set to True, the method only counts the number of differences and returns these counts.
        Otherwise, it returns the actual differences.

        Args:
            fp_task : dict
                The first task to compare. It is a dictionary representing a task,
                which includes information such as the annotations associated with the task.
            sp_task : dict
                The second task to compare. It is also a dictionary representing a task.
            only_amount : bool, optional
                If True, only the amount of differences is returned. If False, the actual differences are returned.
                Defaults to False.
        Returns:
            dict: The differences between the tasks. The dictionary contains keys 'differ_label', 'differ_coords',
                and 'full_match', each associated with a list of differences in that category. If `only_amount` is True,
                the dictionary contains the counts of differences instead of the actual differences.
        """

        _dif = {
            "differ_label": [],
            "differ_coords": [],
            "full_match": [],
            "differ_label_count": 0,
            "differ_coords_count": 0,
            "full_match_count": 0,
        }

        fp_sorted_annotations = cls._sort_task(fp_task)
        sp_sorted_annotations = cls._sort_task(sp_task)

        for fp_annotation in fp_sorted_annotations:
            value_res_1 = fp_annotation["value"]
            crossing = False

            for i, sp_annotation in enumerate(sp_sorted_annotations):
                value_res_2 = sp_annotation["value"]

                if (
                    value_res_1["start"] == value_res_2["start"]
                    and value_res_1["end"] == value_res_2["end"]
                ):
                    if value_res_1["labels"][0] != value_res_2["labels"][0]:
                        _dif["differ_label"].append((value_res_1, value_res_2))
                    else:
                        _dif["full_match"].append(value_res_1)
                    crossing = True
                    continue

                elif (
                    max(value_res_1["start"], value_res_2["start"])
                    < (min(value_res_1["end"], value_res_2["end"]))
                ) and (
                    value_res_1["start"] != value_res_2["start"]
                    or value_res_1["end"] != value_res_2["end"]
                ):
                    _dif["differ_coords"].append((value_res_1, value_res_2))
                    crossing = True
                    continue

                elif (i == (len(sp_sorted_annotations) - 1)) or (
                    value_res_1["end"] <= value_res_2["start"]
                ):
                    if not crossing:
                        _dif["differ_coords"].append((value_res_1, None))
                    break

        for sp_annotation in sp_sorted_annotations:
            value_res_2 = sp_annotation["value"]
            for i, fp_annotation in enumerate(fp_sorted_annotations):
                value_res_1 = fp_annotation["value"]

                if max(value_res_1["start"], value_res_2["start"]) < min(
                    value_res_1["end"], value_res_2["end"]
                ):
                    break

                elif (i == (len(fp_sorted_annotations) - 1)) or (
                    value_res_2["end"] <= value_res_1["start"]
                ):
                    _dif["differ_coords"].append((None, value_res_2))
                    break
        amount = {
            "differ_label_count": len(_dif["differ_label"]),
            "differ_coords_count": len(_dif["differ_coords"]),
            "full_match_count": len(_dif["full_match"]),
        }

        if only_amount:
            return amount

        _dif.update(amount)
        return _dif

    @staticmethod
    def _find_relation_start_index(task):
        for index, ann in enumerate(task["annotations"][0]["result"]):
            if ann["type"] == "relation":
                return index

        return None

    @staticmethod
    def _update_relation_information(project: list):
        """
        Updates the relation information in each task of a project.

        This method iterates over the tasks in a project. For each relation in a task,
        it changes the direction to 'right' if it's 'left', sorts the labels, and adds the text span to
        the relation dictionary.

        The method operates in-place, modifying the original tasks.

        Args:
            project : list
                The project data. It's a list of tasks, where each task is a dictionary that contains
                various fields, including "relations".

        Returns:
            None. The tasks in the project are modified in-place with updated relation information.
        """
        for task in project:
            for rel in task["relations"]:
                # need to be brought to one direction to simplify comparison
                if rel["direction"] == "left":
                    rel["from_id"], rel["to_id"] = rel["to_id"], rel["from_id"]
                    rel["direction"] = "right"
                # sort labels for future comparing
                rel["labels"].sort()

                rel["to_span"], rel["from_span"] = None, None

                # add text span to relation dict
                for ann in task["annotations"][0]["result"]:
                    if ann["id"] == rel["from_id"]:
                        rel["from_span"] = ann["value"]["text"]

                    elif ann["id"] == rel["to_id"]:
                        rel["to_span"] = ann["value"]["text"]

                    elif rel["to_span"] and rel["from_span"]:
                        break

    @classmethod
    def _merge_relations_process(cls, projects_data: dict) -> list[dict]:
        """
        Merges the tasks from different projects.

        This method iterates over the tasks in each pair of projects. For each task in the first project,
        it finds the same task in the second project. If a similar task is found, it makes their annotation
        IDs identical, removes the similar relations, and adjusts the labels in the relations based on the
        differences between the tasks.

        Args:
            projects_data : dict
                The projects data. Each key is a project ID and each value is a list of tasks for that project.

        Returns:
            list[dict]: The list of tasks from the first project after the merging process.
        """
        data_list = list(projects_data.items())

        for i in range(len(data_list)):
            _first_pr = data_list[i][1]

            for j in range(i + 1, len(data_list)):
                _second_pr = data_list[j][1]

                for fp_task in _first_pr:
                    sp_task = cls._find_same_task(
                        fp_task,
                        second_pr=_second_pr,
                        fp_project_id=data_list[i][0],
                        sec_project_id=data_list[j][0],
                    )
                    if sp_task is None:
                        continue

                    cls._make_identical_indexes_for_two_project(
                        fp_task=fp_task, sp_task=sp_task
                    )
                    cls._leave_unique_relations(fp_task=fp_task, sp_task=sp_task)
                    cls._making_special_labels_for_relations(
                        fp_task=fp_task,
                        sp_task=sp_task,
                        fp_project_id=data_list[i][0],
                        sp_project_id=data_list[j][0],
                    )

        # return only first project
        return data_list[0][1]

    @staticmethod
    def _make_identical_indexes_for_two_project(fp_task: dict, sp_task: dict):
        """
        Makes the annotation IDs identical for two tasks from different projects.

        This method creates a dictionary that maps the start and end values of each annotation in the
        first task to its ID. Then, it iterates over the annotations in the second task and changes
        their IDs to match the corresponding IDs in the first task. It also changes the to_id and
        from_id in the relations of the second task to match the IDs in the first task.

        The method operates in-place, modifying the original tasks.

        Args:
            fp_task : dict
                The task from the first project. Each task is a dictionary that contains various fields,
                including "annotations".

            sp_task : dict
                The task from the second project. Each task is a dictionary that contains various fields,
                including "annotations".

        Returns:
            None. The fp_task and sp_task dictionaries are modified in-place with matching annotation IDs.
        """
        matching_id_dict = {
            (ann["value"]["start"], ann["value"]["end"]): ann["id"]
            for ann in fp_task["annotations"][0]["result"]
            # if ann["type"] == "labels"
        }
        ann_id_to_rel_id = {}
        for ann in sp_task["annotations"][0]["result"]:
            # if ann["type"] == "labels":
            key = (ann["value"]["start"], ann["value"]["end"])
            ann_id_to_rel_id[ann["id"]] = matching_id_dict[key]
            ann["id"] = matching_id_dict[key]
        for rel in sp_task["relations"]:
            # elif ann["type"] == "relation":
            rel["to_id"] = ann_id_to_rel_id[rel["to_id"]]
            rel["from_id"] = ann_id_to_rel_id[rel["from_id"]]

    def _split_anns_and_rels(self, project: list[dict]):
        for task in project:
            if index := self._find_relation_start_index(task):
                task["relations"] = task["annotations"][0]["result"][index:]
                task["annotations"][0]["result"] = task["annotations"][0]["result"][
                    :index
                ]
            else:
                task["relations"] = []

    def _count_annotations_per_label(self, projects_data: dict) -> dict:
        """
            Counts the annotations per label for each project.

        This method iterates over the projects data and for each project, it counts the number of annotations
        for each label. The counts are stored in a dictionary where each key is a project ID and each value is
        another dictionary. In this inner dictionary, each key is a task text and each value is a dictionary that
        maps a label to its count in the task.
        Args:
            projects_data : dict
                The projects data. Each key is a project ID and each value is a list of tasks for that project.

        Returns:
            dict: A dictionary where each key is a project ID and each value is another dictionary. In this inner
            dictionary, each key is a task text and each value is a dictionary that maps a label to its count in the task.
        """

        amount_anns_per_project = {}
        for pr_id, pr_data in projects_data.items():
            amount_anns_per_task = {}
            for task in pr_data:
                entity_counter = {label: 0 for label in self.labels}

                # Temporary fix
                # if there are no annotations without checking there will be an error, if there are no annotations.
                if len(task["annotations"]):
                    annotations = task["annotations"][0]["result"]
                    for item in annotations:
                        try:
                            entity_counter[item["value"]["labels"][0]] += 1
                        except KeyError:
                            logger.error(
                                f'There is not entity type "{item["value"]["labels"][0]}"',
                                exc_info=True,
                            )
                    entity_counter["total"] = len(annotations)
                    amount_anns_per_task[task["data"]["text"]] = entity_counter
            amount_anns_per_project[pr_id] = amount_anns_per_task

        return amount_anns_per_project

    async def _create_new_project(self, label_config: str, title: str):
        """
        Create new project to Label Studio
        Args:
            label_config: str
                config for marking up the annotations
            title: str
                new project name

        """
        data = {"label_config": label_config, "title": title}

        async with aiohttp.ClientSession() as session:
            url = f"{self.url}/api/projects/"
            async with session.post(
                url=url,
                headers=self.headers,
                json=data,
            ) as resp:
                if resp.ok:
                    data = json.loads((await resp.read()).decode("utf-8"))
                    return data["id"]
                else:
                    logger.error(
                        f"error during creating new project. Response status {resp.status}",
                        exc_info=True,
                    )
                    return None

    async def upload_data(self, new_project_id: int, data: list[dict]):
        """
        Push processed data to new project
        Args:
            new_project_id: int
                Label studio project ID
            data: dict
                processed data
        """
        async with aiohttp.ClientSession() as session:
            url = f"{self.url}/api/projects/{new_project_id}/import"
            async with session.post(url, headers=self.headers, json=data) as resp:
                if resp.ok:
                    logger.info(f"project {new_project_id}, data uploaded")

    async def fetch_data(self, pr_id: int) -> list:
        """
        Get project data from Label Studio
        Args:
            pr_id: int
                Label studio project ID

        Returns:
            list: Label studio data
        """
        async with aiohttp.ClientSession() as session:
            url = f"{self.url}/api/projects/{pr_id}/tasks/?page_size=-1"
            async with session.get(url, headers=self.headers) as resp:
                return json.loads(await resp.read())[::-1]

    async def get_meta_project_info(self, pr_id: int) -> dict:
        """
        Get project meta information from Label Studio
        Args:
            pr_id: int
                Label studio project ID

        Returns:
            dict: meta info
        """
        async with aiohttp.ClientSession() as session:
            url = f"{self.url}/api/projects/{pr_id}"
            async with session.get(url=url, headers=self.headers) as resp:
                return json.loads(await resp.read())

    @staticmethod
    def _find_same_task(
        fp_task: dict, second_pr: dict, fp_project_id: int, sec_project_id: int
    ) -> Optional[dict]:
        """
        Finds the same task in two different projects.

        This method compares a given task from the first project with all tasks in the second project to find a match.
        The match is based on the text of the task. If a matching task is found, it is returned. If not, None is returned.

        If the given task or the matching task does not have any annotations, an error is logged.

        Args:
            fp_task : dict
                The task from the first project to find a match for. It is a dictionary representing a task, which
                includes information such as the text of the task and its annotations.
            second_pr : dict
                The second project to find the matching task in. It is a dictionary representing a project, which
                includes information such as the tasks of the project.
            fp_project_id : int
                The ID of the first project.
            sec_project_id : int
                The ID of the second project.

        Returns:
            dict or None: The matching task from the second project, if found. If not found, None is returned.
        """
        text = fp_task["data"]["text"]
        if not len(fp_task["annotations"]):
            logger.error(
                f"There is no annotations in {text} either in project "
                f"{fp_project_id}"
            )
            return None

        sp_task = find_first(lambda ts: ts["data"]["text"] == text, second_pr)

        if sp_task is None:
            logger.error(f"There is no sentence {text} in project {sec_project_id}")

        elif not len(sp_task["annotations"]):
            logger.error(
                f"There is no annotations in {text} either in project "
                f"{sec_project_id}"
            )
            return None
        return sp_task

    @classmethod
    def _leave_unique_relations(cls, fp_task: dict, sp_task: dict):
        """
        Removes the similar relations from the tasks of two different projects.

        This method iterates over the relations in the tasks from two different projects.
        For each relation in the first task, it checks if a similar relation exists in the second task.
        If a similar relation exists, it removes the relation from the first task.
        It does the same for the relations in the second task.

        The method operates in-place, modifying the original tasks.

        Args:
            fp_task : dict
                The task from the first project. Each task is a dictionary that contains various fields,
                including "relations".

            sp_task : dict
                The task from the second project. Each task is a dictionary that contains various fields,
                including "relations".

        Returns:
            None. The fp_task and sp_task dictionaries are modified in-place with only unique "relations".
        """
        # drop all of the simular relations in projects
        if (fp_relations := fp_task.get("relations")) and (
            sp_relations := sp_task.get("relations")
        ):
            unique_fp_rels = [
                rel
                for rel in fp_relations
                if not cls._check_simular_relations(rel, sp_relations)
            ]
            unique_sp_rels = [
                rel
                for rel in sp_relations
                if not cls._check_simular_relations(rel, fp_relations)
            ]

            fp_task["relations"] = unique_fp_rels
            sp_task["relations"] = unique_sp_rels

    @staticmethod
    def _check_simular_relations(fp_rel: dict, sec_project_relations: list) -> bool:
        """
        Checks if a relation from the first project exists in the second project.

        This method iterates over the relations in the second project and compares each one to the relation
        from the first project.
        It checks if the labels, from_span, to_span, and direction of the relations are the same.

        Args:
            fp_rel : dict
                The relation from the first project. It's a dictionary that contains fields like "labels",
                "from_span", "to_span", and "direction".

            sec_project_relations : list
                The list of relations from the second project. Each relation is a dictionary similar to fp_rel.

        Returns:
            bool: True if the relation from the first project exists in the second project, False otherwise.
        """
        for sp_rel in sec_project_relations:
            if (
                fp_rel["labels"] == sp_rel["labels"]
                and fp_rel["from_span"] == sp_rel["from_span"]
                and fp_rel["to_span"] == sp_rel["to_span"]
                and fp_rel["direction"] == sp_rel["direction"]
            ):
                return True

        return False

    @staticmethod
    def _making_special_labels_for_relations(
        fp_task, sp_task, fp_project_id: int, sp_project_id: int
    ):
        """
        Modifies the relations in the tasks by comparing and adjusting labels.

        This method iterates over the relations in the tasks from two different projects. For each pair
        of relations, it checks if they are the same. If they are, it adjusts the labels based on the
        differences in the relations. If a relation is only present in one of the tasks, it adds a special
        label to it.

        The method operates in-place, modifying the original tasks.

        Args:
            fp_task : dict
                The task from the first project. Each task is a dictionary that contains various fields,
                including "relations".

        sp_task : dict
            The task from the second project. Each task is a dictionary that contains various fields,
            including "relations".

        fp_project_id : int
            The ID of the first project.

        sp_project_id : int
            The ID of the second project.

        Returns:
            None. The fp_task and sp_task dictionaries are modified in-place with adjusted "labels" in "relations".
        """

        # TODO оптимизировать скрипт и доку
        proceed_relations = []
        fp_relations = fp_task.get("relations")
        sp_relations = sp_task.get("relations")

        for fp_rel in fp_relations:
            for sp_index, sp_rel in enumerate(sp_relations):
                # find two corresponding relations
                if (
                    fp_rel["to_span"] == sp_rel["to_span"]
                    and fp_rel["from_span"] == sp_rel["from_span"]
                ) or (
                    fp_rel["to_span"] == sp_rel["from_span"]
                    and fp_rel["from_span"] == sp_rel["to_span"]
                ):
                    # check relations labels
                    if fp_rel["labels"] != sp_rel["labels"]:
                        fp_rel["labels"] = [
                            "dif_labels",
                            *fp_rel["labels"],
                            *sp_rel["labels"],
                        ]
                    # check relation direction. True: if object and subject are confused,
                    # if relation directions don't match
                    if (
                        fp_rel["direction"] != sp_rel["direction"]
                        or fp_rel["to_span"] == sp_rel["from_span"]
                    ):
                        fp_rel["labels"].append("dif_direction")

                    # IMPORTANT!!! drop possible simular labels
                    fp_rel["labels"] = list(set(fp_rel["labels"]))
                    #  IMPORTANT!!! drop processed relation from second task
                    del sp_relations[sp_index]

                else:
                    # process no matched relation from the first task
                    fp_rel["labels"].extend(("only_one_project", str(fp_project_id)))

                proceed_relations.append(fp_rel)
                break
        # process all the no matched relations from the second task
        for left_sp_rel in sp_relations:
            left_sp_rel["labels"].extend(("only_one_project", str(sp_project_id)))
            proceed_relations.append(left_sp_rel)

        # drop auxiliary keys
        del fp_task["relations"], sp_task["relations"]
        fp_task["annotations"][0]["result"].extend(proceed_relations)

    @classmethod
    def drop_simular_annotations(cls, projects_data: dict) -> dict:
        """
        Removes similar tasks across different projects.

        This method iterates over the projects data and compares each pair of projects. For each pair, it finds
        the tasks that are the same in both projects. Then, it removes the annotations in the first task that are
        also present in the second task, and vice versa.

        The method operates in-place, modifying the original projects data.

        Args:
            projects_data : dict
                The projects data. Each key is a project ID and each value is a list of tasks for that project.

        Returns:
            dict: The modified projects data with similar tasks removed.
        """
        data_list = list(projects_data.items())

        for i in range(len(data_list)):
            _first_pr = data_list[i][1]

            for j in range(i + 1, len(data_list)):
                _second_pr = data_list[j][1]

                for fp_task in _first_pr:
                    sp_task = cls._find_same_task(
                        fp_task,
                        second_pr=_second_pr,
                        fp_project_id=data_list[i][0],
                        sec_project_id=data_list[j][0],
                    )
                    if sp_task is None:
                        continue

                    cls._leave_unique_annotations(fp_task, sp_task)
        return projects_data

    @classmethod
    def _leave_unique_annotations(cls, fp_task: dict, sp_task: dict):
        # drop all of the simular annotations in projects
        fp_task_sorted = cls._sort_task(fp_task)
        sp_sorted_sorted = cls._sort_task(sp_task)

        fp_task["annotations"][0]["result"] = [
            ann
            for ann in fp_task_sorted
            if ann["value"] not in [ann_x["value"] for ann_x in sp_sorted_sorted]
        ]
        sp_task["annotations"][0]["result"] = [
            ann
            for ann in sp_sorted_sorted
            if ann["value"] not in [ann_x["value"] for ann_x in fp_task_sorted]
        ]

    def make_default_views_layout(
        self, projects: list[Union[str, int]], relation_labels: Iterable[str] = None
    ) -> str:
        """
        Creates a default views layout for the given projects in Label Studio format.

        This method generates a string that represents the layout of labels in Label Studio for each project.
        Each label is assigned a unique color.

        Args:
            projects : list[int]
                The list of project IDs for which to create the views layout.
            relation_labels: list[str]
                The list of relation labels, if they exist
        Returns:
            str: The default views layout as a string in Label Studio format.
        """
        default_colors = [
            "red",
            "#18d0ef",
            "#4e72f2",
            "green",
            "#6a2d7e",
            "#FFA39E",
            "#c3bf04",
            "#FFA39E",
            "#c320b3",
            "#FFA39E",
        ]
        relation_template = ""
        if relation_labels:
            relations = "\n".join(
                f'    <Relation value="{rel_label}"/>' for rel_label in relation_labels
            )
            relation_template = f"  <Relations>\n{relations}\n  </Relations>"
            labels = "\n".join(
                f'    <Label value="{label}" background="{color}"/>'
                for label, color in zip(self.labels, default_colors)
            )

        elif len(projects) == 1:
            labels = "\n".join(
                f'    <Label value="{label}" background="{color}"/>'
                for label, color in zip(self.labels, default_colors)
            )

        else:
            color_gen = ColorGenerator()
            labels = "\n".join(
                f'    <Label value="{label}-{pr_id}" background="{color_gen.get_color_for_type(label)}"/>'
                for pr_id in projects
                for label in self.labels
            )

        labels_template = (
            f"  <Labels name='label' toName='text'>\n{labels}\n  </Labels>"
        )

        template = (
            f'<View style="line-height: 3;">\n{relation_template}\n{labels_template}\n'
            '<Text name="text" value="$text"/>\n</View>\n'
        )
        return template

    @staticmethod
    def get_random_color():
        return "#%06x" % random.randint(0, 0xFFFFFF)

    @staticmethod
    def _merging_projects(projects_data: dict) -> list:
        """
        Merges multiple projects into one.

        This method iterates over the projects data and combines their tasks into a single list. If a task from one
        project has the same text as a task from another project, the annotations of the latter are added to the former.

        Args:
            projects_data : dict
                The projects data. Each key is a project ID and each value is a list of tasks for that project.

        Returns:
            dict: The merged projects data. Each key is a task text and each value is a dictionary representing a task,
            which includes the combined annotations from all projects.
        """
        result = []
        _task_pool = {}
        for pr_id, pr_data in projects_data.items():
            for task in pr_data:
                for annotation in task["annotations"]:
                    for label in annotation["result"]:
                        label["value"]["labels"][0] += f"-{pr_id}"

                    if (_text := task["data"]["text"]) in _task_pool:
                        _index = _task_pool[_text]
                        result[_index]["annotations"][0]["result"].extend(
                            annotation["result"]
                        )

                    else:
                        _task_pool[_text] = len(result)
                        result.append(task)
                    break  # work is allowed only if there is one annotation in the task
        return result

    async def _update_annotations(self, annotation: dict):
        """
        Put particular changes into annotation
        Args:
            annotation: dict
                new Label studio annotation data
        """
        async with aiohttp.ClientSession() as session:
            url = f"{self.url}/api/annotations/{annotation['id']}/"
            async with session.patch(
                url, headers=self.headers, json={"result": annotation["result"]}
            ) as resp:
                if resp.ok:
                    logger.info("tutututututututtu)))))")
                else:
                    logger.error(await resp.read(), exc_info=True)

    @staticmethod
    def _remove_outer_spaces(annotation: dict) -> bool:
        need_update = False
        for label in annotation["result"]:
            if label["type"] == "relation":
                continue
            ann_value = label["value"]
            if (trimmed_text := ann_value["text"].strip()) != ann_value["text"]:
                ann_value["start"] = ann_value["start"] + (
                    len(ann_value["text"]) - len(ann_value["text"].lstrip())
                )
                ann_value["end"] = ann_value["start"] + len(trimmed_text)
                ann_value["text"] = trimmed_text
                need_update = True
        return need_update

    async def drop_space_in_start_and_end(
        self, project: list[dict], with_update: bool = False
    ):
        try:
            for task in project:
                for annotation in task["annotations"]:
                    need_update = self._remove_outer_spaces(annotation)
                    if with_update and need_update:
                        await self._update_annotations(annotation)
            return True
        except:
            logger.error("There was error during removing spaces.", exc_info=True)
            return False

    async def create_project_with_data(
        self, data: list, relation_labels: Iterable[str] = None, title: str = "title"
    ):
        label_config = self.make_default_views_layout(
            [
                "hug",
            ],
            relation_labels=relation_labels,
        )

        new_project_id = await self._create_new_project(
            label_config=label_config, title=title if title else "title"
        )

        await self.upload_data(new_project_id, data)
        return new_project_id

    async def make_sentences_as_tasks_with_possible_relations(
        self, project_data, entity_types: set[str] = None
    ):
        self._split_anns_and_rels(project_data)

        anns_per_sentenced_filtered = []
        for task in project_data:
            split_data = self._split_annotations_per_sentence(
                text=task["data"]["text"], annotations=task["annotations"][0]["result"]
            )
            anns_per_sentenced_filtered.extend(
                [
                    filtered_data
                    for sentence_data in split_data
                    if (
                        filtered_data := self._filter_by_entity_type(
                            sentence_data=sentence_data, entity_types=entity_types
                        )
                    )
                    is not None
                ]
            )

        return self._make_upload_data_after_split_annotations_per_sentence(
            anns_per_sentenced_filtered
        )

    @staticmethod
    def _filter_by_entity_type(
        sentence_data,
        entity_types: list[str] = None,
        entity_count_per_sentence: int = 2,
    ):
        if not entity_types:
            return sentence_data
        filtered_annotations = list(
            filter(
                lambda x: x["value"]["labels"][0] in entity_types,
                sentence_data["annotations"],
            )
        )

        if len(filtered_annotations) < entity_count_per_sentence:
            return None

        sentence_data["annotations"] = filtered_annotations
        return sentence_data

    @staticmethod
    def _make_upload_data_after_split_annotations_per_sentence(
        anns_per_sentenced_filtered,
    ):
        return [
            {
                "annotations": [{"result": sent["annotations"]}],
                "data": {"text": sent["sentence"]},
            }
            for sent in anns_per_sentenced_filtered
        ]

    @staticmethod
    def split_text_into_sentences(text: str) -> list[str]:
        """
        Splits the text into sentences.

        Args:
            text : str
                The text to be split into sentences.

        Returns:
            list[str] : A list of sentences.
        """
        return sent_tokenize(text)

    @classmethod
    def _split_annotations_per_sentence(cls, text, annotations):
        """
        Splits the text into sentences and adjusts the annotation indices accordingly.

        Args:
            text (str): The text to be split.
            annotations (list[dict]): The annotations with their original indices.

        Returns:
        list[dict]: The split sentences with their corresponding annotations.
        """
        sentences = cls.split_text_into_sentences(text=text)
        processing_data = []
        for sentence in sentences:
            sentence_annotations = []
            sentence_start = text.find(sentence)
            sentence_end = sentence_start + len(sentence)
            for annotation in annotations:
                if sentence_start <= annotation["value"]["start"] < sentence_end:
                    new_annotation = annotation.copy()
                    # adjusting indexes taking into account the accumulated length of sentences
                    new_annotation["value"]["start"] -= sentence_start
                    new_annotation["value"]["end"] -= sentence_start
                    sentence_annotations.append(new_annotation)
            processing_data.append(
                {"sentence": sentence, "annotations": sentence_annotations}
            )
        return processing_data


class ColorGenerator:
    """
    A class used to generate colors for different entity types.
    """

    def __init__(self):
        self.colors = {}

    def get_color_for_type(self, entity_type: str) -> str:
        """
        Retrieves or assigns a color for a specific entity type.

        This method checks if the entity type already has an assigned color in the `colors` attribute. If it does,
        it makes the color darker and returns the updated color. If the entity type does not have an assigned color,
        it generates a new random bright color, assigns it to the entity type, and returns the new color.

        Args:
            entity_type : str
                The entity type to get or assign a color for.

        Returns:
            str: The hex representation of the color assigned to the entity type.
        """
        if entity_type not in self.colors:
            self.colors[entity_type] = self.get_random_bright_color()
        else:
            self.colors[entity_type] = self.make_color_darker(
                self.colors[entity_type], 60
            )
        return self.rgb_to_hex(self.colors[entity_type])

    @staticmethod
    def get_random_bright_color():
        return (
            random.randint(100, 255),
            random.randint(100, 255),
            random.randint(100, 255),
        )

    @staticmethod
    def make_color_darker(color: tuple, amount: int):
        return tuple(max(0, c - amount) for c in color)

    @staticmethod
    def rgb_to_hex(color):
        return "#{:02x}{:02x}{:02x}".format(*color)

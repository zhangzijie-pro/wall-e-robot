\# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).

\# All rights reserved.

\#

\# SPDX-License-Identifier: BSD-3-Clause



import glob

import os

import shutil

import subprocess

import sys

from datetime import datetime



import jinja2

from common import MULTI\_AGENT\_ALGORITHMS, ROOT\_DIR, SINGLE\_AGENT\_ALGORITHMS, TASKS\_DIR, TEMPLATE\_DIR



jinja\_env = jinja2.Environment(

&nbsp;   loader=jinja2.FileSystemLoader(TEMPLATE\_DIR),

&nbsp;   trim\_blocks=True,

&nbsp;   lstrip\_blocks=True,

)





def \_setup\_git\_repo(project\_dir: str) -> None:

&nbsp;   """Setup the git repository.



&nbsp;   Args:

&nbsp;       project\_dir: The directory of the project.

&nbsp;   """

&nbsp;   commands = \[

&nbsp;       \["git", "init"],

&nbsp;       \["git", "add", "-f", "."],

&nbsp;       \["git", "commit", "-q", "-m", "Initial commit"],

&nbsp;   ]

&nbsp;   for command in commands:

&nbsp;       result = subprocess.run(command, capture\_output=True, text=True, cwd=project\_dir)

&nbsp;       for line in result.stdout.splitlines():

&nbsp;           print(f"  |  {line}")





def \_replace\_in\_file(replacements: list\[tuple\[str, str]], src: str, dst: str | None = None) -> None:

&nbsp;   """Replace the placeholders in the file.



&nbsp;   Args:

&nbsp;       replacements: The replacements to make.

&nbsp;       src: The source file.

&nbsp;       dst: The destination file. If not provided, the source file will be overwritten.

&nbsp;   """

&nbsp;   with open(src) as file:

&nbsp;       content = file.read()

&nbsp;   for old, new in replacements:

&nbsp;       content = content.replace(old, new)

&nbsp;   with open(src if dst is None else dst, "w") as file:

&nbsp;       file.write(content)





def \_write\_file(dst: str, content: str) -> None:

&nbsp;   """Write the content to a file.



&nbsp;   Args:

&nbsp;       dst: The path to the file.

&nbsp;       content: The content to write to the file.

&nbsp;   """

&nbsp;   with open(dst, "w") as file:

&nbsp;       file.write(content)





def \_generate\_task\_per\_workflow(task\_dir: str, specification: dict) -> None:

&nbsp;   """Generate the task files for a single workflow.



&nbsp;   Args:

&nbsp;       task\_dir: The directory where the task files will be generated.

&nbsp;       specification: The specification of the project/task.

&nbsp;   """

&nbsp;   task\_spec = specification\["task"]

&nbsp;   agents\_dir = os.path.join(task\_dir, "agents")

&nbsp;   os.makedirs(agents\_dir, exist\_ok=True)

&nbsp;   # common content

&nbsp;   # - task/\_\_init\_\_.py

&nbsp;   template = jinja\_env.get\_template("tasks/\_\_init\_\_task")

&nbsp;   \_write\_file(os.path.join(task\_dir, "\_\_init\_\_.py"), content=template.render(\*\*specification))

&nbsp;   # - task/agents/\_\_init\_\_.py

&nbsp;   template = jinja\_env.get\_template("tasks/\_\_init\_\_agents")

&nbsp;   \_write\_file(os.path.join(agents\_dir, "\_\_init\_\_.py"), content=template.render(\*\*specification))

&nbsp;   # - task/agents/\*cfg\*

&nbsp;   for rl\_library in specification\["rl\_libraries"]:

&nbsp;       rl\_library\_name = rl\_library\["name"]

&nbsp;       for algorithm in rl\_library.get("algorithms", \[]):

&nbsp;           file\_name = f"{rl\_library\_name}\_{algorithm.lower()}\_cfg"

&nbsp;           file\_ext = ".py" if rl\_library\_name == "rsl\_rl" else ".yaml"

&nbsp;           try:

&nbsp;               template = jinja\_env.get\_template(f"agents/{file\_name}")

&nbsp;           except jinja2.exceptions.TemplateNotFound:

&nbsp;               print(f"Template not found: agents/{file\_name}")

&nbsp;               continue

&nbsp;           \_write\_file(os.path.join(agents\_dir, file\_name + file\_ext), content=template.render(\*\*specification))

&nbsp;   # workflow-specific content

&nbsp;   if task\_spec\["workflow"]\["name"] == "direct":

&nbsp;       # - task/\*env\_cfg.py

&nbsp;       template = jinja\_env.get\_template(f'tasks/direct\_{task\_spec\["workflow"]\["type"]}/env\_cfg')

&nbsp;       \_write\_file(

&nbsp;           os.path.join(task\_dir, f'{task\_spec\["filename"]}\_env\_cfg.py'), content=template.render(\*\*specification)

&nbsp;       )

&nbsp;       # - task/\*env.py

&nbsp;       template = jinja\_env.get\_template(f'tasks/direct\_{task\_spec\["workflow"]\["type"]}/env')

&nbsp;       \_write\_file(os.path.join(task\_dir, f'{task\_spec\["filename"]}\_env.py'), content=template.render(\*\*specification))

&nbsp;   elif task\_spec\["workflow"]\["name"] == "manager-based":

&nbsp;       # - task/\*env\_cfg.py

&nbsp;       template = jinja\_env.get\_template(f'tasks/manager-based\_{task\_spec\["workflow"]\["type"]}/env\_cfg')

&nbsp;       \_write\_file(

&nbsp;           os.path.join(task\_dir, f'{task\_spec\["filename"]}\_env\_cfg.py'), content=template.render(\*\*specification)

&nbsp;       )

&nbsp;       # - task/mdp folder

&nbsp;       shutil.copytree(

&nbsp;           os.path.join(TEMPLATE\_DIR, "tasks", f'manager-based\_{task\_spec\["workflow"]\["type"]}', "mdp"),

&nbsp;           os.path.join(task\_dir, "mdp"),

&nbsp;           dirs\_exist\_ok=True,

&nbsp;       )





def \_generate\_tasks(specification: dict, task\_dir: str) -> list\[dict]:

&nbsp;   """Generate the task files for an external project or an internal task.



&nbsp;   Args:

&nbsp;       specification: The specification of the project/task.

&nbsp;       task\_dir: The directory where the tasks will be generated.



&nbsp;   Returns:

&nbsp;       A list of specifications for the tasks.

&nbsp;   """

&nbsp;   specifications = \[]

&nbsp;   task\_name\_prefix = "Template" if specification\["external"] else "Isaac"

&nbsp;   general\_task\_name = "-".join(\[item.capitalize() for item in specification\["name"].split("\_")])

&nbsp;   for workflow in specification\["workflows"]:

&nbsp;       task\_name = general\_task\_name + ("-Marl" if workflow\["type"] == "multi-agent" else "")

&nbsp;       filename = task\_name.replace("-", "\_").lower()

&nbsp;       task = {

&nbsp;           "workflow": workflow,

&nbsp;           "filename": filename,

&nbsp;           "classname": task\_name.replace("-", ""),

&nbsp;           "dir": os.path.join(task\_dir, workflow\["name"].replace("-", "\_"), filename),

&nbsp;       }

&nbsp;       if task\["workflow"]\["name"] == "direct":

&nbsp;           task\["id"] = f"{task\_name\_prefix}-{task\_name}-Direct-v0"

&nbsp;       elif task\["workflow"]\["name"] == "manager-based":

&nbsp;           task\["id"] = f"{task\_name\_prefix}-{task\_name}-v0"

&nbsp;       print(f"  |    |-- Generating '{task\['id']}' task...")

&nbsp;       \_generate\_task\_per\_workflow(task\["dir"], {\*\*specification, "task": task})

&nbsp;       specifications.append({\*\*specification, "task": task})

&nbsp;   return specifications





def \_external(specification: dict) -> None:

&nbsp;   """Generate an external project.



&nbsp;   Args:

&nbsp;       specification: The specification of the project/task.

&nbsp;   """

&nbsp;   name = specification\["name"]

&nbsp;   project\_dir = os.path.join(specification\["path"], name)

&nbsp;   os.makedirs(project\_dir, exist\_ok=True)

&nbsp;   # repo files

&nbsp;   print("  |-- Copying repo files...")

&nbsp;   shutil.copyfile(os.path.join(ROOT\_DIR, ".dockerignore"), os.path.join(project\_dir, ".dockerignore"))

&nbsp;   shutil.copyfile(os.path.join(ROOT\_DIR, ".flake8"), os.path.join(project\_dir, ".flake8"))

&nbsp;   shutil.copyfile(os.path.join(ROOT\_DIR, ".gitattributes"), os.path.join(project\_dir, ".gitattributes"))

&nbsp;   if os.path.exists(os.path.join(ROOT\_DIR, ".gitignore")):

&nbsp;       shutil.copyfile(os.path.join(ROOT\_DIR, ".gitignore"), os.path.join(project\_dir, ".gitignore"))

&nbsp;   shutil.copyfile(

&nbsp;       os.path.join(ROOT\_DIR, ".pre-commit-config.yaml"), os.path.join(project\_dir, ".pre-commit-config.yaml")

&nbsp;   )

&nbsp;   template = jinja\_env.get\_template("external/README.md")

&nbsp;   \_write\_file(os.path.join(project\_dir, "README.md"), content=template.render(\*\*specification))

&nbsp;   # scripts

&nbsp;   print("  |-- Copying scripts...")

&nbsp;   # reinforcement learning libraries

&nbsp;   dir = os.path.join(project\_dir, "scripts")

&nbsp;   os.makedirs(dir, exist\_ok=True)

&nbsp;   for rl\_library in specification\["rl\_libraries"]:

&nbsp;       shutil.copytree(

&nbsp;           os.path.join(ROOT\_DIR, "scripts", "reinforcement\_learning", rl\_library\["name"]),

&nbsp;           os.path.join(dir, rl\_library\["name"]),

&nbsp;           dirs\_exist\_ok=True,

&nbsp;       )

&nbsp;       # replace placeholder in scripts

&nbsp;       for file in glob.glob(os.path.join(dir, rl\_library\["name"], "\*.py")):

&nbsp;           \_replace\_in\_file(

&nbsp;               \[(

&nbsp;                   "# PLACEHOLDER: Extension template (do not remove this comment)",

&nbsp;                   f"import {name}.tasks  # noqa: F401",

&nbsp;               )],

&nbsp;               src=file,

&nbsp;           )

&nbsp;   # - other scripts

&nbsp;   \_replace\_in\_file(

&nbsp;       \[("import isaaclab\_tasks", f"import {name}.tasks"), ("isaaclab\_tasks", name), ('"Isaac"', '"Template-"')],

&nbsp;       src=os.path.join(ROOT\_DIR, "scripts", "environments", "list\_envs.py"),

&nbsp;       dst=os.path.join(dir, "list\_envs.py"),

&nbsp;   )

&nbsp;   for script in \["zero\_agent.py", "random\_agent.py"]:

&nbsp;       \_replace\_in\_file(

&nbsp;           \[(

&nbsp;               "# PLACEHOLDER: Extension template (do not remove this comment)",

&nbsp;               f"import {name}.tasks  # noqa: F401",

&nbsp;           )],

&nbsp;           src=os.path.join(ROOT\_DIR, "scripts", "environments", script),

&nbsp;           dst=os.path.join(dir, script),

&nbsp;       )

&nbsp;   # # docker files

&nbsp;   # print("  |-- Copying docker files...")

&nbsp;   # dir = os.path.join(project\_dir, "docker")

&nbsp;   # os.makedirs(dir, exist\_ok=True)

&nbsp;   # template = jinja\_env.get\_template("external/docker/.env.base")

&nbsp;   # \_write\_file(os.path.join(dir, ".env.base"), content=template.render(\*\*specification))

&nbsp;   # template = jinja\_env.get\_template("external/docker/docker-compose.yaml")

&nbsp;   # \_write\_file(os.path.join(dir, "docker-compose.yaml"), content=template.render(\*\*specification))

&nbsp;   # template = jinja\_env.get\_template("external/docker/Dockerfile")

&nbsp;   # \_write\_file(os.path.join(dir, "Dockerfile"), content=template.render(\*\*specification))

&nbsp;   # extension files

&nbsp;   print("  |-- Copying extension files...")

&nbsp;   # - config/extension.toml

&nbsp;   dir = os.path.join(project\_dir, "source", name, "config")

&nbsp;   os.makedirs(dir, exist\_ok=True)

&nbsp;   template = jinja\_env.get\_template("extension/config/extension.toml")

&nbsp;   \_write\_file(os.path.join(dir, "extension.toml"), content=template.render(\*\*specification))

&nbsp;   # - docs/CHANGELOG.rst

&nbsp;   dir = os.path.join(project\_dir, "source", name, "docs")

&nbsp;   os.makedirs(dir, exist\_ok=True)

&nbsp;   template = jinja\_env.get\_template("extension/docs/CHANGELOG.rst")

&nbsp;   \_write\_file(

&nbsp;       os.path.join(dir, "CHANGELOG.rst"), content=template.render({"date": datetime.now().strftime("%Y-%m-%d")})

&nbsp;   )

&nbsp;   # - setup.py and pyproject.toml

&nbsp;   dir = os.path.join(project\_dir, "source", name)

&nbsp;   template = jinja\_env.get\_template("extension/setup.py")

&nbsp;   \_write\_file(os.path.join(dir, "setup.py"), content=template.render(\*\*specification))

&nbsp;   shutil.copyfile(os.path.join(TEMPLATE\_DIR, "extension", "pyproject.toml"), os.path.join(dir, "pyproject.toml"))

&nbsp;   # - tasks

&nbsp;   print("  |-- Generating tasks...")

&nbsp;   dir = os.path.join(project\_dir, "source", name, name, "tasks")

&nbsp;   os.makedirs(dir, exist\_ok=True)

&nbsp;   specifications = \_generate\_tasks(specification, dir)

&nbsp;   shutil.copyfile(os.path.join(TEMPLATE\_DIR, "extension", "\_\_init\_\_tasks"), os.path.join(dir, "\_\_init\_\_.py"))

&nbsp;   for workflow in specification\["workflows"]:

&nbsp;       shutil.copyfile(

&nbsp;           os.path.join(TEMPLATE\_DIR, "extension", "\_\_init\_\_workflow"),

&nbsp;           os.path.join(dir, workflow\["name"].replace("-", "\_"), "\_\_init\_\_.py"),

&nbsp;       )

&nbsp;   # - other files

&nbsp;   dir = os.path.join(project\_dir, "source", name, name)

&nbsp;   template = jinja\_env.get\_template("extension/ui\_extension\_example.py")

&nbsp;   \_write\_file(os.path.join(dir, "ui\_extension\_example.py"), content=template.render(\*\*specification))

&nbsp;   shutil.copyfile(os.path.join(TEMPLATE\_DIR, "extension", "\_\_init\_\_ext"), os.path.join(dir, "\_\_init\_\_.py"))

&nbsp;   # .vscode files

&nbsp;   print("  |-- Copying vscode files...")

&nbsp;   dir = os.path.join(project\_dir, ".vscode")

&nbsp;   shutil.copytree(os.path.join(TEMPLATE\_DIR, "external", ".vscode"), dir, dirs\_exist\_ok=True)

&nbsp;   template = jinja\_env.get\_template("external/.vscode/tasks.json")

&nbsp;   \_write\_file(os.path.join(dir, "tasks.json"), content=template.render(\*\*specification))

&nbsp;   template = jinja\_env.get\_template("external/.vscode/tools/launch.template.json")

&nbsp;   \_write\_file(

&nbsp;       os.path.join(dir, "tools", "launch.template.json"), content=template.render(specifications=specifications)

&nbsp;   )

&nbsp;   # setup git repo

&nbsp;   print(f"Setting up git repo in {project\_dir} path...")

&nbsp;   \_setup\_git\_repo(project\_dir)

&nbsp;   # show end message

&nbsp;   print("\\n" + "-" \* 80)

&nbsp;   print(f"Project '{name}' generated successfully in {project\_dir} path.")

&nbsp;   print(f"See {project\_dir}/README.md to get started!")

&nbsp;   print("-" \* 80)





def get\_algorithms\_per\_rl\_library(single\_agent: bool = True, multi\_agent: bool = True):

&nbsp;   assert single\_agent or multi\_agent, "At least one of 'single\_agent' or 'multi\_agent' must be True"

&nbsp;   data = {"rl\_games": \[], "rsl\_rl": \[], "skrl": \[], "sb3": \[]}

&nbsp;   # get algorithms

&nbsp;   for file in glob.glob(os.path.join(TEMPLATE\_DIR, "agents", "\*\_cfg")):

&nbsp;       for rl\_library in data.keys():

&nbsp;           basename = os.path.basename(file).replace("\_cfg", "")

&nbsp;           if basename.startswith(f"{rl\_library}\_"):

&nbsp;               algorithm = basename.replace(f"{rl\_library}\_", "").upper()

&nbsp;               assert (

&nbsp;                   algorithm in SINGLE\_AGENT\_ALGORITHMS or algorithm in MULTI\_AGENT\_ALGORITHMS

&nbsp;               ), f"{algorithm} algorithm is not listed in the supported algorithms"

&nbsp;               if single\_agent and algorithm in SINGLE\_AGENT\_ALGORITHMS:

&nbsp;                   data\[rl\_library].append(algorithm)

&nbsp;               if multi\_agent and algorithm in MULTI\_AGENT\_ALGORITHMS:

&nbsp;                   data\[rl\_library].append(algorithm)

&nbsp;   # remove duplicates and sort

&nbsp;   for rl\_library in data.keys():

&nbsp;       data\[rl\_library] = sorted(list(set(data\[rl\_library])))

&nbsp;   return data





def generate(specification: dict) -> None:

&nbsp;   """Generate the project/task.



&nbsp;   Args:

&nbsp;       specification: The specification of the project/task.

&nbsp;   """

&nbsp;   # validate specification

&nbsp;   print("\\nValidating specification...")

&nbsp;   assert "external" in specification, "External flag is required"

&nbsp;   assert specification.get("name", "").isidentifier(), "Name must be a valid identifier"

&nbsp;   for workflow in specification\["workflows"]:

&nbsp;       assert workflow\["name"] in \["direct", "manager-based"], f"Invalid workflow: {workflow}"

&nbsp;       assert workflow\["type"] in \["single-agent", "multi-agent"], f"Invalid workflow type: {workflow}"

&nbsp;   if specification\["external"]:

&nbsp;       assert "path" in specification, "Path is required for external projects"

&nbsp;   # add other information to specification

&nbsp;   specification\["platform"] = sys.platform

&nbsp;   # generate project/task

&nbsp;   if specification\["external"]:

&nbsp;       print("Generating external project...")

&nbsp;       \_external(specification)

&nbsp;   else:

&nbsp;       print("Generating internal task...")

&nbsp;       print("  |-- Generating tasks...")

&nbsp;       \_generate\_tasks(specification, TASKS\_DIR)



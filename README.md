# label-studio-advanced-functions

# Project: Extended Functions for LabelStudio

## Project Description

This project provides extended functions for the [LabelStudio](https://labelstud.io/) service using its API. The main features include finding and fixing discrepancies in annotations, building statistical graphs, and improving the efficiency of entity relationship annotations.

## Environment Variables

### `.env.server` File

```env
TITLE=lbs_extended_function
SWAGGER=1
LABEL_STUDIO_API=

# --- Label studio config ----
LABEL_STUDIO_HOST=
LABEL_STUDIO_PORT=
```

### `.env` File (for Docker)

```env
# --- label studio ext func container ----
LBS_EXT_FA_HOST=0.0.0.0
LBS_EXT_FA_PORT=
LBS_EXT_LOCAL_FA_PORT=8000
```

### Main Features
- Finding Annotation Differences When Comparing Two Projects: Track discrepancies in annotations between two projects.
- Fixing Annotations: Automatically correct annotation errors such as extra spaces and word breaks.
- Building Comparison Statistics Graphs: Visualize comparison statistics in the form of graphs.
- Finding Relationship Annotation Differences When Comparing Two Projects: Track discrepancies in entity relationship annotations between two projects.
- Splitting Project Texts into "1 Task - 1 Sentence" Format: Split texts into sentences to improve the efficiency of entity relationship annotations.


### START

Create and configure the .env.server and .env files in the root directory of the project.

Run the project using Docker:
```
docker-compose up --build
```

### Usage
After starting the project, the API will be available at http://localhost:<LBS_EXT_LOCAL_FA_PORT>. You can use Swagger for testing and documenting the API if the SWAGGER variable is set to 1.

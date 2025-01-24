from diagrams import Diagram, Cluster
from diagrams.c4 import Person, Container

with Diagram("Workflow", direction="TB"):
    assist: Container = Container(
        name = "Assist.org",
        description = "Official course transfer and articulation system for California's public colleges and universities."
    )

    with Cluster("Data Ingest"):
        langchain: Container = Container(
            name = "Langchain API"
        )

        data_extract: Container = Container(
            name = "Extract Course Data",
            description = "We extract course data from raw text. (e.g. discipline, topics, school, etc.)"
        )
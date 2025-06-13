"""Module with the functions to generate the SynPop report."""
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional
from typing import Union

import yaml

from synpop.synpop_tables import _resolve_synpop_path


HERE = Path(__file__).parent

JUPYTERBOOK_CONFIG = {
    "title": "SynPop Report",
    "author": "MOBi Team",
    "logo": "simba-mobi-logo.png",
    "execute": {
        "execute_notebooks": "cache",
        "timeout": 600,
        "allow_errors": True,
        "stderr_output": "show",
    },
    "launch_buttons": {"binderhub_url": ""},
}

BASE_TOC = {
    "format": "jb-book",
    "root": "intro",
    "parts": [
        {
            "caption": "SynPop Structure",
            "chapters": [
                {"file": "population"},
                {"file": "age_structure"},
                {"file": "households"},
                {"file": "businesses"},
            ],
        },
        {
            "caption": "Person Attributes",
            "chapters": [
                {"file": "overview"},
                {"file": "nationality"},
                {"file": "level_of_employment"},
                {"file": "education"},
                {"file": "job_rank"},
                {"file": "mobility_tools"},
            ],
        },
    ],
}


def generate_report(  # pylint: disable=too-many-locals
    synpop_path: Union[Path, str],
    ref_synpop_path: Union[Path, str],
    output_report_folder: Union[Path, str],
    year: Optional[int] = None,
    year_reference: Optional[int] = None,
    validate: bool = True,
    cache: bool = True,
) -> Path:
    """Generate the SynPop report."""
    synpop_path = Path(synpop_path).resolve(strict=True)
    ref_synpop_path = Path(ref_synpop_path).resolve(strict=True)
    if not year:
        year = int(_resolve_synpop_path(synpop_path, "persons").stem.split("_")[1])
    if not year_reference:
        year_reference = int(
            _resolve_synpop_path(ref_synpop_path, "persons").stem.split("_")[1]
        )
    synpop_report_config = {
        "codebase": str(HERE.parent.resolve()),
        "reference_synpop": str(ref_synpop_path),
        "reference_year": year_reference,
        "target_synpop": str(synpop_path),
        "target_year": year,
        "fitting_output": str(synpop_path.joinpath("fitting_output")),
        "validate": validate,
    }

    report_sources = HERE.parent / "assets/report"
    temp_report_dir = Path(tempfile.mkdtemp()) / "report"
    # if rerun_chapters and cache_path is not None and cache_path.exists(): TODO: reactivate feature
    #     if not isinstance(rerun_chapters, list):
    #         rerun_chapters = [rerun_chapters]
    #     logging.info("Will re-run target chapters %s", ", ".join(rerun_chapters))
    #     for chapter in rerun_chapters:
    #         remove(cache_path.joinpath(f"_build/jupyter_execute/{chapter}.ipynb"))
    #         remove(cache_path.joinpath(f"_build/jupyter_execute/{chapter}.py"))
    #         # get newest version of the chapter file
    #         remove(cache_path.joinpath(f"{chapter}.md"))
    #         remove(cache_path.joinpath(f"{chapter}.md"))
    #         shutil.copy(
    #             report_sources.joinpath(f"{chapter}.md"),
    #             cache_path.joinpath(f"{chapter}.md"),
    #         )
    #     report_sources = cache_path
    #     old_path = Path(cache_path.parent.joinpath("temp_dir").open("r").read())
    #     temp_report_dir = Path(tempfile.mkdtemp(dir=old_path))
    #     temp_report_dir = temp_report_dir.parent / "report"

    # copy report sources to temp folder
    shutil.copytree(report_sources, temp_report_dir)

    # generate _config.yml file (with or without marginal_fitting)
    JUPYTERBOOK_CONFIG.update({"synpop_report_config": synpop_report_config})
    with (temp_report_dir / "_config.yml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(JUPYTERBOOK_CONFIG, file)

    # generate _toc.yml file
    fitting_output = Path(synpop_report_config["fitting_output"])
    if fitting_output.exists():
        chapters = []
        for file in [
            "current_edu",
            "current_job_rank",
            "language",
            "is_swiss",
            "is_employed",
            "pop_total",
        ]:
            if (fitting_output / file).exists():
                chapters.append({"file": f"marginal_fitting/{file}"})
            else:
                temp_report_dir.joinpath(f"marginal_fitting/{file}.md").unlink()
        BASE_TOC["parts"].append({"caption": "Marginal Fitting", "chapters": chapters})
    else:
        shutil.rmtree(temp_report_dir.joinpath("marginal_fitting"), ignore_errors=True)
    with (temp_report_dir / "_toc.yml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(BASE_TOC, file)

    # run jupyter-book
    subprocess.run(["jupyter-book", "build", "."], cwd=str(temp_report_dir), check=True)

    # copy resulting html folder back
    output_report_folder = Path(output_report_folder)
    output_report_folder.mkdir(exist_ok=True, parents=True)
    final_contents = output_report_folder / "synpop_report_content"
    final_index = output_report_folder / "synpop_report.html"
    if final_contents.exists():
        shutil.rmtree(final_contents)
    if final_index.exists():
        final_index.unlink()

    shutil.copytree(temp_report_dir / "_build/html", final_contents)

    # generate report.html folder on the side
    with final_index.open("w", encoding="utf-8") as file:
        file.write(
            '<meta http-equiv="Refresh" content="0; url=synpop_report_content/intro.html" />\n'
        )

    # store report folder in local cache and clear from temp
    if cache:
        cache_path = output_report_folder / "report_cache"
        if cache_path.exists():
            shutil.rmtree(cache_path)
        shutil.copytree(temp_report_dir, cache_path)
    shutil.rmtree(temp_report_dir)

    logging.info("Report is located at: %s.", final_index.resolve())
    return final_index.resolve()

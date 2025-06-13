"""Command line interface for SynPop."""
import logging
import runpy
import sys
from pathlib import Path
from typing import List
from typing import Optional
from typing import Union

import click

logging.getLogger().setLevel(logging.INFO)

CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}


@click.group(context_settings=CONTEXT_SETTINGS)
def synpop():
    """Synthetic population manipulation and visualization."""


@synpop.command()
@click.argument(
    "synpop_path",
    required=True,
    type=click.Path(exists=True),
)
@click.argument(
    "ref_synpop_path",
    required=True,
    type=click.Path(exists=True),
)
@click.argument(
    "year",
    required=False,
    type=click.INT,
)
@click.argument(
    "year_ref",
    required=False,
    type=click.INT,
)
@click.option(
    "--output_folder",
    "-o",
    required=False,
    type=click.Path(),
    help="Output folder for the report SynPop. Defaults to the current working directory.",
)
@click.option(
    "--no-cache",
    "-nc",
    is_flag=True,
    help="Cache the build files in the output folder. Defaults to True. \
        Required for re-running chapters.",
)
@click.option(
    "-sv",
    "--skip-validation",
    is_flag=True,
    help="Whether to validate the SynPop.",
)
def report(
    synpop_path: str,
    ref_synpop_path: str,
    year: Optional[int] = None,
    year_ref: Optional[int] = None,
    output_folder: Optional[str] = None,
    no_cache: bool = False,
    skip_validation: bool = False,
) -> None:
    """Generate a SynPop report."""
    from synpop.visualization.report import (  # pylint: disable=import-outside-toplevel
        generate_report,
    )

    if output_folder is None:
        output_folder = str(Path(synpop_path).parent / "report")
    click.echo(f"Generating SynPop report and writing to {output_folder}...")
    generate_report(
        synpop_path=synpop_path,
        ref_synpop_path=ref_synpop_path,
        year=year,
        year_reference=year_ref,
        output_report_folder=output_folder,
        cache=not no_cache,
        validate=not skip_validation,
    )


@synpop.command()
@click.argument("year", type=click.INT, required=True)
@click.argument("synpop_path", required=True, type=click.Path(exists=True))
@click.argument("output_synpop_path", required=True, type=click.Path())
@click.argument("synpop_parse_config", required=False, type=click.Path())
@click.option("-v", "validate", default=True)
def preprocess(
    year: int,
    synpop_path: Union[str, Path],
    output_synpop_path: Union[str, Path],
    synpop_parse_config: Union[str, Path],
    validate: bool = True,
) -> None:
    """Preprocess a raw SynPop."""
    from synpop import api  # pylint: disable=import-outside-toplevel

    api.parse_raw_synpop(
        year=year,
        synpop_folder=synpop_path,
        synpop_preprocessing_config=synpop_parse_config,
        validate=validate,
    ).write(output_synpop_path)


@synpop.command()
@click.argument("synpop_path", required=True, type=click.Path(exists=True))
@click.argument("output_synpop_path", required=True, type=click.Path())
@click.argument(
    "blurring_target",
    required=False,
    type=click.STRING,
    # help="Which of the three SynPop objects is to be anonymized.",
)
@click.argument(
    "blurring_hh_min_size_threshold",
    required=False,
    type=click.INT,
    # help="Minimum household per location. \
    # Combines single-household-buildings to improve anonymization.",
)
def anonymize(
    synpop_path: Union[str, Path],
    output_synpop_path: Union[str, Path],
    blurring_target: str = "households",
    blurring_hh_min_size_threshold: int = 1,
) -> None:
    """Anonymize a SynPop."""
    from synpop import api  # pylint: disable=import-outside-toplevel
    from synpop.synpop_tables import (  # pylint: disable=import-outside-toplevel
        SynPop,
    )

    if blurring_target not in ["households", "persons", "businesses"]:
        raise click.BadParameter(
            "Blurring target must be one of 'households', 'persons', or 'businesses'."
        )
    synpop_ = SynPop.from_path(synpop_path)
    anonymized = api.anonymize(synpop_, blurring_target, blurring_hh_min_size_threshold)
    anonymized.write(output_synpop_path)


@synpop.command()
@click.argument("synpop_path", required=True, type=click.Path(exists=True))
@click.argument("output_synpop_path", required=True, type=click.Path())
@click.argument("ref_synpop_path", required=False, type=click.Path(exists=True))
@click.option("-t", "--fitting_tables", required=True, type=click.Path(), multiple=True)
@click.option(
    "-s",
    "--write_stats",
    is_flag=True,
    help="Set flag if stats are required. May need 'ref_synpop_path' for it to work.",
)
@click.option(
    "-nohh",
    "--clone-single-agents",
    is_flag=True,
    help="Set flag if cloning entire households is not desired. Improves accuracy in difficult tight fitting but reduces realism.",
)
def fit(
    synpop_path: Union[str, Path],
    output_synpop_path: Union[str, Path],
    fitting_tables: List[Path],
    ref_synpop_path: Optional[Union[str, Path]] = None,
    write_stats: bool = False,
    clone_single_agents: bool = False,
) -> None:
    """Fit a SynPop to target counts from fitting excel."""
    from synpop.synpop_tables import (  # pylint: disable=import-outside-toplevel
        SynPop,
    )
    from synpop import api  # pylint: disable=import-outside-toplevel
    from synpop.fitting.fitting_config import (  # pylint: disable=import-outside-toplevel
        parse_fitting_goals,
    )

    synpop_ = SynPop.from_path(synpop_path)
    for fitting_tables_path in fitting_tables:
        fitting_configs = parse_fitting_goals(fitting_tables_path)
        click.echo(f"Fitting SynPop for targets: {fitting_configs.keys()}")

        if "pop_total" in fitting_configs:
            synpop_ = api.fit_population(
                synpop_,
                fitting_tables_path,
                fitting_configs["pop_total"],
                Path(output_synpop_path) / "fitting_output",
                write_stats=write_stats,
                entire_households=not clone_single_agents,
            )
            del fitting_configs["pop_total"]
        if "jobs_endo" in fitting_configs:
            synpop_ = api.fit_businesses(
                synpop_,
                fitting_tables_path,
                fitting_configs["jobs_endo"],
                Path(output_synpop_path) / "fitting_output",
                write_stats=write_stats,
            )
            del fitting_configs["jobs_endo"]
        if "jobs_exo" in fitting_configs:
            synpop_ = api.fit_businesses(
                synpop_,
                fitting_tables_path,
                fitting_configs["jobs_exo"],
                Path(output_synpop_path) / "fitting_output",
                write_stats=write_stats,
            )
            del fitting_configs["jobs_exo"]
        if len(fitting_configs) > 0:
            ref_synpop = None
            if ref_synpop_path is not None:
                ref_synpop = SynPop.from_path(ref_synpop_path)
            synpop_ = api.fit_marginals(
                synpop_,
                ref_synpop,
                fitting_tables_path,
                Path(output_synpop_path) / "fitting_output",
                write_stats=write_stats,
            )
    synpop_.write(output_synpop_path)


@synpop.command()
@click.argument("synpop_path", required=True, type=click.Path(exists=True))
@click.argument("output_path", required=False, type=click.Path(exists=False))
@click.argument("aggregation", required=False, type=click.STRING)
@click.argument("aggregate_age_groups", required=False, type=click.BOOL)
@click.argument("aggregate_extra_vars", required=False, type=click.BOOL)
def aggregate(
    synpop_path: Path,
    output_path: Optional[Union[str, Path]] = None,
    aggregation: str = "zone_id",
    aggregate_age_groups: bool = False,
    aggregate_extra_vars: bool = False,
) -> None:
    """Aggregate a SynPop to a given geographical level."""
    from synpop.synpop_tables import (  # pylint: disable=import-outside-toplevel
        SynPop,
    )
    from synpop.visualization import (  # pylint: disable=import-outside-toplevel
        zone_maps,
    )

    synpop_path = Path(synpop_path)
    synpop_ = SynPop.from_path(synpop_path)
    gdf = zone_maps.aggregate_synpop(
        synpop_.persons.data,
        synpop_.businesses.data,
        aggregate_age_groups,
        aggregate_extra_vars,
    )
    if aggregation != "zone_id":
        gdf = gdf.dissolve(by=aggregation, aggfunc="sum")
    if output_path is None:
        output_path = synpop_path.parent if synpop_path.is_file() else synpop_path
        output_path = output_path / f"mobi-zones_{synpop_.year}"
    else:
        output_path = Path(output_path)
    click.echo(f"Writing aggregated SynPop to {output_path}")
    gdf.to_file(output_path.with_suffix(".gpkg"), driver="GPKG")  # type: ignore
    gdf.drop("geometry", axis=1).to_csv(output_path.with_suffix(".csv"), sep=";")  # type: ignore


def _validate_geoquery(_, __, value):
    geo_cols = [
        "kt_name",
        "mun_name",
        "agglo_name",
        "amgr_name",
        "amr_name",
        "msr_name",
        "sl3_name",
    ]
    if value is not None:
        parts = value.split("==")
        if len(parts) != 2 or '"' in parts[1]:
            raise click.BadParameter('geoquery must be of the format key=="value"')
        if parts[0] not in geo_cols:
            raise click.BadParameter(f"geoquery key must be one of {geo_cols}.")
    return value


@synpop.command()
@click.argument("zones1", required=False, type=click.Path(exists=True))
@click.argument("suffix1", required=False)
@click.argument("zones2", required=False, type=click.Path(exists=True))
@click.argument("suffix2", required=False)
@click.argument("Variable", required=False)
@click.argument("Aggregation", required=False)
@click.argument("geoquery", required=False, callback=_validate_geoquery)
def zzwidget(
    zones1: Optional[Path] = None,
    suffix1: Optional[str] = None,
    zones2: Optional[Path] = None,
    suffix2: Optional[str] = None,
    variable: Optional[str] = None,
    aggregation: Optional[str] = None,
    geoquery: Optional[str] = None,
) -> None:
    """Analyse mobi-zones across different spatial aggregation levels."""
    click.echo("Starting Zonal Comparison ...")
    from synpop.widget import (  # pylint: disable=import-outside-toplevel
        run_zone_diff_widget,
    )

    run_zone_diff_widget(
        datasource1=zones1,
        datasource2=zones2,
        suffix1=suffix1,
        suffix2=suffix2,
        target_variable=variable,
        groupby_agg=aggregation,
        geoquery=geoquery,
    )


@synpop.command()
@click.option(
    "-l",
    "--local",
    is_flag=True,
    default=False,
    help="Run in local mode (instead of using data from Snowflake).",
)
@click.option(
    "-z",
    "--add-zones",
    multiple=True,
    help="Additional zones in the format name=path.",
)
def st_zzwidget(
    local: bool,
    add_zones: tuple[str],
) -> None:
    """Analyse mobi-zones across different spatial aggregation levels."""
    click.echo("Starting Zonal Comparison ...")
    from synpop.visualization import (  # pylint: disable=import-outside-toplevel
        zzwidget as st_zzwidget_,
    )

    sys.argv = ["streamlit", "run", st_zzwidget_.__file__, str(local)] + list(add_zones)
    runpy.run_module("streamlit", run_name="__main__")


def main():
    synpop()

if __name__ == "__main__":
    main()

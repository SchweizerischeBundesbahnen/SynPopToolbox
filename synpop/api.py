import shutil
from pathlib import Path
import tempfile
import subprocess
import logging

import pandas as pd
import yaml

from synpop.preprocessing import SynPopPreprocessor
from synpop.zone_maps import MOBI_ZONES
from synpop.synpop_tables import SynPop
from synpop.anonymization import blur_per_zone
from synpop import marginal_fitting

HERE = Path(__file__).parent


def load_synpop(year, synpop_folder):
    synpop = SynPop(year)
    synpop.load(synpop_folder)
    if 'is_swiss' not in synpop.persons.data.columns:
        synpop.persons.data['is_swiss'] = synpop.persons.data['nation'] == 'swiss'
    return synpop


def preprocess(input_falc_folder, year, synpop_loading_config=None):
    preprocessor = SynPopPreprocessor(input_falc_folder, year=year, mobi_zones_shapefile_path=MOBI_ZONES,
                                      config_path=synpop_loading_config)
    persons, households = preprocessor.preprocess_persons_and_households()
    businesses = preprocessor.preprocess_businesses()
    synpop = SynPop(year, persons, households, businesses)
    return synpop


def anonymize(synpop, blurring_target='households', blurring_hh_min_size_threshold=1):
    blurring_df = getattr(synpop, blurring_target)
    df = blur_per_zone(blurring_df.data, id_var=blurring_target[:-1]+'_id',
                       hh_min_size_threshold=blurring_hh_min_size_threshold)
    setattr(synpop, blurring_target, df)
    if blurring_target == 'households':
        synpop.persons.data = synpop.persons.data.drop(['xcoord', 'ycoord'], axis=1)
        synpop.persons.data = pd.merge(synpop.persons.data, df, how='left', on='household_id')
    return synpop


def fit_population():
    raise NotImplementedError


def fit_marginals(synpop: SynPop, synpop_ref: SynPop, fitting_tables_path, output_synpop_folder=None):
    configs = marginal_fitting.parse_fitting_goals(fitting_tables_path)

    fitting_outputs = None
    if output_synpop_folder:
        fitting_outputs = Path(output_synpop_folder) / 'fitting_outputs'

    persons_fixed = synpop.persons.data.copy()
    for target_variable in configs.keys():
        config = configs[target_variable][0]
        expected_pop_counts = configs[target_variable][1]
        persons_fixed = marginal_fitting.fit_target_variable(persons_fixed,
                                                             synpop_ref.persons.data,
                                                             synpop.year,
                                                             synpop_ref.year,
                                                             config,
                                                             expected_pop_counts,
                                                             fitting_outputs
                                                             )
    return SynPop(synpop.year, persons_fixed, synpop.households.data, synpop.businesses.data)


def write_synpop(synpop: SynPop, output_synpop_folder):
    if not Path(output_synpop_folder).exists():
        Path(output_synpop_folder).mkdir(parents=True)
    synpop.write(output_synpop_folder)
    return output_synpop_folder


def generate_report(output_synpop_folder, year, reference_synpop_path, year_reference,
                    output_report_folder=str(HERE.parent), rerun_chapters='', cache_path=None):
    jbook_config = {
        'title': 'SynPop Report',
        'author': 'MOBi Team',
        'logo': 'simba-mobi-logo.png',
        'execute': {
            'execute_notebooks': 'cache',
            'timeout': 600,
            'allow_errors': True,
            'stderr_output': 'show'},
        'launch_buttons': {'binderhub_url': ""}
    }
    output_synpop_folder = Path(output_synpop_folder)
    reference_synpop_path = Path(reference_synpop_path)
    synpop_report_config = {
        'codebase': str(HERE.parent.resolve()),
        'reference_synpop': str(reference_synpop_path.resolve()),
        'reference_year': year_reference,
        'target_synpop': str(output_synpop_folder.resolve()),
        'target_year': year,
        'fitting_outputs': str(output_synpop_folder.joinpath('fitting_outputs').resolve())
    }

    toc = {'format': 'jb-book',
           'root': 'intro',
           'parts': [{'caption': 'SynPop Structure',
                      'chapters': [{'file': 'population'},
                                   {'file': 'age_structure'},
                                   {'file': 'households'},
                                   {'file': 'businesses'}]},
                     {'caption': 'Person Attributes',
                      'chapters': [{'file': 'nationality'},
                                   {'file': 'language'},
                                   {'file': 'level_of_employment'},
                                   {'file': 'education'},
                                   {'file': 'job_rank'},
                                   {'file': 'mobility_tools'}]}]}
    toc_fitting = {'caption': 'Marginal Fitting',
                   'chapters': [{'file': 'marginal_fitting/language_fitting'},
                                {'file': 'marginal_fitting/is_swiss_fitting'},
                                {'file': 'marginal_fitting/education_fitting'},
                                {'file': 'marginal_fitting/job_rank_fitting'},
                                {'file': 'marginal_fitting/is_employed_fitting'}]}

    report_sources = HERE.parent / "report"
    temp_report_dir = Path(tempfile.mkdtemp()) / "report"
    if rerun_chapters and cache_path is not None and cache_path.exists():
        if not isinstance(rerun_chapters, list):
            rerun_chapters = [rerun_chapters]
        logging.info(f'Will re-run target chapters {", ".join(rerun_chapters)}')
        for ch in rerun_chapters:
            rm(cache_path.joinpath(f'_build/jupyter_execute/{ch}.ipynb'))
            rm(cache_path.joinpath(f'_build/jupyter_execute/{ch}.py'))
            # get newest version of the chapter file
            rm(cache_path.joinpath(f'{ch}.md'))
            rm(cache_path.joinpath(f'{ch}.md'))
            shutil.copy(report_sources.joinpath(f'{ch}.md'), cache_path.joinpath(f'{ch}.md'))
        report_sources = cache_path
        old_path = Path(cache_path.parent.joinpath('temp_dir').open('r').read())
        temp_report_dir = Path(tempfile.mkdtemp(dir=old_path))
        temp_report_dir = temp_report_dir.parent / 'report'

    # copy report sources to temp folder
    shutil.copytree(report_sources, temp_report_dir)

    # generate _config.yml file (with or without marginal_fitting)
    jbook_config.update({'synpop_report_config': synpop_report_config})
    with (temp_report_dir / "_config.yml").open('w') as f:
        yaml.safe_dump(jbook_config, f)

    # generate _toc.yml file
    if Path(synpop_report_config['fitting_outputs']).exists():
        toc['parts'].append(toc_fitting)
    else:
        shutil.rmtree(temp_report_dir.joinpath('marginal_fitting'), ignore_errors=True)
    with (temp_report_dir / "_toc.yml").open('w') as f:
        yaml.safe_dump(toc, f)

    # run jupyter-book
    subprocess.run(["jupyter-book", "build", '.'], cwd=str(temp_report_dir))

    # copy resulting html folder back
    final_contents = Path(output_report_folder) / 'synpop_report_content'
    final_index = Path(output_report_folder) / 'synpop_report.html'
    if final_contents.exists():
        shutil.rmtree(final_contents)
    if final_index.exists():
        final_index.unlink()

    shutil.copytree(temp_report_dir / '_build/html', final_contents)

    # generate report.html folder on the side
    with final_index.open('w') as f:
        f.write('<meta http-equiv="Refresh" content="0; url=synpop_report_content/intro.html" />\n')

    # store report folder in local cache and clear from temp
    if cache_path.exists():
        shutil.rmtree(cache_path)
    shutil.copytree(temp_report_dir, cache_path)
    with cache_path.parent.joinpath('temp_dir').open('w') as f:
        f.write(str(temp_report_dir))
    shutil.rmtree(temp_report_dir)

    logging.info(f'Report is located at: {final_index.resolve()}')
    return str(final_index.resolve())


def rm(file):
    if Path(file).exists():
        Path(file).unlink()


import logging
import os
from pathlib import Path

import click
import yaml
from synpp import stage
import synpp

from synpop.synpop_tables import SynPop
from synpop import api

HERE = Path(__file__).parent


# helper stages to resolve whether optional stages are run or not
class WriteStageResolver:
    def configure(self, context: synpp.ConfigurationContext):
        output_folder = context.config('output_synpop_folder', None)
        if (not output_folder) or (not Path(output_folder).exists()) or (context.stage_is_config_requested(preprocess)):
            context.stage(write_synpop, alias='synpop_path')
            return
        output_folder = Path(output_folder)
        has_persons = len(list(output_folder.glob('*persons*'))) >= 1
        has_households = len(list(output_folder.glob('*households*'))) >= 1
        has_businesses = len(list(output_folder.glob('*businesses*'))) >= 1
        if not all([has_businesses, has_households, has_persons]):
            context.stage(write_synpop, alias='synpop_path')
        else:
            def __output_folder__():
                return output_folder
            context.stage(stage(__output_folder__), alias='synpop_path')

    def execute(self, context: synpp.ExecuteContext):
        return context.stage('synpop_path')


class AnonymizeStageResolver:
    def configure(self, context: synpp.ConfigurationContext):
        if context.stage_is_config_requested(anonymize):
            context.stage(anonymize, alias='synpop')
        else:
            context.stage(preprocess, alias='synpop')

    def execute(self, context: synpp.ExecuteContext):
        return context.stage('synpop')


class FittingStageResolver:
    def configure(self, context: synpp.ConfigurationContext):
        if context.stage_is_config_requested(fit_marginals):
            context.stage(fit_marginals, alias='synpop')
        elif context.stage_is_config_requested(anonymize):
            context.stage(anonymize, alias='synpop')
        else:
            context.stage(preprocess, alias='synpop')

    def execute(self, context: synpp.ExecuteContext):
        return context.stage('synpop')


@stage
def load_synpop(year, synpop_folder):
    return api.load_synpop(year, synpop_folder)


@stage
def preprocess(input_falc_folder, year, synpop_loading_config=None):
    return api.preprocess(input_falc_folder, year, synpop_loading_config)


@stage(synpop=preprocess)
def anonymize(synpop, blurring_target='households', blurring_hh_min_size_threshold=1):
    return api.anonymize(synpop, blurring_target, blurring_hh_min_size_threshold)


@stage(synpop=AnonymizeStageResolver,
       synpop_ref=stage(load_synpop, year='year_reference', synpop_folder='reference_synpop_path'))
def fit_marginals(synpop: SynPop, synpop_ref: SynPop, fitting_tables_path, output_synpop_folder=None):
    return api.fit_marginals(synpop, synpop_ref, fitting_tables_path, output_synpop_folder)


@stage(synpop=FittingStageResolver)
def write_synpop(synpop: SynPop, output_synpop_folder):
    return api.write_synpop(synpop, output_synpop_folder)


@stage(output_synpop_folder=WriteStageResolver)
def generate_report(output_synpop_folder, year, reference_synpop_path, year_reference,
                    output_report_folder=str(HERE.parent), rerun_chapters=''):
    return api.generate_report(output_synpop_folder, year, reference_synpop_path, year_reference, output_report_folder,
                               rerun_chapters, Path(synpp.get_context().path()) / 'report')


@click.group(context_settings=dict(help_option_names=['-h', '--help']))
def synpop():
    pass


@synpop.command()
@click.argument('config_path', required=True, type=click.Path())
def run(config_path=None):
    click.echo("Running SynPop pipeline...")
    _run(config_path)


def _run(config_path):
    config = setup_config(config_path, None)
    _ = synpp.run([{'descriptor': s} for s in config['run']], config=config['config'],
                  working_directory=config['working_directory'],
                  ensure_working_directory=True, rerun_required=False)


@synpop.command()
@click.argument('config_path', required=True, type=click.Path())
@click.argument('output_report_folder', required=False, type=click.Path())
@click.argument('cache_path', required=False, type=click.Path())
# @click.argument('rerun_chapters', required=False)  # not working so far since synpp will devalidate anyway (new arg)
def report(config_path, output_report_folder=None, cache_path=None):
    click.echo("Generating SynPop report...")
    _report(config_path, output_report_folder, cache_path)


def _report(config_path, output_report_folder=None, cache_path=None):
    config = setup_config(config_path, cache_path)
    if output_report_folder is None:
        output_report_folder = Path(config_path).parent
    output_report_folder = Path(output_report_folder)
    output_report_folder.mkdir(exist_ok=True, parents=True)
    _ = synpp.run([{'descriptor': generate_report, 'config': {'output_report_folder': str(output_report_folder)}}],
                  config=config['config'], working_directory=config['working_directory'],
                  ensure_working_directory=True, rerun_required=True)


def setup_config(config_path=None, cache_path=None):
    logging.getLogger().setLevel(logging.DEBUG)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # paths made relative to config whenever not absolute
    os.chdir(Path(config_path).parent)
    path_params = ['synpop_loading_config', 'output_synpop_folder', 'reference_synpop_path',
                   'input_falc_folder', 'fitting_tables_path']
    for p in path_params:
        if p in config['config'].keys():
            path = Path(config['config'][p])
            if not path.is_absolute():
                path = Path(config_path).parent / path
                config['config'][p] = str(path)
    if cache_path is None and 'working_directory' not in config.keys():
        config['working_directory'] = Path(config_path).parent / 'cache'
    if not Path(config['working_directory']).exists() and not Path(config['working_directory']).is_absolute():
        config['working_directory'] = Path(config_path).parent / config['working_directory']

    return config


def invoke_wrapper(f):
    import functools
    import sys
    """Augment CliRunner.invoke to emit its output to stdout.

    This enables pytest to show the output in its logs on test
    failures.

    """

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        echo = kwargs.pop('echo', False)
        result = f(*args, **kwargs)

        if echo is True:
            sys.stdout.write(result.output)
            with open('log.txt', 'a') as log:
                log.write(result.output)

        return result

    return wrapper


if __name__ == '__main__':
    synpop()

    # Run from PyCharm
    config_path = r'Z:\40_Projekte\20210407_Prognose_2050\02_SynPop\Sprint2\PreProcessed\2050_uncorrected\config.yml'
    # _run(config_path)

    # CLI debugging
    from click.testing import CliRunner
    runner_class = CliRunner
    runner_class.invoke = invoke_wrapper(runner_class.invoke)
    runner = runner_class()
    # runner.invoke(synpop, ['report', config_path])

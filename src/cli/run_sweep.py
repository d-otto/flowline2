import click
import time

from flowline.sweep import FlowlineSweep

@click.command()
@click.argument('config_file', type=click.Path(exists=True, dir_okay=False))
@click.option('-o', '--output-dir', 'output_dir', type=click.Path(),
              default=None,
              help="Directory to save sweep results. [default: sweep_output_<timestamp>]")
@click.option('--workers', type=int, default=None,
              help="Number of Dask workers (cores) to use. [default: all available]")
@click.option('--no-combine', is_flag=True, default=False,
              help="Do not combine individual run outputs into a single file.")
def main(config_file, output_dir, workers, no_combine):
    """
    Run a flowline model parameter sweep from a CONFIG_FILE.
    """
    sweep = FlowlineSweep(
        config_file=config_file,
        output_dir=output_dir,
        workers=workers,
        no_combine=no_combine
    )
    sweep.run()

if __name__ == '__main__':
    main()

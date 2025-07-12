"""
CLI utilities for flowline parameter sweeps.

This module provides reusable command-line interface functionality that can be
easily added to any sweep script to handle common arguments like output directory,
worker count, and result combination options.
"""

import argparse


def add_sweep_cli_args(parser):
    """
    Add standard sweep CLI arguments to an ArgumentParser.
    
    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser to add arguments to.
        
    Returns
    -------
    argparse.ArgumentParser
        The same parser with sweep arguments added.
    """
    parser.add_argument(
        '-o', '--output-dir', 
        type=str,
        default=None,
        help="Directory to save sweep results. [default: sweep_output_<timestamp>]"
    )
    parser.add_argument(
        '--workers', 
        type=int, 
        default=None,
        help="Number of Dask workers (cores) to use. [default: all available]"
    )
    parser.add_argument(
        '--no-combine', 
        action='store_true',
        help="Do not combine individual run outputs into a single file after the sweep."
    )
    parser.add_argument(
        '--no-progress', 
        action='store_true',
        help="Disable progress bars (tqdm)."
    )
    return parser


def parse_sweep_cli_args(description="Run a flowline model parameter sweep."):
    """
    Create a parser with sweep arguments and parse command line arguments.
    
    Parameters
    ----------
    description : str, optional
        Description for the argument parser.
        
    Returns
    -------
    argparse.Namespace
        Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description=description)
    add_sweep_cli_args(parser)
    return parser.parse_args()


def get_sweep_cli_kwargs(args=None):
    """
    Get sweep arguments as keyword arguments suitable for FlowlineSweep.
    
    Parameters
    ----------
    args : argparse.Namespace, optional
        Parsed arguments. If None, will parse from command line.
        
    Returns
    -------
    dict
        Dictionary of keyword arguments for FlowlineSweep constructor.
    """
    if args is None:
        args = parse_sweep_cli_args()
    
    return {
        'output_dir': args.output_dir,
        'workers': args.workers,
        'no_combine': args.no_combine,
        'no_progress': args.no_progress
    }
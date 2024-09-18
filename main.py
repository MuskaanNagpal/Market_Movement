#!/usr/bin/env python
import typer
import pandas as pd

from pathlib import Path

import module


cli = typer.Typer( pretty_exceptions_enable=False )

model_path = 'xgb_model.pkl'
scaler_path = 'scaler.pkl'


@cli.command()
def train(
    input_data_path: Path = typer.Option( 'data/train.csv', '--input-data-path', '-i', help='Path to the input data file'),
):
    module.train_model( input_data_path, model_path, scaler_path )

    data = pd.read_csv( input_data_path )
    module.evaluate_model( data, model_path, scaler_path )


@cli.command()
def test(
    input_data_path: Path = typer.Option( 'data/test.csv', '--input-data-path', '-i', help='Path to the input data file'),
):
    data = pd.read_csv( input_data_path )
    module.evaluate_model( data, model_path, scaler_path )

if __name__ == "__main__":
    cli()
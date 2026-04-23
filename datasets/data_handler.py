import pandas as pd
from pathlib import Path

DIR = Path(__file__).resolve().parent


def load_chf(train_path='chf_train_synth.csv', test_path='chf_test_synth.csv'):
    """
    CHF Data — Pre-split train/test

    Features: D, L, P, G, Tin, Xe
    Target: CHF

    :return: features, targets, train_df, test_df
    """
    # Load the CSV files
    train_df = pd.read_csv(DIR / train_path)
    test_df = pd.read_csv(DIR / test_path)

    feature_map = {
           'D (m)': "D",
           'L (m)': "L",
           'P (kPa)': "P",
           'G (kg m-2s-1)': "G",
           'Tin (C)': "Tin",
           'Xe (-)': "Xe",
           'CHF (kW m-2)': "CHF",
    }
    train_df.rename(columns=feature_map, inplace=True)
    test_df.rename(columns=feature_map, inplace=True)
    feature_map.pop("CHF (kW m-2)")

    return feature_map.values(), ["CHF"], train_df, test_df


def load_reactor(input_path='rea_inputs.csv', output_path='rea_outputs.csv'):
    """
    MIT Reactor Data

    Inputs: rod_worth, beta, h_gap, gamma_frac
    Outputs: max_power, burst_width, max_Tf, avg_Tcool

    :return: features, targets, data_df
    """
    # Load CSVs
    x_df = pd.read_csv(DIR / input_path)
    y_df = pd.read_csv(DIR / output_path)

    # Combine inputs and outputs into a single DataFrame
    data_df = pd.concat([x_df, y_df], axis=1)

    feature_cols = ['rod_worth', 'beta', 'h_gap', 'gamma_frac']
    target_cols = ['max_power', 'burst_width', 'max_Tf', 'avg_Tcool']

    return feature_cols, target_cols, data_df


def load_bwr(input_path="bwr_input.csv", output_path="bwr_output.csv"):
    """
    Boiler Water Reactor Data

    Inputs: PSZ, DOM, vanA, vanB, subcool, CRD, flow_rate, power_density, VFNGAP
    Outputs: K-eff, Max3Pin, Max4Pin, F-delta-H, Max-Fxy

    :return: features, targets, data_df
    """
    x_df = pd.read_csv(DIR / input_path)
    y_df = pd.read_csv(DIR / output_path)

    data_df = pd.concat([x_df, y_df], axis=1)

    feature_cols = ['PSZ', 'DOM', 'vanA', 'vanB', 'subcool', 'CRD', 'flow_rate', 'power_density', 'VFNGAP']
    target_cols = ['K-eff', 'Max3Pin', 'Max4Pin', 'F-delta-H', 'Max-Fxy']

    return feature_cols, target_cols, data_df


if __name__ == "__main__":
    print(DIR)


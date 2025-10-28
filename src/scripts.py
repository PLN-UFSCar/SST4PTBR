import pandas as pd
import os
FILES_DIRECTORY = os.path.join(os.path.dirname(__file__), '..', 'PLNCrawler', 'datasets')

def get_df_sensacionalista():
    """Returns Sensacionalista news as a pandas DataFrame."""
    return pd.read_json(FILES_DIRECTORY + '/sensacionalista.json', lines=True)

def get_df_estadao():
    """Returns Estadão news as a pandas DataFrame."""
    return pd.read_json(FILES_DIRECTORY + '/estadao.json', lines=True)

def get_df_the_piaui_herald():
    """Returns The piaui Herald news as a pandas DataFrame."""
    return pd.read_json(FILES_DIRECTORY + '/the_piaui_herald.json', lines=True)

def merge_dfs(df_sensacionalista, df_estadao: pd.DataFrame, df_piaui):
    """
    Merges the dataframes from the three newspapers in a balanced way.

    Returns:
    - pd.DataFrame: Unified DataFrame.
    """

    num_sarcastics_samples = len(df_sensacionalista) + len(df_piaui)

    new_df = pd.concat([df_sensacionalista, df_piaui, df_estadao.sample(num_sarcastics_samples, random_state=42)], ignore_index=True)


    # Shuffle the resulting DataFrame
    new_df = new_df.sample(frac=1, random_state=42).reset_index(drop=True)

    return new_df
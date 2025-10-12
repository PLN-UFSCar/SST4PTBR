import pandas as pd
import os
FILES_DIRECTORY = os.path.join(os.path.dirname(__file__), '..', 'PLNCrawler', 'datasets')

def get_df_sensacionalista():
    """Retorna as notícias do Sensacionalista como um DataFrame do pandas."""
    return pd.read_json(FILES_DIRECTORY + '/sensacionalista.json', lines=True)

def get_df_estadao():
    """Retorna as notícias do Estadão como um DataFrame do pandas."""
    return pd.read_json(FILES_DIRECTORY + '/estadao.json', lines=True)

def get_df_the_piaui_herald():
    """Retorna as notícias do The piaui Herald como um DataFrame do pandas."""
    return pd.read_json(FILES_DIRECTORY + '/the_piaui_herald.json', lines=True)

def merge_dfs(df_sensacionalista, df_estadao: pd.DataFrame, df_piaui):
    """
    Une os dataframes dos três jornais de maneira equilibrada.

    Retorna:
    - pd.DataFrame: DataFrame unificado.
    """

    num_sarcastics_samples = len(df_sensacionalista) + len(df_piaui)

    new_df = pd.concat([df_sensacionalista, df_piaui, df_estadao.sample(num_sarcastics_samples, random_state=42)], ignore_index=True)


    # Embaralhar o DataFrame resultante
    new_df = new_df.sample(frac=1, random_state=42).reset_index(drop=True)

    return new_df






                                                                                                                                                                        # src/ingestion.py
from __future__ import annotations

from datetime import datetime, timezone
import pandas as pd

from .paths import raw_data_path


def load_raw_data() -> pd.DataFrame: 
    """
    Loading the raw telco churn CSV and attach the provenance metadata
    
    Below is a list of why the provenance columns are important (DE mindset):
    - Traceability: there must be evidence to prove what file a dataset came from
    - Reproducibility: the dataset loading time is always tracked 
    - Debugging: provenance helps to isolate distortions if downstream steps produce strange results
    
    Returns
    -------
    pd.DataFrame
        A raw dataset with provenance metadata columns:
        - __source_file: this is where the data came from
        - __ingested_at_utc: which is when the file was read (UTC ISO-8601 string)
    """ 
    # Reading the raw CSV from a central path constant
    # Please note: If the dataset has strange NA strings (" " etc.), then you can control that here
    # with no 'no_values = [...]' and 'keep_default_na = True'.
    data = pd.read_csv(raw_data_path)
    
    # Provenance (Traceability)
    # These columns are prefixed with "__" to indicate that they are metadata and not business features
    data["__source_file"] = str(raw_data_path)
    data["__ingested_at_utc"] = datetime.now(timezone.utc).isoformat()
    
    return data
import logging
import os
from typing import List, Optional
import pandas as pd
from datetime import datetime

from common.constants.aggreg_operations import AggregOpeNames
from common.constants.datatypes import DATATYPE_NAMES
from common.long_term_uc_io import COLUMN_NAMES, DATE_FORMAT, FILES_FORMAT, HYDRO_VALUE_COLUMNS, HYDRO_FILES, \
    HYDRO_KEY_COLUMNS, HYDRO_DEFAULT_VALUES
from utils.basic_utils import str_sanitizer, robust_cast_str_to_float
from utils.dates import set_date_from_year_and_iso_idx, set_date_from_year_and_day_idx
from utils.df_utils import cast_df_col_as_date, concatenate_dfs, selec_in_df_based_on_list, \
    get_subdf_from_date_range, replace_none_values_in_df


def filter_input_data(df: pd.DataFrame, date_col: str, climatic_year_col: str, period_start: datetime, 
                      period_end: datetime, climatic_year: int) -> pd.DataFrame:
    # If ERAA date format not automatically cast by pd
    first_date = df[date_col].iloc[0]
    if not isinstance(first_date, datetime):
        df = cast_df_col_as_date(df=df, date_col=date_col, date_format=DATE_FORMAT)
    # keep only wanted date range
    df_filtered = get_subdf_from_date_range(df=df, date_col=date_col, date_min=period_start, date_max=period_end)
    # then selected climatic year
    if climatic_year_col in df_filtered.columns:
        df_filtered = selec_in_df_based_on_list(df=df_filtered, selec_col=climatic_year_col, selec_vals=[climatic_year])
    # cases with data inependent of climatic years (e.g. hydro reservoir min/max levels)
    # -> add climatic year col (+ selected value in it) to have uniform formats hereafter
    else:
        df_filtered[climatic_year_col] = climatic_year
    return df_filtered


def set_aggreg_cf_prod_types_data(df_cf_list: List[pd.DataFrame], pt_agg_col: str, date_col: str,
                                  val_col: str) -> pd.DataFrame:
    # concatenate, aggreg. over prod type of same aggreg. type and avg
    df_cf_agg = concatenate_dfs(dfs=df_cf_list)
    df_cf_agg = df_cf_agg.groupby([pt_agg_col, date_col]).agg({val_col: AggregOpeNames.mean}).reset_index()
    return df_cf_agg


def gen_capa_pt_str_sanitizer(gen_capa_prod_type: str) -> str:
    # very ad-hoc operation
    sanitized_gen_capa_pt = gen_capa_prod_type.replace(' - ', ' ')
    sanitized_gen_capa_pt = str_sanitizer(raw_str=sanitized_gen_capa_pt, 
                                          ad_hoc_replacements={'gas_': 'gas', '(': '', ')': ''})
    return sanitized_gen_capa_pt


def select_interco_capas(df_intercos_capa: pd.DataFrame, countries: List[str]) -> pd.DataFrame:
    selection_col = 'selected'
    # add selection column
    origin_col = COLUMN_NAMES.zone_origin
    destination_col = COLUMN_NAMES.zone_destination
    df_intercos_capa[selection_col] = \
        df_intercos_capa.apply(lambda col: 1 if (col[origin_col] in countries 
                                                 and col[destination_col] in countries) else 0, axis=1)
    # keep only lines with both origin and destination zones in the list of available countries
    df_intercos_capa = df_intercos_capa[df_intercos_capa[selection_col] == 1]
    # remove selection column
    all_cols = list(df_intercos_capa.columns)
    all_cols.remove(selection_col)
    df_intercos_capa = df_intercos_capa[all_cols]
    return df_intercos_capa


def read_and_process_hydro_data(hydro_dt: str, folder: str, rm_week_and_day_cols: bool = True) \
        -> Optional[pd.DataFrame]:
    """
    Read and process hydro data files -> that share some common structure (in particular with only week - and day - idx
    values, i.o. dates)
    Returns: df with read data
    """
    hydro_file = f'{folder}/{HYDRO_FILES[hydro_dt]}'
    if not os.path.exists(hydro_file):
        logging.warning(f'{hydro_dt.capitalize()} data file does not exist: not accounted for here')
        return None

    df_hydro = pd.read_csv(hydro_file, sep=FILES_FORMAT.column_sep, decimal=FILES_FORMAT.decimal_sep)
    # robust cast to numeric values -> got some pbs with data... TODO: fix this more properly
    value_cols = HYDRO_VALUE_COLUMNS[hydro_dt]
    for col in value_cols:
        df_hydro[col] = df_hydro[col].apply(robust_cast_str_to_float)
    # replace none values by default ones
    df_hydro = replace_none_values_in_df(df=df_hydro, per_col_repl_values=HYDRO_DEFAULT_VALUES[hydro_dt],
                                         key_cols=HYDRO_KEY_COLUMNS[hydro_dt])
    # specific treatment for hydro weekly/daily data -> set date column based on week(/and day) values
    df_cols = list(df_hydro.columns)
    week_col = COLUMN_NAMES.week
    day_col = COLUMN_NAMES.day
    if day_col not in df_cols:  # set date from week index only
        # add day column with 1 for all (i.e. Monday)
        df_hydro[day_col] = 1
        df_cols.append(day_col)
        # remove rows with invalid week idx (> 52)
        init_len = len(df_hydro)
        df_hydro = df_hydro[df_hydro[week_col] < 53]
        new_len = len(df_hydro)
        if new_len < init_len:
            logging.warning(f'{init_len - new_len} rows suppressed in {hydro_dt} data due to invalid week idx (> 52)')
        # set date column based on week and day=1 index values
        df_hydro[COLUMN_NAMES.date] = (
            df_hydro.apply(lambda row:
                           set_date_from_year_and_iso_idx(year=1900, week_idx=row[week_col], day_idx=row[day_col]),
                           axis=1)
        )
    else:  # only from day index from 1 to 365
        df_hydro[COLUMN_NAMES.date] = (df_hydro[day_col]
                                       .apply(lambda x: set_date_from_year_and_day_idx(year=1900, day_idx=x))
                                       )
    if rm_week_and_day_cols:
        cols_tb_rmed = [week_col]
        if day_col in df_cols:
            cols_tb_rmed.append(day_col)
        df_hydro.drop(columns=cols_tb_rmed, inplace=True)
    return df_hydro


def get_hydro_ror_generation(folder: str, zone: str, climatic_year: int,
                              period_start: datetime, period_end: datetime) -> Optional[pd.DataFrame]:
    """
    Load daily run-of-river hydro generation profile and convert to hourly time series.
    
    Args:
        folder: Path to ERAA hydro data folder
        zone: Zone name (e.g., 'germany', 'france')
        climatic_year: Climatic year to use
        period_start: Start date of simulation
        period_end: End date of simulation
    
    Returns:
        DataFrame with columns ['date', 'value'] where value is hourly ROR generation in MW
        Returns None if data not available for the zone
    """
    # Load daily ROR data using existing function
    from common.constants.datatypes import DATATYPE_NAMES
    df_ror = read_and_process_hydro_data(hydro_dt=DATATYPE_NAMES.hydro_ror, folder=folder, rm_week_and_day_cols=True)
    
    if df_ror is None:
        logging.warning(f'No ROR hydro data available for zone {zone}')
        return None
    
    # Filter by zone and climatic year
    df_ror = df_ror[df_ror[COLUMN_NAMES.zone] == zone]
    if COLUMN_NAMES.climatic_year in df_ror.columns:
        df_ror = df_ror[df_ror[COLUMN_NAMES.climatic_year] == climatic_year]
    
    if df_ror.empty:
        logging.warning(f'No ROR hydro data available for zone {zone}, climatic year {climatic_year}')
        return None
    
    # Filter by date range
    df_ror = filter_input_data(df=df_ror, date_col=COLUMN_NAMES.date, 
                               climatic_year_col=COLUMN_NAMES.climatic_year,
                               period_start=period_start, period_end=period_end, 
                               climatic_year=climatic_year)
    
    # Expand daily values to hourly (replicate each daily value 24 times)
    hourly_data = []
    for _, row in df_ror.iterrows():
        daily_date = row[COLUMN_NAMES.date]
        daily_value = row['value']
        for hour in range(24):
            hourly_date = daily_date + pd.Timedelta(hours=hour)
            hourly_data.append({'date': hourly_date, 'value': daily_value})
    
    df_ror_hourly = pd.DataFrame(hourly_data)
    return df_ror_hourly


def get_hydro_inflows(folder: str, zone: str, climatic_year: int,
                      period_start: datetime, period_end: datetime) -> dict:
    """
    Load weekly hydro inflows and distribute to hourly time series.
    
    Args:
        folder: Path to ERAA hydro data folder
        zone: Zone name (e.g., 'germany', 'france')
        climatic_year: Climatic year to use
        period_start: Start date of simulation
        period_end: End date of simulation
    
    Returns:
        Dictionary with keys 'reservoir' and 'pump_open', each containing a DataFrame
        with columns ['date', 'value'] where value is hourly inflow in MWh
    """
    # Load weekly inflow data using existing function
    from common.constants.datatypes import DATATYPE_NAMES
    df_inflows = read_and_process_hydro_data(hydro_dt=DATATYPE_NAMES.hydro_inflows, folder=folder, rm_week_and_day_cols=False)
    
    if df_inflows is None:
        logging.warning(f'No hydro inflow data available')
        return {'reservoir': None, 'pump_open': None}
    
    # Filter by zone and climatic year
    df_inflows = df_inflows[df_inflows[COLUMN_NAMES.zone] == zone]
    if COLUMN_NAMES.climatic_year in df_inflows.columns:
        df_inflows = df_inflows[df_inflows[COLUMN_NAMES.climatic_year] == climatic_year]
    
    if df_inflows.empty:
        logging.warning(f'No hydro inflow data available for zone {zone}, climatic year {climatic_year}')
        return {'reservoir': None, 'pump_open': None}
    
    # Filter by date range
    df_inflows = filter_input_data(df=df_inflows, date_col=COLUMN_NAMES.date,
                                   climatic_year_col=COLUMN_NAMES.climatic_year,
                                   period_start=period_start, period_end=period_end,
                                   climatic_year=climatic_year)
    
    # Process reservoir inflows
    reservoir_inflows = []
    pump_open_inflows = []
    
    for _, row in df_inflows.iterrows():
        week_start = row[COLUMN_NAMES.date]  # Monday of the week
        reservoir_weekly = row.get('cum_inflow_into_reservoirs', 0)
        pump_open_weekly = row.get('cum_nat_inflow_into_pump-storage_reservoirs', 0)
        
        # Distribute weekly value evenly across 7 days * 24 hours = 168 hours
        hours_per_week = 168
        reservoir_hourly = reservoir_weekly / hours_per_week if reservoir_weekly else 0
        pump_open_hourly = pump_open_weekly / hours_per_week if pump_open_weekly else 0
        
        # Create hourly entries for the week
        for hour_offset in range(hours_per_week):
            hourly_date = week_start + pd.Timedelta(hours=hour_offset)
            if period_start <= hourly_date <= period_end + pd.Timedelta(days=1):
                reservoir_inflows.append({'date': hourly_date, 'value': reservoir_hourly})
                pump_open_inflows.append({'date': hourly_date, 'value': pump_open_hourly})
    
    df_reservoir = pd.DataFrame(reservoir_inflows) if reservoir_inflows else None
    df_pump_open = pd.DataFrame(pump_open_inflows) if pump_open_inflows else None
    
    return {'reservoir': df_reservoir, 'pump_open': df_pump_open}


def get_hydro_level_constraints(
    folder: str,
    zone: str,
    climatic_year: int = None,
    period_start: datetime = None,
    period_end: datetime = None
) -> dict:
    """
    Load hydro storage level constraints (min/max) from ERAA dataset.
    
    Phase 3: Load weekly min/max level constraints for reservoir and pump storage
    
    Args:
        folder: Path to ERAA data folder
        zone: Zone name (e.g., 'germany')
        climatic_year: Climatic year (default: None, ignored for level constraints)
        period_start: Start date of simulation period
        period_end: End date of simulation period
    
    Returns:
        Dictionary with keys 'reservoir_min', 'reservoir_max', 'pump_open_min', 'pump_open_max',
        'pump_closed_min', 'pump_closed_max', each containing a DataFrame
        with columns ['date', 'value'] where value is the level constraint (0-1)
    """
    # Load weekly level data using existing function
    from common.constants.datatypes import DATATYPE_NAMES
    df_levels_min = read_and_process_hydro_data(hydro_dt=DATATYPE_NAMES.hydro_levels_min, folder=folder, rm_week_and_day_cols=False)
    df_levels_max = read_and_process_hydro_data(hydro_dt=DATATYPE_NAMES.hydro_levels_max, folder=folder, rm_week_and_day_cols=False)
    
    if df_levels_min is None or df_levels_max is None:
        logging.warning(f'No hydro level constraint data available')
        return {
            'reservoir_min': None, 'reservoir_max': None,
            'pump_open_min': None, 'pump_open_max': None,
            'pump_closed_min': None, 'pump_closed_max': None
        }
    
    # Filter for the specified zone (df already filtered by zone in read_and_process_hydro_data)
    
    # Map week numbers to dates for the specified period
    if period_start is None or period_end is None:
        logging.warning('Period start and end are required for level constraints')
        return {
            'reservoir_min': None, 'reservoir_max': None,
            'pump_open_min': None, 'pump_open_max': None,
            'pump_closed_min': None, 'pump_closed_max': None
        }
    
    # Create a date range for the simulation period
    date_range = pd.date_range(start=period_start, end=period_end, freq='h')
    
    # Get week numbers for each date (ISO week)
    week_numbers = date_range.isocalendar().week.values
    
    # Create dataframes for each storage type and constraint type
    result = {}
    
    for storage_type in ['reservoir', 'pump_open', 'pump_closed']:
        # Column names in ERAA data
        if storage_type == 'reservoir':
            col_suffix = 'reservoirs'
        elif storage_type == 'pump_open':
            col_suffix = 'pump-storage_reservoirs_with_natural_inflow'
        else:  # pump_closed
            col_suffix = 'pump-storage_reservoirs'
        
        for constraint_type, df_source in [('min', df_levels_min), ('max', df_levels_max)]:
            col_name = f'{constraint_type}_{col_suffix}'
            
            if col_name not in df_source.columns:
                logging.warning(f'Column {col_name} not found in level data')
                result[f'{storage_type}_{constraint_type}'] = None
                continue
            
            # Create a mapping from week to constraint value
            week_to_value = df_source.set_index('week')[col_name].to_dict()
            
            # Map constraint values to hourly resolution
            constraint_values = [week_to_value.get(week, 0.0 if constraint_type == 'min' else 1.0) 
                                for week in week_numbers]
            
            # Create dataframe
            df_constraint = pd.DataFrame({
                'date': date_range,
                'value': constraint_values
            })
            
            result[f'{storage_type}_{constraint_type}'] = df_constraint
    
    return result
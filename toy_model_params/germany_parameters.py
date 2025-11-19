from typing import Dict, List, Union

import pandas as pd
import numpy as np

from common.constants.pypsa_params import GEN_UNITS_PYPSA_PARAMS
from common.fuel_sources import FuelSource, FuelNames, DummyFuelNames
from include.dataset_builder import GenerationUnitData

GENERATOR_DICT_TYPE = Dict[str, Union[float, int, str]]
gps_coords = (12.5674, 41.8719)


def get_generators(country_trigram: str, fuel_sources: Dict[str, FuelSource], wind_on_shore_cf_data: pd.DataFrame,
                   wind_off_shore_cf_data: pd.DataFrame, solar_pv_cf_data: pd.DataFrame) -> List[GENERATOR_DICT_TYPE]:
    """
    Get list of generators to be set on a given node of a PyPSA model
    :param country_trigram: name of considered country, as a trigram (ex: "ben", "fra", etc.)
    :param fuel_sources
    :param wind_on_shore_cf_data
    :param wind_off_shore_cf_data
    :param solar_pv_cf_data
    N.B.
    (i) Better in this function to use CONSTANT names of the different fuel sources to avoid trouble
    in the code (i.e. GEN_UNITS_PYPSA_PARAMS, FuelNames and DummyFuelNames dataclasses = sort of dict.). If you prefer
    to directly use str you can Ctrl+click on the constants below and see the corresponding str (e.g.,
    'name' for GEN_UNITS_PYPSA_PARAMS.name)
    (ii) When default PyPSA values have to be used for the generator parameters they are not provided below -> e.g.,
    efficiency=1, committable=False (i.e., not switch on/off integer variables in the model),
    min_power_pu/max_power_pu=0/1, marginal_cost=0
    -> see field 'generator_params_default_vals' in file input/long_term_uc/pypsa_static_params.json
    """
    n_ts = len(wind_on_shore_cf_data['value'].values)
    generators = [
        # Batteries
        {
            GEN_UNITS_PYPSA_PARAMS.name: f'{country_trigram}_battery',
            GEN_UNITS_PYPSA_PARAMS.carrier: 'battery',
            GEN_UNITS_PYPSA_PARAMS.nominal_power: 2188.91,
            GEN_UNITS_PYPSA_PARAMS.max_hours: 2,  # 4377.82 / 2188.91
            GEN_UNITS_PYPSA_PARAMS.soc_init: 1000,
            GEN_UNITS_PYPSA_PARAMS.efficiency_store: 0.95,
            GEN_UNITS_PYPSA_PARAMS.efficiency_dispatch: 0.95
        },
        # Biofuel
        {
            GEN_UNITS_PYPSA_PARAMS.name: f'{country_trigram}_biofuel',
            GEN_UNITS_PYPSA_PARAMS.carrier: 'biofuel',
            GEN_UNITS_PYPSA_PARAMS.nominal_power: 0.0,
        },
        # Demand Side Response
        {
            GEN_UNITS_PYPSA_PARAMS.name: f'{country_trigram}_dsr',
            GEN_UNITS_PYPSA_PARAMS.carrier: 'dsr',
            GEN_UNITS_PYPSA_PARAMS.nominal_power: 945.56,
        },
        # Gas
        {
            GEN_UNITS_PYPSA_PARAMS.name: f'{country_trigram}_gas',
            GEN_UNITS_PYPSA_PARAMS.carrier: 'gas',
            GEN_UNITS_PYPSA_PARAMS.nominal_power: 32541.3,
            GEN_UNITS_PYPSA_PARAMS.efficiency: 0.5
        },
        # Hard Coal
        {
            GEN_UNITS_PYPSA_PARAMS.name: f'{country_trigram}_hard-coal',
            GEN_UNITS_PYPSA_PARAMS.carrier: 'hard-coal',
            GEN_UNITS_PYPSA_PARAMS.nominal_power: 12786.24,
            GEN_UNITS_PYPSA_PARAMS.efficiency: 0.37
        },
        # Hydro - Pondage
        {
            GEN_UNITS_PYPSA_PARAMS.name: f'{country_trigram}_hydro-pondage',
            GEN_UNITS_PYPSA_PARAMS.carrier: 'hydro-pondage',
            GEN_UNITS_PYPSA_PARAMS.nominal_power: 0.0,
        },
        # Hydro - Pump Storage Closed Loop
        {
            GEN_UNITS_PYPSA_PARAMS.name: f'{country_trigram}_hydro-pump-storage-closed-loop',
            GEN_UNITS_PYPSA_PARAMS.carrier: 'hydro-pump-storage-closed-loop',
            GEN_UNITS_PYPSA_PARAMS.nominal_power: 6409.84,
            GEN_UNITS_PYPSA_PARAMS.max_hours: 61.12,  # 391579.78 / 6409.84
        },
        # Hydro - Pump Storage Open Loop
        {
            GEN_UNITS_PYPSA_PARAMS.name: f'{country_trigram}_hydro-pump-storage-open-loop',
            GEN_UNITS_PYPSA_PARAMS.carrier: 'hydro-pump-storage-open-loop',
            GEN_UNITS_PYPSA_PARAMS.nominal_power: 2144.1,
            GEN_UNITS_PYPSA_PARAMS.max_hours: 219.81,  # 471229.0 / 2144.1
        },
        # Hydro - Reservoir
        {
            GEN_UNITS_PYPSA_PARAMS.name: f'{country_trigram}_hydro-reservoir',
            GEN_UNITS_PYPSA_PARAMS.carrier: 'hydro-reservoir',
            GEN_UNITS_PYPSA_PARAMS.nominal_power: 819.0,
            GEN_UNITS_PYPSA_PARAMS.max_hours: 289.56,  # 237217.03 / 819.0
        },
        # Hydro - Run of River
        {
            GEN_UNITS_PYPSA_PARAMS.name: f'{country_trigram}_hydro-run-of-river',
            GEN_UNITS_PYPSA_PARAMS.carrier: 'hydro-run-of-river',
            GEN_UNITS_PYPSA_PARAMS.nominal_power: 3933.9,
        },
        # Lignite
        {
            GEN_UNITS_PYPSA_PARAMS.name: f'{country_trigram}_lignite',
            GEN_UNITS_PYPSA_PARAMS.carrier: 'lignite',
            GEN_UNITS_PYPSA_PARAMS.nominal_power: 14687.0,
        },
        # Nuclear
        {
            GEN_UNITS_PYPSA_PARAMS.name: f'{country_trigram}_nuclear',
            GEN_UNITS_PYPSA_PARAMS.carrier: 'nuclear',
            GEN_UNITS_PYPSA_PARAMS.nominal_power: 0.0,
        },
        # Oil
        {
            GEN_UNITS_PYPSA_PARAMS.name: f'{country_trigram}_oil',
            GEN_UNITS_PYPSA_PARAMS.carrier: 'oil',
            GEN_UNITS_PYPSA_PARAMS.nominal_power: 2821.33,
            GEN_UNITS_PYPSA_PARAMS.efficiency: 0.4
        },
        # Others non-renewable
        {
            GEN_UNITS_PYPSA_PARAMS.name: f'{country_trigram}_other-non-renewables',
            GEN_UNITS_PYPSA_PARAMS.carrier: 'other-non-renewables',
            GEN_UNITS_PYPSA_PARAMS.nominal_power: 9080.02,
        },
        # Others renewable
        {
            GEN_UNITS_PYPSA_PARAMS.name: f'{country_trigram}_other-renewables',
            GEN_UNITS_PYPSA_PARAMS.carrier: 'other-renewables',
            GEN_UNITS_PYPSA_PARAMS.nominal_power: 11056.79,
        },
        # Solar PV
        {
            GEN_UNITS_PYPSA_PARAMS.name: f'{country_trigram}_solar-pv',
            GEN_UNITS_PYPSA_PARAMS.carrier: 'solar-pv',
            GEN_UNITS_PYPSA_PARAMS.nominal_power: 88447.85,
            GEN_UNITS_PYPSA_PARAMS.max_power_pu: solar_pv_cf_data['value'].values,
        },
        # Solar Thermal
        {
            GEN_UNITS_PYPSA_PARAMS.name: f'{country_trigram}_solar-thermal',
            GEN_UNITS_PYPSA_PARAMS.carrier: 'solar-thermal',
            GEN_UNITS_PYPSA_PARAMS.nominal_power: 0.0,
        },
        # Wind Offshore
        {
            GEN_UNITS_PYPSA_PARAMS.name: f'{country_trigram}_wind-off-shore',
            GEN_UNITS_PYPSA_PARAMS.carrier: 'wind-off-shore',
            GEN_UNITS_PYPSA_PARAMS.nominal_power: 11105.0,
            GEN_UNITS_PYPSA_PARAMS.max_power_pu: wind_off_shore_cf_data['value'].values,
        },
        # Wind Onshore
        {
            GEN_UNITS_PYPSA_PARAMS.name: f'{country_trigram}_wind-on-shore',
            GEN_UNITS_PYPSA_PARAMS.carrier: 'wind-on-shore',
            GEN_UNITS_PYPSA_PARAMS.nominal_power: 69017.4,
            GEN_UNITS_PYPSA_PARAMS.max_power_pu: wind_on_shore_cf_data['value'].values,
        },
        # Fictive failure asset
        {
            GEN_UNITS_PYPSA_PARAMS.name: f'{country_trigram}_failure',
            GEN_UNITS_PYPSA_PARAMS.carrier: DummyFuelNames.ac,
            GEN_UNITS_PYPSA_PARAMS.nominal_power: 1e10,
            GEN_UNITS_PYPSA_PARAMS.marginal_cost: 1e5
        },
    ]
    return generators


def set_gen_as_list_of_gen_units_data(generators: List[GENERATOR_DICT_TYPE]) -> List[GenerationUnitData]:
    # add type of units
    for elt_gen in generators:
        elt_gen['type'] = f'{elt_gen["carrier"]}_agg'
    # then cas as list of GenerationUnitData objects
    return [GenerationUnitData(**elt_gen_dict) for elt_gen_dict in generators]

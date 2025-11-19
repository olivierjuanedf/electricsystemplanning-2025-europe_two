from typing import Dict, List, Union

import pandas as pd

from common.constants.pypsa_params import GEN_UNITS_PYPSA_PARAMS
from common.fuel_sources import FuelSource, FuelNames, DummyFuelNames
from include.dataset_builder import GenerationUnitData

GENERATOR_DICT_TYPE = Dict[str, Union[float, int, str]]

# GPS coordinates for Germany (Berlin)
gps_coords = (13.4050, 52.5200)

# Generation capacities from ERAA 2023-2 dataset for Germany 2033
# Source: data/ERAA_2023-2/generation_capas/generation-capa_2033_germany.csv
# NOTE: Major energy transition - coal/lignite phased out, massive renewable expansion
HARD_COAL_CAPACITY = 0.0  # MW - Phased out by 2033
LIGNITE_CAPACITY = 0.0  # MW - Phased out by 2033
GAS_CAPACITY = 35084.62  # MW
OIL_CAPACITY = 1044.83  # MW
OTHER_NON_RENEWABLE_CAPACITY = 5167.52  # MW
WIND_ONSHORE_CAPACITY = 133515.93  # MW - Nearly doubled from 2025!
WIND_OFFSHORE_CAPACITY = 42185.0  # MW - 4x increase from 2025!
SOLAR_PV_CAPACITY = 270886.85  # MW - 3x increase from 2025!
OTHER_RENEWABLE_CAPACITY = 10355.84  # MW

# Storage capacities from ERAA 2023-2 dataset for Germany 2033
# Source: data/ERAA_2023-2/generation_capas/generation-capa_2033_germany.csv
#         data/ERAA_2023-2/hydro/PECD-hydro-capacities.csv
# NOTE: Battery capacity increased ~5x from 2025! Storage becomes critical with high renewable penetration
BATTERY_POWER = 11985.95  # MW (injection/offtake capacity - ~5x larger than 2025!)
BATTERY_ENERGY = 23971.9  # MWh (~5x larger than 2025!)
PUMP_CLOSED_POWER_TURBINE = 7009.84  # MW (increased from 2025)
PUMP_CLOSED_POWER_PUMP = 7166.6  # MW (absolute value, will be used as negative for pumping)
PUMP_CLOSED_ENERGY = 391579.78  # MWh (391.6 GWh - increased from 2025)
PUMP_OPEN_POWER_TURBINE = 2144.1  # MW (increased from 2025)
PUMP_OPEN_POWER_PUMP = 1861.0  # MW (absolute value, will be used as negative for pumping)
PUMP_OPEN_ENERGY = 471229.0  # MWh (471.2 GWh - increased from 2025)
RESERVOIR_POWER = 819.0  # MW (decreased from 2025)
RESERVOIR_ENERGY = 237217.03  # MWh (237.2 GWh - slightly decreased from 2025)
ROR_POWER = 3933.9  # MW (decreased from 2025)


def get_generators(country_trigram: str, fuel_sources: Dict[str, FuelSource], wind_on_shore_cf_data: pd.DataFrame,
                   wind_off_shore_cf_data: pd.DataFrame, solar_pv_cf_data: pd.DataFrame,
                   hydro_ror_profile: pd.DataFrame = None,
                   hydro_inflows: dict = None,
                   hydro_level_constraints: dict = None) -> List[GENERATOR_DICT_TYPE]:
    """
    Get list of generators to be set on a given node of a PyPSA model for Germany 2033
    :param country_trigram: name of considered country, as a trigram (ex: "ger")
    :param fuel_sources: dictionary of fuel source parameters
    :param wind_on_shore_cf_data: wind onshore capacity factor data
    :param wind_off_shore_cf_data: wind offshore capacity factor data
    :param solar_pv_cf_data: solar PV capacity factor data
    :param hydro_ror_profile: run-of-river generation profile (optional)
    :param hydro_inflows: dictionary with inflow profiles for reservoir and pump_open (optional)
    :param hydro_level_constraints: dictionary with min/max level constraints (optional)
    
    N.B.
    (i) Better in this function to use CONSTANT names of the different fuel sources to avoid trouble
    in the code (i.e. GEN_UNITS_PYPSA_PARAMS, FuelNames and DummyFuelNames dataclasses = sort of dict.). If you prefer
    to directly use str you can Ctrl+click on the constants below and see the corresponding str (e.g.,
    'name' for GEN_UNITS_PYPSA_PARAMS.name)
    (ii) When default PyPSA values have to be used for the generator parameters they are not provided below -> e.g.,
    efficiency=1, committable=False (i.e., not switch on/off integer variables in the model),
    min_power_pu/max_power_pu=0/1, marginal_cost=0
    -> see field 'generator_params_default_vals' in file input/long_term_uc/pypsa_static_params.json
    (iii) All capacity values are extracted from ERAA 2023-2 dataset (generation-capa_2033_germany.csv)
    (iv) Marginal cost = primary_cost / efficiency (NOT multiply!)
        - Lower efficiency means you need MORE fuel per MWh of electricity
        - Therefore marginal cost is HIGHER for less efficient plants
    (v) 2033 shows Germany's energy transition: coal/lignite phased out, massive renewable expansion
    (vi) Battery capacity is ~5x larger in 2033 vs 2025, reflecting the critical role of storage in high-renewable systems
    """
    generators = [
        # Note: Hard coal and lignite are NOT included - phased out by 2033
        {
            GEN_UNITS_PYPSA_PARAMS.name: f'{country_trigram}_gas',
            GEN_UNITS_PYPSA_PARAMS.carrier: FuelNames.gas,
            GEN_UNITS_PYPSA_PARAMS.nominal_power: GAS_CAPACITY,
            GEN_UNITS_PYPSA_PARAMS.marginal_cost: fuel_sources[FuelNames.gas].primary_cost / 0.5,
            GEN_UNITS_PYPSA_PARAMS.efficiency: 0.5
        },
        {
            GEN_UNITS_PYPSA_PARAMS.name: f'{country_trigram}_oil',
            GEN_UNITS_PYPSA_PARAMS.carrier: FuelNames.oil,
            GEN_UNITS_PYPSA_PARAMS.nominal_power: OIL_CAPACITY,
            GEN_UNITS_PYPSA_PARAMS.marginal_cost: fuel_sources[FuelNames.oil].primary_cost / 0.4,
            GEN_UNITS_PYPSA_PARAMS.efficiency: 0.4
        },
        {
            GEN_UNITS_PYPSA_PARAMS.name: f'{country_trigram}_other_non_renewables',
            GEN_UNITS_PYPSA_PARAMS.carrier: FuelNames.other_non_renewables,
            GEN_UNITS_PYPSA_PARAMS.nominal_power: OTHER_NON_RENEWABLE_CAPACITY,
            GEN_UNITS_PYPSA_PARAMS.marginal_cost: fuel_sources[FuelNames.other_non_renewables].primary_cost / 0.4,
            GEN_UNITS_PYPSA_PARAMS.efficiency: 0.4
        },
        {
            GEN_UNITS_PYPSA_PARAMS.name: f'{country_trigram}_wind_onshore',
            GEN_UNITS_PYPSA_PARAMS.carrier: FuelNames.wind,
            GEN_UNITS_PYPSA_PARAMS.nominal_power: WIND_ONSHORE_CAPACITY,
            GEN_UNITS_PYPSA_PARAMS.max_power_pu: wind_on_shore_cf_data['value'].values,
            GEN_UNITS_PYPSA_PARAMS.marginal_cost: fuel_sources[FuelNames.wind].primary_cost
        },
        {
            GEN_UNITS_PYPSA_PARAMS.name: f'{country_trigram}_wind_offshore',
            GEN_UNITS_PYPSA_PARAMS.carrier: FuelNames.wind,
            GEN_UNITS_PYPSA_PARAMS.nominal_power: WIND_OFFSHORE_CAPACITY,
            GEN_UNITS_PYPSA_PARAMS.max_power_pu: wind_off_shore_cf_data['value'].values,
            GEN_UNITS_PYPSA_PARAMS.marginal_cost: fuel_sources[FuelNames.wind].primary_cost
        },
        {
            GEN_UNITS_PYPSA_PARAMS.name: f'{country_trigram}_solar_pv',
            GEN_UNITS_PYPSA_PARAMS.carrier: FuelNames.solar,
            GEN_UNITS_PYPSA_PARAMS.nominal_power: SOLAR_PV_CAPACITY,
            GEN_UNITS_PYPSA_PARAMS.max_power_pu: solar_pv_cf_data['value'].values,
            GEN_UNITS_PYPSA_PARAMS.marginal_cost: fuel_sources[FuelNames.solar].primary_cost
        },
        {
            GEN_UNITS_PYPSA_PARAMS.name: f'{country_trigram}_other_renewables',
            GEN_UNITS_PYPSA_PARAMS.carrier: FuelNames.other_renewables,
            GEN_UNITS_PYPSA_PARAMS.nominal_power: OTHER_RENEWABLE_CAPACITY,
        },
        # Storage units - integrated with time-series data and constraints
        # Battery storage with ~85% round-trip efficiency - MUCH LARGER in 2033!
        {
            GEN_UNITS_PYPSA_PARAMS.name: f'{country_trigram}_battery',
            GEN_UNITS_PYPSA_PARAMS.carrier: FuelNames.battery,
            GEN_UNITS_PYPSA_PARAMS.nominal_power: BATTERY_POWER,
            GEN_UNITS_PYPSA_PARAMS.max_hours: BATTERY_ENERGY / BATTERY_POWER,
            GEN_UNITS_PYPSA_PARAMS.efficiency_store: 0.92,
            GEN_UNITS_PYPSA_PARAMS.efficiency_dispatch: 0.92,
            GEN_UNITS_PYPSA_PARAMS.cyclic_state_of_charge: True,
            GEN_UNITS_PYPSA_PARAMS.marginal_cost: 0.0
        },
        # Pump storage closed loop (no natural inflows, with constraints)
        {
            GEN_UNITS_PYPSA_PARAMS.name: f'{country_trigram}_pump_closed',
            GEN_UNITS_PYPSA_PARAMS.carrier: FuelNames.hydro_pump_closed,
            GEN_UNITS_PYPSA_PARAMS.nominal_power: PUMP_CLOSED_POWER_TURBINE,
            GEN_UNITS_PYPSA_PARAMS.max_hours: PUMP_CLOSED_ENERGY / PUMP_CLOSED_POWER_TURBINE,
            GEN_UNITS_PYPSA_PARAMS.efficiency_store: 0.9,
            GEN_UNITS_PYPSA_PARAMS.efficiency_dispatch: 0.9,
            GEN_UNITS_PYPSA_PARAMS.cyclic_state_of_charge: True,
            GEN_UNITS_PYPSA_PARAMS.marginal_cost: 0.0,
            # Add min/max level constraints (as fraction of max energy capacity)
            **({'state_of_charge_set': hydro_level_constraints['pump_closed_min']['value'].values * PUMP_CLOSED_ENERGY} 
               if hydro_level_constraints and hydro_level_constraints.get('pump_closed_min') is not None else {})
        },
        # Pump storage open loop (with natural inflows and constraints)
        {
            GEN_UNITS_PYPSA_PARAMS.name: f'{country_trigram}_pump_open',
            GEN_UNITS_PYPSA_PARAMS.carrier: FuelNames.hydro_pump_open,
            GEN_UNITS_PYPSA_PARAMS.nominal_power: PUMP_OPEN_POWER_TURBINE,
            GEN_UNITS_PYPSA_PARAMS.max_hours: PUMP_OPEN_ENERGY / PUMP_OPEN_POWER_TURBINE,
            GEN_UNITS_PYPSA_PARAMS.efficiency_store: 0.9,
            GEN_UNITS_PYPSA_PARAMS.efficiency_dispatch: 0.9,
            GEN_UNITS_PYPSA_PARAMS.cyclic_state_of_charge: False,  # Open loop has natural inflows
            GEN_UNITS_PYPSA_PARAMS.marginal_cost: 0.0,
            **({'inflow': hydro_inflows['pump_open']['value'].values} if hydro_inflows and hydro_inflows.get('pump_open') is not None else {}),
            # Add min/max level constraints (as fraction of max energy capacity)
            **({'state_of_charge_set': hydro_level_constraints['pump_open_min']['value'].values * PUMP_OPEN_ENERGY} 
               if hydro_level_constraints and hydro_level_constraints.get('pump_open_min') is not None else {})
        },
        # Reservoir hydro (with inflows and constraints)
        {
            GEN_UNITS_PYPSA_PARAMS.name: f'{country_trigram}_reservoir',
            GEN_UNITS_PYPSA_PARAMS.carrier: FuelNames.hydro_reservoir,
            GEN_UNITS_PYPSA_PARAMS.nominal_power: RESERVOIR_POWER,
            GEN_UNITS_PYPSA_PARAMS.max_hours: RESERVOIR_ENERGY / RESERVOIR_POWER,
            GEN_UNITS_PYPSA_PARAMS.efficiency_dispatch: 0.9,
            GEN_UNITS_PYPSA_PARAMS.cyclic_state_of_charge: False,
            GEN_UNITS_PYPSA_PARAMS.marginal_cost: 0.0,
            **({'inflow': hydro_inflows['reservoir']['value'].values} if hydro_inflows and hydro_inflows.get('reservoir') is not None else {}),
            # Add min/max level constraints (as fraction of max energy capacity)
            **({'state_of_charge_set': hydro_level_constraints['reservoir_min']['value'].values * RESERVOIR_ENERGY} 
               if hydro_level_constraints and hydro_level_constraints.get('reservoir_min') is not None else {})
        },
        # Run-of-river hydro (with generation profile)
        {
            GEN_UNITS_PYPSA_PARAMS.name: f'{country_trigram}_ror',
            GEN_UNITS_PYPSA_PARAMS.carrier: FuelNames.hydro_ror,
            GEN_UNITS_PYPSA_PARAMS.nominal_power: ROR_POWER,
            GEN_UNITS_PYPSA_PARAMS.max_power_pu: (hydro_ror_profile['value'].values / ROR_POWER) if hydro_ror_profile is not None else 1.0,
            GEN_UNITS_PYPSA_PARAMS.marginal_cost: 0.0
        },
        # Fictive failure asset - ensures model feasibility by providing very expensive backup capacity
        {
            GEN_UNITS_PYPSA_PARAMS.name: f'{country_trigram}_failure',
            GEN_UNITS_PYPSA_PARAMS.carrier: DummyFuelNames.ac,
            GEN_UNITS_PYPSA_PARAMS.nominal_power: 1e10,
            GEN_UNITS_PYPSA_PARAMS.marginal_cost: 1e5
        }
    ]
    return generators


def set_gen_as_list_of_gen_units_data(generators: List[GENERATOR_DICT_TYPE]) -> List[GenerationUnitData]:
    """Convert generator dictionaries to GenerationUnitData objects"""
    # add type of units
    for elt_gen in generators:
        elt_gen['type'] = f'{elt_gen["carrier"]}_agg'
    # then cast as list of GenerationUnitData objects
    return [GenerationUnitData(**elt_gen_dict) for elt_gen_dict in generators]


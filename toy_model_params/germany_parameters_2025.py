from typing import Dict, List, Union

import pandas as pd

from common.constants.pypsa_params import GEN_UNITS_PYPSA_PARAMS
from common.fuel_sources import FuelSource, FuelNames, DummyFuelNames
from include.dataset_builder import GenerationUnitData

GENERATOR_DICT_TYPE = Dict[str, Union[float, int, str]]

# GPS coordinates for Germany (Berlin)
gps_coords = (13.4050, 52.5200)

# Generation capacities from ERAA 2023-2 dataset for Germany 2025
# Source: data/ERAA_2023-2/generation_capas/generation-capa_2025_germany.csv
HARD_COAL_CAPACITY = 12786.24  # MW
LIGNITE_CAPACITY = 14687.0  # MW
GAS_CAPACITY = 32541.3  # MW
OIL_CAPACITY = 2821.33  # MW
OTHER_NON_RENEWABLE_CAPACITY = 9080.02  # MW
WIND_ONSHORE_CAPACITY = 69017.4  # MW
WIND_OFFSHORE_CAPACITY = 11105.0  # MW
SOLAR_PV_CAPACITY = 88447.85  # MW
OTHER_RENEWABLE_CAPACITY = 11056.79  # MW

# Storage capacities from ERAA 2023-2 dataset for Germany 2025
# Source: data/ERAA_2023-2/generation_capas/generation-capa_2025_germany.csv
#         data/ERAA_2023-2/hydro/PECD-hydro-capacities.csv
BATTERY_POWER = 2565.0  # MW
BATTERY_ENERGY = 8195.0  # MWh
PUMP_CLOSED_POWER_TURBINE = 6056.0  # MW
PUMP_CLOSED_POWER_PUMP = 6068.0  # MW (absolute value, will be used as negative for pumping)
PUMP_CLOSED_ENERGY = 355000.0  # MWh (355 GWh)
PUMP_OPEN_POWER_TURBINE = 1644.0  # MW
PUMP_OPEN_POWER_PUMP = 1361.0  # MW (absolute value, will be used as negative for pumping)
PUMP_OPEN_ENERGY = 417000.0  # MWh (417 GWh)
RESERVOIR_POWER = 1297.0  # MW
RESERVOIR_ENERGY = 258000.0  # MWh (258 GWh)
ROR_POWER = 5078.0  # MW


def get_generators(country_trigram: str, fuel_sources: Dict[str, FuelSource], wind_on_shore_cf_data: pd.DataFrame,
                   wind_off_shore_cf_data: pd.DataFrame, solar_pv_cf_data: pd.DataFrame,
                   hydro_ror_profile: pd.DataFrame = None, hydro_inflows: dict = None,
                   hydro_level_constraints: dict = None) -> List[GENERATOR_DICT_TYPE]:
    """
    Get list of generators to be set on a given node of a PyPSA model for Germany
    :param country_trigram: name of considered country, as a trigram (ex: "ger")
    :param fuel_sources: dictionary of fuel source parameters
    :param wind_on_shore_cf_data: wind onshore capacity factor data
    :param wind_off_shore_cf_data: wind offshore capacity factor data
    :param solar_pv_cf_data: solar PV capacity factor data
    :param hydro_ror_profile: run-of-river hydro generation profile (hourly, MW)
    :param hydro_inflows: dictionary with 'reservoir' and 'pump_open' inflow time series (hourly, MWh)
    :param hydro_level_constraints: dictionary with min/max level constraints for storage units (0-1)
    
    N.B.
    (i) Better in this function to use CONSTANT names of the different fuel sources to avoid trouble
    in the code (i.e. GEN_UNITS_PYPSA_PARAMS, FuelNames and DummyFuelNames dataclasses = sort of dict.). If you prefer
    to directly use str you can Ctrl+click on the constants below and see the corresponding str (e.g.,
    'name' for GEN_UNITS_PYPSA_PARAMS.name)
    (ii) When default PyPSA values have to be used for the generator parameters they are not provided below -> e.g.,
    efficiency=1, committable=False (i.e., not switch on/off integer variables in the model),
    min_power_pu/max_power_pu=0/1, marginal_cost=0
    -> see field 'generator_params_default_vals' in file input/long_term_uc/pypsa_static_params.json
    (iii) All capacity values are extracted from ERAA 2023-2 dataset (generation-capa_2025_germany.csv)
    (iv) Marginal cost = primary_cost / efficiency (NOT multiply!)
        - Lower efficiency means you need MORE fuel per MWh of electricity
        - Therefore marginal cost is HIGHER for less efficient plants
    """
    generators = [
        {
            GEN_UNITS_PYPSA_PARAMS.name: f'{country_trigram}_hard_coal',
            GEN_UNITS_PYPSA_PARAMS.carrier: FuelNames.coal,
            GEN_UNITS_PYPSA_PARAMS.nominal_power: HARD_COAL_CAPACITY,
            GEN_UNITS_PYPSA_PARAMS.marginal_cost: fuel_sources[FuelNames.coal].primary_cost / 0.37,
            GEN_UNITS_PYPSA_PARAMS.efficiency: 0.37
        },
        {
            GEN_UNITS_PYPSA_PARAMS.name: f'{country_trigram}_lignite',
            GEN_UNITS_PYPSA_PARAMS.carrier: FuelNames.coal,
            GEN_UNITS_PYPSA_PARAMS.nominal_power: LIGNITE_CAPACITY,
            GEN_UNITS_PYPSA_PARAMS.marginal_cost: fuel_sources[FuelNames.coal].primary_cost / 0.35,
            GEN_UNITS_PYPSA_PARAMS.efficiency: 0.35
        },
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
        # Storage units - Phase 1: Basic integration with static parameters
        # Battery storage with ~85% round-trip efficiency
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
        # Pump storage closed loop (no natural inflows, with constraints - Phase 3)
        {
            GEN_UNITS_PYPSA_PARAMS.name: f'{country_trigram}_pump_closed',
            GEN_UNITS_PYPSA_PARAMS.carrier: FuelNames.hydro_pump_closed,
            GEN_UNITS_PYPSA_PARAMS.nominal_power: PUMP_CLOSED_POWER_TURBINE,
            GEN_UNITS_PYPSA_PARAMS.max_hours: PUMP_CLOSED_ENERGY / PUMP_CLOSED_POWER_TURBINE,
            GEN_UNITS_PYPSA_PARAMS.efficiency_store: 0.9,
            GEN_UNITS_PYPSA_PARAMS.efficiency_dispatch: 0.9,
            GEN_UNITS_PYPSA_PARAMS.cyclic_state_of_charge: True,
            GEN_UNITS_PYPSA_PARAMS.marginal_cost: 0.0,
            # Phase 3: Add min/max level constraints (as fraction of max energy capacity)
            **({'state_of_charge_set': hydro_level_constraints['pump_closed_min']['value'].values * PUMP_CLOSED_ENERGY} 
               if hydro_level_constraints and hydro_level_constraints.get('pump_closed_min') is not None else {})
        },
        # Pump storage open loop (with natural inflows - Phase 2, constraints - Phase 3)
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
            # Phase 3: Add min/max level constraints (as fraction of max energy capacity)
            **({'state_of_charge_set': hydro_level_constraints['pump_open_min']['value'].values * PUMP_OPEN_ENERGY} 
               if hydro_level_constraints and hydro_level_constraints.get('pump_open_min') is not None else {})
        },
        # Reservoir hydro (with inflows - Phase 2, constraints - Phase 3)
        {
            GEN_UNITS_PYPSA_PARAMS.name: f'{country_trigram}_reservoir',
            GEN_UNITS_PYPSA_PARAMS.carrier: FuelNames.hydro_reservoir,
            GEN_UNITS_PYPSA_PARAMS.nominal_power: RESERVOIR_POWER,
            GEN_UNITS_PYPSA_PARAMS.max_hours: RESERVOIR_ENERGY / RESERVOIR_POWER,
            GEN_UNITS_PYPSA_PARAMS.efficiency_dispatch: 0.9,
            GEN_UNITS_PYPSA_PARAMS.cyclic_state_of_charge: False,
            GEN_UNITS_PYPSA_PARAMS.marginal_cost: 0.0,
            **({'inflow': hydro_inflows['reservoir']['value'].values} if hydro_inflows and hydro_inflows.get('reservoir') is not None else {}),
            # Phase 3: Add min/max level constraints (as fraction of max energy capacity)
            **({'state_of_charge_set': hydro_level_constraints['reservoir_min']['value'].values * RESERVOIR_ENERGY} 
               if hydro_level_constraints and hydro_level_constraints.get('reservoir_min') is not None else {})
        },
        # Run-of-river hydro (with generation profile - Phase 2)
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


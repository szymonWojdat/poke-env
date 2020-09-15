# -*- coding: utf-8 -*-
"""This module contains constant values used in the repository.
"""

import json
import os

from typing import Any, Union
from typing import Dict


def _compute_type_chart(chart_path: str) -> Dict[str, Dict[str, float]]:
    """Returns the pokemon type_chart.

    Returns a dictionnary representing the Pokemon type chart, loaded from a json file
    in `data`. This dictionnary is computed in this file as `TYPE_CHART`.

    :return: The pokemon type chart
    :rtype: Dict[str, Dict[str, float]]
    """
    with open(chart_path) as chart:
        json_chart = json.load(chart)

    types = [str(entry["name"]).upper() for entry in json_chart]

    type_chart = {type_1: {type_2: 1.0 for type_2 in types} for type_1 in types}

    for entry in json_chart:
        type_ = entry["name"].upper()
        for immunity in entry["immunes"]:
            type_chart[type_][immunity.upper()] = 0.0
        for weakness in entry["weaknesses"]:
            type_chart[type_][weakness.upper()] = 0.5
        for strength in entry["strengths"]:
            type_chart[type_][strength.upper()] = 2.0

    return type_chart


STR_TO_ID = {}
for name, filename in [
						("abilities", "abilities.json"),
						("items", "items.json"),
						("moves", "moves.json"),
						("natures", "natures.json"),
						("species", "species.json"),
						("types", "types.json")
					  ]:
	filepath = os.path.join(
	    os.path.dirname(os.path.realpath(__file__)), "data/ids/gen7", filename
	)
	with open(filepath) as fp:
		STR_TO_ID[name] = json.load(fp)

GEN7_ABILITIES: str = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "data", "typeChart.json"
)





_TYPE_CHART_PATH: str = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "data", "typeChart.json"
)
"Path to the json file containing type informations."

POKEDEX: Dict[str, Any] = {}

with open(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "pokedex.json")
) as pokedex:
    POKEDEX = json.load(pokedex)

_missing_dex: Dict[str, Any] = {}
for key, value in POKEDEX.items():
    if "otherForms" in value:
        for other_form in value["otherForms"]:
            _missing_dex[other_form] = value

POKEDEX.update(_missing_dex)

_equivalent_forms = {"darmanitangalarzen": "darmanitanzengalar"}

POKEDEX.update({k: POKEDEX[v] for k, v in _equivalent_forms.items()})

<<<<<<< HEAD
ABILITYDEX: Dict = {}

with open(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "abilities.json")
) as abilities:
    ABILITYDEX = json.load(abilities)

ITEMS: Dict = {}

with open(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "items.json")
) as items:
    ITEMS = json.load(items)



MOVES: Dict = {}
=======
MOVES: Dict[str, Any] = {}
>>>>>>> f16d70e8b80e2c880170730d9b6ef9c61c2b6bf2

with open(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "moves.json")
) as moves:
    MOVES = json.load(moves)

TYPE_CHART: Dict[str, Dict[str, float]] = _compute_type_chart(_TYPE_CHART_PATH)

NATURES: Dict[str, Dict[str, Union[int, float]]] = {}

with open(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "natures.json")
) as natures:
    NATURES = json.load(natures)


"""
A dictionnary representing the Pokemon type chart.

Each key is a string representing a type (corresponding to `Type` names), and each value
is a dictionnary whose keys are string representation of types, and whose values are
floats.

TYPE_CHART[type_1][type_2] corresponds to the damage multiplier of an attack of type_1
on a Pokemon of type_2. This dictionnary isncomputed using the `compute_type_chart`
function.
"""

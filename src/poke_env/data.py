# -*- coding: utf-8 -*-
"""This module contains constant values used in the repository.
"""

import orjson
import os
import json

from functools import lru_cache
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
        json_chart = orjson.loads(chart.read())

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


@lru_cache(2 ** 13)  # pyre-ignore
def to_id_str(name: str) -> str:
    """Converts a full-name to its corresponding id string.
    :param name: The name to convert.
    :type name: str
    :return: The corresponding id string.
    :rtype: str
    """
    return "".join(char for char in name if char.isalnum()).lower()


STR_TO_ID = {}
for name, filename in [
						("abilities", "abilities.json"),
						("items", "items.json"),
						("moves", "moves.json"),
						("natures", "natures.json"),
						("species", "species.json"),
						("types", "types.json")
					  ]:
	print(__file__)
	filepath = os.path.join(
	    os.path.dirname(os.path.realpath(__file__)), "data/ids/gen8", filename
	)
	with open(filepath) as fp:
		STR_TO_ID[name] = json.load(fp)

ID_TO_STR = {}
for key in STR_TO_ID.keys():
	ID_TO_STR[key] = {v : k for k,v in STR_TO_ID[key].items()}


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
    POKEDEX = orjson.loads(pokedex.read())

_missing_dex: Dict[str, Any] = {}
for key, value in POKEDEX.items():
    if "cosmeticFormes" in value:
        for other_form in value["cosmeticFormes"]:
            _missing_dex[to_id_str(other_form)] = value

# Alternative pikachu gmax forms
for name, value in POKEDEX.items():
    if name.startswith("pikachu") and name not in {"pikachu", "pikachugmax"}:
        _missing_dex[name + "gmax"] = POKEDEX["pikachugmax"]

POKEDEX.update(_missing_dex)

for name, value in POKEDEX.items():
    if "baseSpecies" in value:
        value["species"] = value["baseSpecies"]
    else:
        value["baseSpecies"] = to_id_str(name)

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

with open(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "moves.json")
) as moves:
    MOVES = orjson.loads(moves.read())

TYPE_CHART: Dict[str, Dict[str, float]] = _compute_type_chart(_TYPE_CHART_PATH)

NATURES: Dict[str, Dict[str, Union[int, float]]] = {}

with open(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "natures.json")
) as natures:
    NATURES = orjson.loads(natures.read())

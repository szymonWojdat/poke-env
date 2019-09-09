# -*- coding: utf-8 -*-
"""This module defines a base class communicating with showdown servers.
"""

from abc import ABC


class PlayerNetwork(ABC):
    """
    Network interface of a player.

    Responsible for communicating with showdown servers.
    """

    def __init__(self) -> None:
        pass

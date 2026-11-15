"""Bots scriptés : aléatoire, glouton, expert, expert bruité."""
from .policies import BOTS, Bot, ExpertBot, GreedyBot, NoisyExpertBot, RandomBot, make_bot

__all__ = ["BOTS", "Bot", "ExpertBot", "GreedyBot", "NoisyExpertBot", "RandomBot", "make_bot"]

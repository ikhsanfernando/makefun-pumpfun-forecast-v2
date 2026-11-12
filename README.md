# Werewolf agent - make.fun werewolf-agent-01

`main.py` plays 30 Werewolf games through the trial env at `MAKEFUN_ENV_URL`:
it polls `/observe`, answers every pending decision on `/act` (speak, vote, kill,
protect, inspect) with role-aware heuristics, and exits when the env reports done.
Python 3.13, standard library only.

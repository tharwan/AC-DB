# AC-DB
## Agent-based, Coupled Day-ahead and Balancing market simulator

This project is a research focused simulator of electricity markets. It simulates all actors as agents of an agent based simulation including electric power consumers, producers and retailers. All agents are meant to be easily extendable in their behaviour. 

The electricity market is modeled in two stages, a day-ahead stage and a balancing stage. That is at the day ahead market trades are made based on forecasts about the next day, while the balancing market is running in real time, accounting for actual usage. Therefore all changes occurring after the day ahead phase, i.e. after day ahead prices are published will lead to balancing costs/income. 

The simulations is based on a comparably high time resolution of 1 minute. The day ahead stage also includes a DC flow model to account for network constraints.

The jupyter folder includes a notebook showing the basic usage of the model, further details can be found in the following publications:

https://arxiv.org/abs/1612.04512

see also: F. Kühnlenz and P. H. J. Nardelli, “Agent-based model for spot and balancing electricity markets,” presented at the 2017 IEEE International Conference on Communications Workshops (ICC Workshops), 2017, pp. 1123–1127.

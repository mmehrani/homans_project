# Homans Project

Here, we have tried to build a computational framework of George Homans' Social Exchange Theory.\
Homans explained social structures (like formation of groups within society) by proposing propositions concerning behaviours of individuals (in other words, his theory is an agent-based model).\
What we have done is that we made a simulation of society based on Homans' assumptions in order to investigate and see its implications.\
Method and results are available in the [paper](https://arxiv.org/pdf/2007.14953)

This project is built using Python Language.\
**Note:** The code has been tested on Windows and Linux.

## Description
`Homans.py` is the core algorithm. Every agent has some properties and they interact with each other by doing transactions. Initially, agents doesn't know anybody, so they explore for new neighbors. The dynamic of exploration and transaction goes on till a society is created.\
Directory named `runned_files` is created and raw data of agents and transactions are saved in it.\
After the simulation, you may run `Results_analysis_Homans.py` for analysis of the results (recommended), or call certain functions.

## Getting Started

### Prerequisites
**Community:** first
```
pip install python-louvain
```
then,
```
pip install community
```
**Note:** This library is used in `homans_tools\graph_tools_glossary.py`, and can be replaced by Networkx algorithms (the corresponding lines are marked by `#XXX`).

#### Other Libraries
Following libraries are used but they are available built-in:
- **Decimal**
- **pickle**
- **shutil**

### Executing Program
- First clone the project to a desired directory.
- Open `Homans.py`.
- Then either change the parameters and initial conditions (lines 510-534), or just run the file with given initial conditions.\
We recommend you to at least change `version` variable which is the name of the saved files (line 513).
- Run `Homans.py`.
- Open `Results_analysis_Homans.py` and match the *version*, *N*, and *T* variables with the ones from runned version. Then run it.
- Check the results in the following OS directory: `runned_files/N_T/VERSION`

## Contact
Mohsen Mehrani: <m.mehrani14@gmail.com> \
Taha Enayat: <ta.enayat@gmail.com> \
Project Link: <https://github.com/mmehrani/homans_project>

## License
Distributed under the MIT License. See LICENSE for more information.
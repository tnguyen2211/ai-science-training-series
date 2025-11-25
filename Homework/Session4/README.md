# Session 5: Building Agentic AI Workflows

## Objective
Design and implement a multi-agent system using **LangGraph** and **ALCF Inference Endpoints** to automate a chemical discovery workflow. The goal was to optimize a molecule's geometry using MACE (AI potential) and calculate its physical properties using a custom tool, aggregating the results into a structured JSON format.

---

## Code Modifications

### Custom Tools (`trangNguyen_tools.py`)
I extended the provided toolset by implementing a new molecular property calculator.

* **Modification:** Added `calculate_molecular_weight` using RDKit.

```python
@tool
def calculate_molecular_weight(smiles: str) -> float:
    """Calculate the molecular weight of a molecule from its SMILES string."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string provided.")
    
    return Descriptors.MolWt(mol)
```
### Multi-Agent Logic (`trangNguyen_multi_agent.py`)

I constructed a StateGraph with two specialized agents to decouple the simulation logic from the data analysis:
 1. Simulation Agent: Responsible for identifying the molecule (SMILES), generating 3D coordinates, and running the MACE geometry optimization.
 2. Analysis Agent: Responsible for post-processing. It calls the custom `calculate_molecular_weight` tool and aggregates that result with the energy values from the Simulation Agent into a final JSON report.

## Outcome
The agent successfully processed the prompt *"Optimize Acetone..."*, executed the MACE simulation on ALCF resources, calculated the weight, and returned a valid JSON:

```json
{
  "molecule_name": "Acetone",
  "smiles": "CC(=O)C",
  "molecular_weight_gmol": 58.08,
  "mace_optimization": {
    "model": "small",
    "device": "cpu",
    "converged": true,
    "final_energy_eV": -55.54700469970703,
    "fmax_used_eV_per_A": 0.05,
    "max_steps_used": 200,
    "final_positions_A": [
      [-1.3068784638904904, -0.17905183869247054, -0.16341393884172842],
      [-0.031484336949516044, 0.5203588454131503, 0.2854141103553286],
      [-0.09704321242844147, 1.6038831650750887, 0.879722473416247],
      [1.3244673966895781, -0.1116486555812218, 0.003965923980663663],
      [-1.063501448478351, -1.122419279802387, -0.6716308762462977],
      [-1.8715019158751045, 0.46865142635799045, -0.8507326532232292],
      [-1.9484992994008685, -0.3895167228659122, 0.7053515178309578],
      [1.8608064170355096, -0.2919411086839573, 0.9476612577481365],
      [1.937810181396671, 0.5662291637654461, -0.6084212592758362],
      [1.195824700597403, -1.0645449834429028, -0.5279165232678739]
    ],
    "final_cell_A": [
      [0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0]
    ]
  }
}
```

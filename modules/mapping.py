import json
import numpy as np

def load_lattice(filepath):
    """
    Loads a symbolic lattice from a JSON file.
    The JSON should contain a list of nodes and optional connections.
    Example:
    {
        "nodes": ["Water", "Earth", "Air", "Fire"],
        "connections": [["Water", "Earth"], ["Air", "Fire"]]
    }
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        lattice = json.load(f)
    return lattice

def generate_connections(nodes, pattern="pairwise"):
    """
    Generates symbolic connections automatically.
    - 'pairwise': connects nodes sequentially
    - 'complete': every node connected to every other node
    - 'random': random sparse network
    """
    import itertools, random

    if pattern == "pairwise":
        return [(nodes[i], nodes[i+1]) for i in range(len(nodes)-1)]

    elif pattern == "complete":
        return list(itertools.combinations(nodes, 2))

    elif pattern == "random":
        all_pairs = list(itertools.combinations(nodes, 2))
        return random.sample(all_pairs, max(1, len(all_pairs)//3))

    else:
        raise ValueError("Unknown pattern type. Use 'pairwise', 'complete', or 'random'.")

def merge_lattices(lattice_a, lattice_b):
    """
    Combines two lattice structures together.
    Duplicates are merged gracefully.
    """
    nodes = list(set(lattice_a['nodes'] + lattice_b['nodes']))
    connections = lattice_a.get('connections', []) + lattice_b.get('connections', [])
    connections = [tuple(sorted(c)) for c in connections]
    connections = list(set(connections))
    return {"nodes": nodes, "connections": connections}

def build_symbolic_lattice_from_vector_data(vector_data, threshold=0.5):
    """
    Converts vector field data into symbolic lattice connections.
    If two vectors have similar direction/magnitude, link their symbolic nodes.
    """
    connections = []
    magnitudes = np.sqrt(vector_data[:, 2]**2 + vector_data[:, 3]**2)
    normalized = magnitudes / magnitudes.max()

    for i in range(len(normalized)-1):
        if abs(normalized[i] - normalized[i+1]) < threshold:
            connections.append((f"Node_{i}", f"Node_{i+1}"))

    nodes = [f"Node_{i}" for i in range(len(normalized))]
    return {"nodes": nodes, "connections": connections}

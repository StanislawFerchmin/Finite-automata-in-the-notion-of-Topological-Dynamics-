import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Set, Tuple, List, Union
import itertools
F = nx.MultiDiGraph()
transitions = [
    ("00", "00", "0"), ("00", "01", "1"),  # From 00
    ("01", "10", "0"), ("01", "t", "1"),   # From 01 (11 forbidden)
    ("10", "00", "0"), ("10", "t", "1"),   # From 10 (101 forbidden)
    ("t", "t", "0"),   ("t", "t", "1")     # Trap state
]





class FiniteAutomata():
    """
    A class to represent a Finite Automaton for an SFT.

    Necessary liblaries are listed in req.txt

    Args:
        states (set): A set of state names, e.g., {'00', '01', '10', 't'}.
        alphabet (set): The symbols, e.g., {'0', '1'}.
        initial_state (str): The starting node.
        final_states (set): The set of non-trap states.
        Note that trap state should be called "t"
        dynamics (dict): Dictionary mapping (current_state, label) -> next_state.
    """
    def __init__(self,states: Set[str],alphabet: Set[str],initial_state: str,final_states: Set[str],dynamics: Dict[Tuple[str,str], str]):
        """dynamics need to be written in a form: {(from_edge, transition_label) : to_edge , ...}"""
        self.states = states
        self.alphabet = alphabet
        self.initial = initial_state
        self.final = final_states
        self.dynamics = dynamics
        self.graph = self.create_graph()
        self.matrix = self.adj_matrix()
    def accepts(self, sequence):
        """Check if sequence is accepted by defined FA"""
        if isinstance(sequence,list):
            sequence = ''.join(map(str,sequence))
        state = self.initial
        for item in sequence:
            state = self.dynamics.get((state,item), 't')
            if state == 't':
                return False
        return state in self.final    
    def create_graph(self):
        """Creating networkx graph for further analyisis"""
        G = nx.MultiDiGraph()
        for (u , v), call in self.dynamics.items():
            G.add_edge(u, v, label = call)
        return G
    def adj_matrix(self):
        """Returns adjency matrix of legal transitions"""
        legal = sorted([n for n in self.states if n != "t"])
        A = nx.DiGraph()
        A.add_nodes_from(legal)
        for (u, call), v in self.dynamics.items():
            if u in legal and v in legal:
                A.add_edge(u,v)
        return nx.to_numpy_array(A, nodelist=legal)
    def topological_entropy(self):
        lambdas = np.linalg.eigvals(self.matrix)
        abs_lambdas = np.abs(lambdas)
        return np.log2(max(abs_lambdas))
    def get_language(self, lenght: int):
        return [''.join(p) for p in itertools.product(self.alphabet, repeat=lenght) if self.accepts(''.join(p))]








states_set = {"00", "01", "10", "t"}
alphabet_set = {"0", "1"}
initial_q = "00"
final_f = {"00", "01", "10"}

# Transition Logic (delta function)
dynamics_dict = {
    ("00", "0"): "00", ("00", "1"): "01",
    ("01", "0"): "10", ("01", "1"): "t",   # 11 is forbidden
    ("10", "0"): "00", ("10", "1"): "t",   # 101 is forbidden
    ("t", "0"): "t",   ("t", "1"): "t"     # Trap remains in trap
}
sft = FiniteAutomata(states_set, alphabet_set, initial_q, final_f, dynamics_dict)
FA = FiniteAutomata({'a', 'b', 't'}, {'0', '1'}, 'a', {'b', 'a'}, {('a', '0'): 'a',('a', '1') : 'b', ('b','0') : 'a', ('b','1') : 't', ('b', '0') : 'a', ('t','0') : 't', ('t','1') : 't'})
# --- EXECUTION ---
if __name__ == "__main__":
    print(f"Topological Entropy: {sft.topological_entropy():.4f}")
    print(f"Adjency matrix of our system: {sft.adj_matrix()}")
    print(f"Legal words of length 4: {sft.get_language(4)}")
    test_str = "010010"
    print(f"Is '{test_str}' accepted? {sft.accepts(test_str)}")

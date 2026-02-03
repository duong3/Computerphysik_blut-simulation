import networkx as nx
import numpy as np
import scipy.sparse.linalg

class FlowSolver:
    """
    Berechnet Druck und Fluss im Gefäßsystem basierend auf Hagen-Poiseuille.
    """
    def __init__(self, viscosity=0.004):
        # Blutviskosität (ca. 0.004 Pa*s)
        self.mu = viscosity

    def calculate_resistance(self, graph):
        """
        Schritt 1: Widerstand R für jedes Gefäß berechnen.
        Formel: R = (8 * mu * L) / (pi * r^4)
        """
        for u, v, data in graph.edges(data=True):
            r = data['radius']
            L = data['length']
            
            # Schutz vor Radius 0 (Vermeidung von Division durch Null)
            if r < 1e-6: 
                r = 1e-6
                
            # Hagen-Poiseuille Gesetz
            resistance = (8 * self.mu * L) / (np.pi * r**4)
            
            # Wir speichern den Widerstand direkt in der Kante
            graph[u][v]['resistance'] = resistance

    def solve_network(self, graph, p_inlet, p_outlet):
        """
        Schritt 2: Das lineare Gleichungssystem lösen.
        Wir suchen den Druck P an jedem Knoten.
        """
        nodes = list(graph.nodes())
        node_to_idx = {n: i for i, n in enumerate(nodes)}
        n_nodes = len(nodes)
        
        # Matrix A (Leitwert-Matrix) und Vektor b (Randbedingungen)
        # Wir nutzen einfache Numpy-Arrays für die Übersichtlichkeit
        A = np.zeros((n_nodes, n_nodes))
        b = np.zeros(n_nodes)
        
        # Zuerst Widerstände berechnen
        self.calculate_resistance(graph)
        
        for i, node in enumerate(nodes):
            # Fall 1: Eingang (Inlet) - Fester Druck
            if node == "A_Start": # Unser Eingangsknoten (Hardcoded für MVP)
                A[i, i] = 1.0
                b[i] = p_inlet
                
            # Fall 2: Ausgang (Outlet) - Fester Druck
            elif node == "E_End": # Unser Ausgangsknoten
                A[i, i] = 1.0
                b[i] = p_outlet
                
            # Fall 3: Innere Knoten - Kirchhoffsche Regel (Summe der Ströme = 0)
            else:
                # Wir suchen alle Nachbarn (eingehend und ausgehend)
                # Da NetworkX DiGraph gerichtet ist, müssen wir es als ungerichtet betrachten
                # für die Druckberechnung, oder wir iterieren über alle Kanten
                neighbors = list(graph.successors(node)) + list(graph.predecessors(node))
                # Setze Duplikate (falls doppelte Verbindung) zurück
                neighbors = list(set(neighbors))
                
                conductance_sum = 0
                
                for neighbor in neighbors:
                    # Wir müssen die Kante finden (egal welche Richtung)
                    if graph.has_edge(node, neighbor):
                        res = graph[node][neighbor]['resistance']
                    else:
                        res = graph[neighbor][node]['resistance']
                        
                    conductance = 1.0 / res
                    
                    # Matrix füllen: P_node * sum(G) - sum(P_neighbor * G) = 0
                    A[i, node_to_idx[neighbor]] -= conductance
                    conductance_sum += conductance
                
                A[i, i] = conductance_sum

        # Lösen des Systems A * x = b
        pressures = np.linalg.solve(A, b)
        
        # Ergebnisse in den Graphen schreiben
        for i, node in enumerate(nodes):
            graph.nodes[node]['pressure'] = pressures[i]
            
        return pressures

    def update_flow(self, graph):
        """
        Schritt 3: Fluss Q basierend auf Druckdifferenz berechnen.
        Q = (P_start - P_end) / R
        """
        for u, v, data in graph.edges(data=True):
            p_start = graph.nodes[u]['pressure']
            p_end = graph.nodes[v]['pressure']
            resistance = data['resistance']
            
            # Fluss berechnen
            flow = (p_start - p_end) / resistance
            graph[u][v]['flow'] = flow
            

    def calculate_velocity(self, graph):
        """
        Berechnet die Flussgeschwindigkeit v = Q / A für jedes Gefäß.
        Wird für die Bewegung der Partikel benötigt.
        """
        for u, v, data in graph.edges(data=True):
            flow = data['flow']
            radius = data['radius']
            
            # Fläche A = pi * r^2
            area = np.pi * (radius ** 2)
            
            # Geschwindigkeit v = Flow / Area
            # (Wir skalieren das hier etwas herunter, damit die Animation nicht zu schnell rast)
            velocity = (flow / area) * 0.1 
            
            graph[u][v]['velocity'] = velocity
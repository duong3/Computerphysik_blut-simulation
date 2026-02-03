import networkx as nx
import numpy as np
import pyvista as pv

class VascularGraph:
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_node(self, node_id, position):
        self.graph.add_node(node_id, pos=np.array(position))

    def add_vessel(self, start_node, end_node, radius, label="vessel"):
        """
        Jetzt neu: Wir müssen einen RADIUS übergeben!
        """
        pos_a = self.graph.nodes[start_node]['pos']
        pos_b = self.graph.nodes[end_node]['pos']
        length = np.linalg.norm(pos_b - pos_a)
        
        # Wir speichern den Radius direkt in der Kante (Edge Data)
        self.graph.add_edge(start_node, end_node, 
                            length=length, 
                            radius=radius,  # <--- Das ist neu
                            label=label)

    def visualize_3d(self):
        """
        Zeigt echte Röhren basierend auf den Daten im Graphen.
        """
        plotter = pv.Plotter()
        plotter.add_text("Schritt 1.2: Physikalische Eigenschaften (Radien)", font_size=10)
        
        # Wir gehen durch jede Verbindung im logischen Graphen
        for u, v, data in self.graph.edges(data=True):
            p1 = self.graph.nodes[u]['pos']
            p2 = self.graph.nodes[v]['pos']
            r = data['radius'] # Wir holen den Radius aus dem "Gehirn"
            
            # Linie erstellen
            line = pv.Line(p1, p2)
            # Rohr erstellen (mit dem gespeicherten Radius!)
            # n_sides=15 ist ein guter Kompromiss für deinen Laptop
            tube = line.tube(radius=r, n_sides=15)
            
            # Farbe wählen
            color = 'white'
            if data['label'] == 'artery': color = '#ff4444' # Rot
            elif data['label'] == 'capillary': color = '#44ff44' # Grün
            elif data['label'] == 'vein': color = '#4444ff' # Blau
            
            plotter.add_mesh(tube, color=color, show_edges=False)
            
        plotter.show_grid()
        plotter.show()

if __name__ == "__main__":
    network = VascularGraph()
    
    # --- Szenario: Die Mikrovaskuläre Einheit (Der Diamant) ---
    
    # 1. Knoten (Punkte im Raum)
    network.add_node("A_Start", [0, 0, 0])
    network.add_node("B_Split", [10, 0, 0])
    
    # Die Kapillaren gehen oben und unten vorbei
    network.add_node("C_Top", [15, 3, 0])
    network.add_node("C_Bottom", [15, -3, 0])
    
    network.add_node("D_Join", [20, 0, 0])
    network.add_node("E_End", [30, 0, 0])
    
    # 2. Gefäße (Kanten mit Radien)
    # Arterie: Dick (Radius 1.5)
    network.add_vessel("A_Start", "B_Split", radius=1.5, label="artery")
    
    # Kapillaren: Sehr dünn (Radius 0.5)
    network.add_vessel("B_Split", "C_Top", radius=0.5, label="capillary")
    network.add_vessel("B_Split", "C_Bottom", radius=0.5, label="capillary")
    
    # Zusammenfluss in Vene
    network.add_vessel("C_Top", "D_Join", radius=0.5, label="capillary")
    network.add_vessel("C_Bottom", "D_Join", radius=0.5, label="capillary")
    
    # Vene: Mittel-Dick (Radius 1.2), aber elastischer (kommt später)
    network.add_vessel("D_Join", "E_End", radius=1.2, label="vein")
    
    print(f"Netzwerk gebaut: {network.graph.number_of_nodes()} Knoten.")
    
    # 3. Visualisieren
    network.visualize_3d()
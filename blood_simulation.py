import numpy as np
import pyvista as pv
import random
import time

# Unsere eigenen Module importieren
from vascular_graph import VascularGraph
from physics import FlowSolver

SLOW_LAPTOP = False  # Änder auf True, wenn Laptop langsam. Dann Blut als rote Vierecke statt runde Kugeln => leichter zu rendern

class BloodParticle:
    def __init__(self, graph, start_node):
        self.graph = graph
        self.current_node = start_node
        self.target_node = None
        self.progress = 0.0 # 0.0 (Start) bis 1.0 (Ende des Rohrs)
        self.velocity = 0.0
        self.finished = False # Wenn es den Ausgang erreicht hat
        
        # Wir suchen uns sofort den ersten Weg
        self.pick_next_path()

    def pick_next_path(self):
        """
        Entscheidet an einer Kreuzung, wo es lang geht.
        Basiert auf dem Fluss: Mehr Fluss = höhere Wahrscheinlichkeit.
        """
        # Alle möglichen Ausgänge finden
        neighbors = list(self.graph.successors(self.current_node))
        
        if not neighbors:
            self.finished = True
            return

        # Wenn es nur einen Weg gibt (keine Verzweigung)
        if len(neighbors) == 1:
            self.target_node = neighbors[0]
        else:
            # An einer Verzweigung: Wahrscheinlichkeit basierend auf Flow berechnen
            flows = []
            for n in neighbors:
                flows.append(self.graph[self.current_node][n]['flow'])
            
            # Zufallsauswahl gewichtet nach Flussstärke
            # (Das ist der Moment, wo sich Blut physikalisch korrekt aufteilt!)
            self.target_node = random.choices(neighbors, weights=flows, k=1)[0]

        # Daten für die Bewegung laden
        edge_data = self.graph[self.current_node][self.target_node]
        self.edge_length = edge_data['length']
        self.velocity = edge_data['velocity']
        self.progress = 0.0

    def update(self, dt=0.5):
        if self.finished: return

        # Bewegung: Strecke = Geschwindigkeit * Zeit
        step = self.velocity * dt
        
        # Fortschritt in % umrechnen (Step / Gesamtlänge)
        self.progress += step / self.edge_length
        
        # Wenn Ende des Rohrs erreicht
        if self.progress >= 1.0:
            self.current_node = self.target_node
            self.pick_next_path()

    def get_position(self):
        """Berechnet die echte 3D-Koordinate"""
        if self.finished: return np.array([0,0,0]) # Dummy

        p_start = self.graph.nodes[self.current_node]['pos']
        p_end = self.graph.nodes[self.target_node]['pos']
        
        # Lineare Interpolation zwischen Start und Ende
        # Pos = Start + (Vektor * Fortschritt)
        return p_start + (p_end - p_start) * self.progress

class SimulationApp:
    def __init__(self, num_particles=100):
        # 1. Setup Graph (Der Diamant)
        self.network = VascularGraph()
        self.setup_diamond_geometry()
        
        # 2. Setup Physik
        self.solver = FlowSolver()
        self.solver.solve_network(self.network.graph, p_inlet=100, p_outlet=0)
        self.solver.update_flow(self.network.graph)
        self.solver.calculate_velocity(self.network.graph) # <--- Neu!
        
        # 3. Partikel erstellen
        self.particles = [BloodParticle(self.network.graph, "A_Start") for _ in range(num_particles)]
        # Wir verteilen sie am Anfang etwas zufällig, damit nicht alle als Klumpen starten
        for p in self.particles:
            p.progress = random.random()
            # Kleiner Hack: Wir müssen sicherstellen, dass sie korrekte Ziele haben
            # Da wir progress manuell gesetzt haben, lassen wir es so, 
            # im echten Loop korrigiert sich das gleich.

        # 4. Grafik Vorbereitung (für Performance optimiert)
        self.plotter = pv.Plotter()
        self.plotter.add_text("Schritt 3: Physikalische Bewegung", font_size=10)
        
        # Statische Röhren zeichnen (Ganz dünn als Referenz)
        for u, v, data in self.network.graph.edges(data=True):
            p1 = self.network.graph.nodes[u]['pos']
            p2 = self.network.graph.nodes[v]['pos']
            radius = data['radius'] # Wir holen den Radius aus den Daten
            
            # Linie erstellen und zum Rohr aufblasen
            line = pv.Line(p1, p2)
            tube = line.tube(radius=radius, n_sides=10) # n_sides=10 reicht für den Laptop
            
            # Hellgrau und leicht durchsichtig ('gainsboro' ist ein helles Grau)
            self.plotter.add_mesh(tube, color='gainsboro', opacity=0.5, show_edges=False)

        # Partikel als Punktwolke vorbereiten
        self.particle_cloud = pv.PolyData(np.zeros((num_particles, 3)))

        if SLOW_LAPTOP:
            self.plotter.add_mesh(self.particle_cloud, color='red', point_size=8, style='points')
        else:
            # echtes Blut.
            self.plotter.add_mesh(self.particle_cloud, color='red', point_size=10, render_points_as_spheres=True)

    def setup_diamond_geometry(self):
        # Gleiche Geometrie wie in Schritt 1.2
        n = self.network
        n.add_node("A_Start", [0, 0, 0])
        n.add_node("B_Split", [10, 0, 0])
        n.add_node("C_Top", [15, 3, 0])
        n.add_node("C_Bottom", [15, -3, 0])
        n.add_node("D_Join", [20, 0, 0])
        n.add_node("E_End", [30, 0, 0])
        
        # ARTERIE: 1. Station des Bluts)
        n.add_vessel("A_Start", "B_Split", radius=1.5, label="artery")
        # KAPILLAREN: Mittelstücke
        n.add_vessel("B_Split", "C_Top", radius=0.5, label="capillary") 
        n.add_vessel("B_Split", "C_Bottom", radius=0.5, label="capillary")
        n.add_vessel("C_Top", "D_Join", radius=0.5, label="capillary")
        n.add_vessel("C_Bottom", "D_Join", radius=0.5, label="capillary")
        # VENE: Letzte Station
        n.add_vessel("D_Join", "E_End", radius=1.2, label="vein")

    def run(self):
        print("Simulation läuft... (Fenster schließen zum Beenden)")
        self.plotter.show(interactive_update=True)
        
        while True:
            # 1. Daten Update
            positions = []
            for p in self.particles:
                p.update()
                if p.finished:
                    # Reset zum Anfang (Endlosschleife)
                    p.current_node = "A_Start"
                    p.pick_next_path()
                    p.finished = False
                
                positions.append(p.get_position())
            
            # 2. Grafik Update (Ganzes Array auf einmal austauschen -> Schnell!)
            self.particle_cloud.points = np.array(positions)
            
            self.plotter.update()
            time.sleep(0.01)

if __name__ == "__main__":
    app = SimulationApp(num_particles=200)
    app.run()
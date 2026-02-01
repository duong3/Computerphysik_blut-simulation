import numpy as np
import pyvista as pv
import time

class BloodVessel:
    """
    Klasse für die Geometrie der Ader.
    Später kann hier die Dehnbarkeit (Compliance) simuliert werden.
    """
    def __init__(self, start_point, end_point, radius=1.0):
        self.start = np.array(start_point)
        self.end = np.array(end_point)
        self.radius = radius
        self.length = np.linalg.norm(self.end - self.start)
        
        # Erstelle das 3D-Objekt (Ein Rohr)
        # Resolution bestimmt, wie "rund" das Rohr ist
        line = pv.Line(self.start, self.end)
        self.mesh = line.tube(radius=self.radius, n_sides=20)

class BloodFlowSimulation:
    """
    Hauptklasse für die Simulation. Verwaltet Partikel und Physik.
    """
    def __init__(self, num_particles=200):
        self.vessel = BloodVessel([0, 0, 0], [20, 0, 0], radius=2.0)
        self.num_particles = num_particles
        
        # --- NUMPY TEIL ---
        # Wir speichern alle Positionen in einem großen Array.
        # Spalten: [x, y, z]
        self.positions = np.zeros((num_particles, 3))
        
        # Initialisierung: Zufällige Verteilung im Rohr
        # X: Zufällig entlang der Länge
        self.positions[:, 0] = np.random.uniform(0, self.vessel.length, num_particles)
        # Y und Z: Zufällig im Radius (vereinfacht als Box, später Kreis)
        self.positions[:, 1] = np.random.uniform(-1.5, 1.5, num_particles)
        self.positions[:, 2] = np.random.uniform(-1.5, 1.5, num_particles)
        
        # Geschwindigkeit der Partikel (konstant für den Anfang)
        self.velocity = 0.1 

        # --- PYVISTA VISUALISIERUNG TEIL ---
        # Wir erstellen eine Punktwolke (PolyData) für PyVista
        self.cloud = pv.PolyData(self.positions)
        
        # Hier könnten wir später Daten anheften (z.B. 'type': 'leukozyt')
        self.cloud['velocity'] = np.ones(num_particles) * self.velocity

    def update(self):
        """
        Ein Zeitschritt der Simulation.
        Hier kommt später die Logik für Kollisionen und Thrombozyten rein.
        """
        # Bewegung entlang der X-Achse
        self.positions[:, 0] += self.velocity
        
        # Endlos-Schleife: Wenn Partikel hinten rausfliegen, vorne wieder reinsetzen
        # (Simuliert einen kontinuierlichen Fluss)
        mask = self.positions[:, 0] > self.vessel.length
        self.positions[mask, 0] = 0
        
        # Update der Grafik-Daten
        self.cloud.points = self.positions

    def run(self):
        """
        Startet das Visualisierungs-Fenster
        """
        plotter = pv.Plotter()
        plotter.add_text("Blutbahn Simulation - MVP", font_size=12)
        
        # 1. Das Gefäß zeichnen (Transparent, damit man reinsieht)
        plotter.add_mesh(self.vessel.mesh, color='pink', opacity=0.3, show_edges=False)
        
        # 2. Die Blutzellen zeichnen (Als rote Kugeln)
        # render_points_as_spheres macht es hübsch und performant
        plotter.add_mesh(self.cloud, color='red', render_points_as_spheres=True, point_size=10)
        
        print("Simulation gestartet... (Schließe das Fenster zum Beenden)")
        
        # Die Animations-Schleife
        plotter.show(interactive_update=True)
        
        while True:
            self.update()
            plotter.update()
            time.sleep(0.01) # Kleine Pause, damit es nicht zu schnell ist

if __name__ == "__main__":
    sim = BloodFlowSimulation(num_particles=300)
    sim.run()
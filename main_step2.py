from vascular_graph import VascularGraph
from physics import FlowSolver

def run_simulation_step():
    # 1. Graphen erstellen (Das Diamant-Modell aus Schritt 1)
    net = VascularGraph()
    
    # Knoten
    net.add_node("A_Start", [0, 0, 0])
    net.add_node("B_Split", [10, 0, 0])
    net.add_node("C_Top", [15, 3, 0])
    net.add_node("C_Bottom", [15, -3, 0])
    net.add_node("D_Join", [20, 0, 0])
    net.add_node("E_End", [30, 0, 0])
    
    # Kanten (Arterie -> Kapillaren -> Vene)
    net.add_vessel("A_Start", "B_Split", radius=1.5, label="artery")
    net.add_vessel("B_Split", "C_Top", radius=0.5, label="capillary")     # Weg 1
    net.add_vessel("B_Split", "C_Bottom", radius=0.5, label="capillary")  # Weg 2
    net.add_vessel("C_Top", "D_Join", radius=0.5, label="capillary")
    net.add_vessel("C_Bottom", "D_Join", radius=0.5, label="capillary")
    net.add_vessel("D_Join", "E_End", radius=1.2, label="vein")
    
    # 2. Physik-Engine initialisieren
    solver = FlowSolver()
    
    # 3. Berechnen (Eingangsdruck 100 mmHg, Ausgang 0 mmHg)
    # mmHg müssen wir eigentlich in Pascal umrechnen, aber für die relativen Werte reicht das so.
    print("--- Starte Berechnung ---")
    solver.solve_network(net.graph, p_inlet=100.0, p_outlet=0.0)
    solver.update_flow(net.graph)
    
    # 4. Ergebnisse ausgeben (Text-basiert zur Überprüfung)
    print("\nErgebnisse:")
    
    # Fluss in der Arterie
    flow_artery = net.graph["A_Start"]["B_Split"]["flow"]
    print(f"Fluss Arterie (Eingang): {flow_artery:.4f}")
    
    # Fluss in den Kapillaren
    flow_top = net.graph["B_Split"]["C_Top"]["flow"]
    flow_bot = net.graph["B_Split"]["C_Bottom"]["flow"]
    print(f"Fluss Kapillare Oben:    {flow_top:.4f}")
    print(f"Fluss Kapillare Unten:   {flow_bot:.4f}")
    
    # Plausibilitäts-Check
    print("\n--- Plausibilitäts-Check ---")
    if abs(flow_artery - (flow_top + flow_bot)) < 0.001:
        print("✅ EINGANG = SUMME DER KAPILLAREN")
        print("   Physik funktioniert: Masseerhaltung gilt.")
    else:
        print("❌ FEHLER: Irgendwo geht Blut verloren!")

    if abs(flow_top - flow_bot) < 0.001:
        print("✅ SYMMETRIE")
        print("   Beide Wege sind gleich lang/dick, also fließt gleich viel.")
    else:
        print("❌ ASYMMETRIE: Das sollte bei gleichem Radius nicht sein.")

if __name__ == "__main__":
    run_simulation_step()
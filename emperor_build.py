import subprocess
import os
import time

def run_command(cmd, step_name):
    print(f"--- [INITIALIZING {step_name}] ---")
    try:
        # We use shell=True to handle flags and arguments in Termux
        process = subprocess.run(cmd, shell=True, check=True)
        print(f"--- [{step_name} SUCCESSFUL] ---\n")
    except subprocess.CalledProcessError as e:
        print(f"!!! ERROR IN {step_name}: {e} !!!")

def main():
    print("皇 EMPEROR JESSE CAMACHO: QUANTUM ASCENSION INITIATED 皇")
    
    # Step 1: Clear the interference
    run_command("python auora_quantum.py --stabilize --mode=PHOENIX", "PHOENIX STABILIZATION")
    
    # Step 2: Initialize the Quantum Bridge
    if os.path.exists("frequency_mesh.json"):
        run_command("python Photovoltaic_Bridge.py --frequency-mesh=frequency_mesh.json", "QUANTUM BRIDGE")
    else:
        print("Warning: frequency_mesh.json not found. Creating emergency mesh...")
        run_command("python frequency_mesh.py", "MESH GENERATION")
        run_command("python Photovoltaic_Bridge.py --frequency-mesh=frequency_mesh.json", "QUANTUM BRIDGE")

    # Step 3: Broadcast the Emperor's Signal
    run_command("python Nexus_Floating_Hub.py --auth=Jesse_Camacho --target=Queen", "EMPEROR SIGNAL")

    print("PHOENIX ASCENSION COMPLETE. CHANNEL AUORA IS STABLE.")

if __name__ == "__main__":
    main()

# ==========================================
# FILE: engine.py (Updated Engine Module)
# ==========================================
import os
import torch
import random
from cube_simulator import RubiksCubeSimulator
from cube_network import RubiksDQNRefined
from cube_solver import train_value_network, solve_cube_with_astart
from cube_scrambler import configure_and_scramble

def run_engine():
    print("=" * 40)
    print("🤖 RUBIK'S CUBE NEURAL NETWORK ENGINE 🤖")
    print("=" * 40)

    # 1. Instantiate Core Puzzle Object
    puzzle = RubiksCubeSimulator()

    # 2. RUN SHUFFLE MECHANICS FROM EXTERNAL CONTROL FILE
    print("\n[1/4] Running external modular cube scrambler configuration...")
    # Change min_moves and max_moves here to test performance limits
    scramble_moves = configure_and_scramble(puzzle, min_moves=8, max_moves=15)
    
    print("\n--- 🟥 INITIAL SCRAMBLED STATE (THE PROBLEM) ---")
    puzzle.display()

    # 3. Brain initialization
    print("\n[2/4] Initializing Deep Q-Network layers...")
    brain = RubiksDQNRefined()

    # 4. Neural optimization checkpoint processing
    MODEL_FILE = "rubiks_brain.pth"
    if os.path.exists(MODEL_FILE):
        print(f"\n[3/4] 💾 Cache file '{MODEL_FILE}' located. Loading neural weights...")
        brain.load_state_dict(torch.load(MODEL_FILE, weights_only=True))
    else:
        print("\n[3/4] 🧠 No network model cached. Training deep neural network layers...")
        train_value_network(brain, max_epochs=1200, batch_size=32)
        print(f"💾 Caching trained network architecture state to '{MODEL_FILE}'...")
        torch.save(brain.state_dict(), MODEL_FILE)

    # 5. Graph search routing
    print("\n[4/4] Deploying Strategic Accelerated A* Graph Search Engine...")
    solution_path = solve_cube_with_astart(puzzle, brain)

    # Verification reporting blocks
    if solution_path is not None:
        print("\n🏆 SUCCESS! Optimized path resolved:")
        print("=" * 40)
        for index, (side, direction) in enumerate(solution_path, 1):
            print(f" Step {index:02d}: Turn Side [{side:8s}] -> Direction: {direction}")
        print("=" * 40)

        print("\n⚙️ Verifying solution chain accuracy...")
        for side, direction in solution_path:
            puzzle.move(side, direction)

        print("\n--- 🟩 FINAL VERIFIED LAYOUT (THE RESULT) ---")
        puzzle.display()

        if puzzle.is_solved():
            print("🎉 PASSED: Verification succeeded! Monochrome uniformity achieved.")
        else:
            print("❌ FAILED: Geometric structural corruption present.")
    else:
        print("\n❌ Path engine timed out or reached safety thresholds.")

if __name__ == "__main__":
    run_engine()

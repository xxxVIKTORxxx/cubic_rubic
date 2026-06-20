# ==========================================
# FILE: engine.py (Main Execution Hub)
# ==========================================
import sys
import os
import torch
from cube_simulator import RubiksCubeSimulator
from cube_network import RubiksDQNRefined
from cube_solver import train_value_network, solve_cube_with_astart

def run_engine():
    print("=" * 40)
    print("🤖 RUBIK'S CUBE NEURAL NETWORK ENGINE 🤖")
    print("=" * 40)

    # 1. Scramble and print the layout first
    print("\n[1/4] Creating a clean puzzle state & shuffling it...")
    puzzle = RubiksCubeSimulator()
    scramble_moves = puzzle.scramble(steps=4)
    print(f"🎲 Scrambled using sequence: {scramble_moves}")
    print("\n--- 🟥 INITIAL SCRAMBLED STATE (THE PROBLEM) ---")
    puzzle.display()

    # 2. Initialize Brain
    print("\n[2/4] Initializing Deep Q-Network layers...")
    brain = RubiksDQNRefined()

    # 3. Smart Checkpoint Handling: Load or Train
    MODEL_FILE = "rubiks_brain.pth"
    if os.path.exists(MODEL_FILE):
        print(f"\n[3/4] 💾 Found existing brain file '{MODEL_FILE}'. Loading trained weights instantly...")
        brain.load_state_dict(torch.load(MODEL_FILE, weights_only=True))
    else:
        print("\n[3/4] 🧠 No existing brain found. Bootstrapping training via backward simulation...")
        train_value_network(brain, max_epochs=1000, batch_size=32)
        print(f"💾 Saving trained brain weights to '{MODEL_FILE}' for next launch...")
        torch.save(brain.state_dict(), MODEL_FILE)

    # 4. Deploy Solver
    print("\n[4/4] Deploying Strategic A* Graph Search Engine...")
    solution_path = solve_cube_with_astart(puzzle, brain)

    # Final Verification View
    if solution_path is not None:
        print("\n🏆 SUCCESS! The engine optimized a solution path:")
        print("=" * 40)
        for index, (side, direction) in enumerate(solution_path, 1):
            print(f" Step {index:02d}: Turn Side [{side:8s}] -> Direction: {direction}")
        print("=" * 40)

        print("\n⚙️ Live-executing the found path onto our scrambled cube...")
        for side, direction in solution_path:
            puzzle.move(side, direction)

        print("\n--- 🟩 FINAL STATE AFTER CORRECTION (THE RESULT) ---")
        puzzle.display()

        if puzzle.is_solved():
            print("🎉 VERIFICATION PASSED: Every face is 100% monochrome and solved!")
        else:
            print("❌ VERIFICATION FAILED: The layout did not return to its base color uniformity.")
    else:
        print("\n❌ The engine could not resolve the cube path.")


if __name__ == "__main__":
    run_engine()
from cube_simulator import RubiksCubeSimulator
from cube_network import RubiksDQNRefined, encode_cube_state

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import heapq
import copy

# --- Reuse the Simulator & Network Architecture from previous steps ---
# (Assuming RubiksCubeSimulator and RubiksDQNRefined are loaded)

def get_state_string(cube_instance):
    """Creates a unique string token of the cube state to use as a dictionary key."""
    flat_chars = []
    for side in cube_instance.SIDES:
        for row in cube_instance.sides[side]:
            flat_chars.extend(row)
    return "".join(flat_chars)

def encode_cube_state(cube_instance):
    """Encodes side-based matrix to a 324-element 1D array for PyTorch."""
    color_map = {col: idx for idx, col in enumerate(cube_instance.COLORS)}
    flattened_numerical = []
    for side in cube_instance.SIDES:
        for row in cube_instance.sides[side]:
            for sticker in row:
                flattened_numerical.append(color_map[sticker])
    one_hot = np.eye(6)[flattened_numerical]
    return torch.tensor(one_hot.flatten(), dtype=torch.float32)

def get_state_string(cube_instance):
    """Creates a unique string token of the cube state to use as a dictionary key."""
    flat_chars = []
    for side in cube_instance.SIDES:
        for row in cube_instance.sides[side]:
            flat_chars.extend(row)
    return "".join(flat_chars)

# --- 1. Autodidactic Iteration Training Loop ---
def train_value_network(model, max_epochs=2000, batch_size=64):
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
    criterion = nn.MSELoss()
    model.train()

    all_actions = []
    for side in RubiksCubeSimulator.SIDES:
        for direction in ['left', 'right']:
            all_actions.append((side, direction))

    best_loss = float('inf')
    plateau_counter = 0
    patience_limit = 5

    print("Starting Deep Autodidactic Loop with Dynamic Complexity Scaling...")
    for epoch in range(max_epochs):
        states_batch = []
        targets_batch = []

        # CURRICULUM LEARNING: Slowly increase max distance as epochs get higher
        if epoch < 200:
            max_scramble_distance = 6
        elif epoch < 500:
            max_scramble_distance = 12
        elif epoch < 1000:
            max_scramble_distance = 18
        else:
            max_scramble_distance = 25 # Absolute ceiling for 3x3 complexity

        for _ in range(batch_size):
            env = RubiksCubeSimulator()
            scramble_steps = random.randint(1, max_scramble_distance)
            
            for _ in range(scramble_steps):
                side, direction = random.choice(all_actions)
                env.move(side, direction)
            
            states_batch.append(encode_cube_state(env))
            targets_batch.append([float(scramble_steps)])

        X = torch.stack(states_batch)
        y = torch.tensor(targets_batch, dtype=torch.float32)

        optimizer.zero_grad()
        predictions = model(X)
        loss = criterion(predictions, y)
        loss.backward()
        optimizer.step()

        current_loss = loss.item()
        
        if epoch % 50 == 0:
            print(f"🔄 Epoch {epoch:04d} (Max Train Depth: {max_scramble_distance:02d}) | Loss: {current_loss:.4f}")
            
            if current_loss < best_loss - 0.05:
                best_loss = current_loss
                plateau_counter = 0
            else:
                plateau_counter += 1
                
            # Only trigger dynamic stop if we are already training on the deepest configurations
            if plateau_counter >= patience_limit and epoch > 1100:
                print(f"🎯 Chaos Convergence reached at epoch {epoch} (Loss: {current_loss:.4f}). Optimization complete!")
                break

# --- 2. Strategic A* Solver (The Global Picture Engine) ---
# ==========================================
# PARTIAL CODE FOR UPDATE WITHIN: cube_solver.py
# ==========================================
import heapq
import torch
import copy

def solve_cube_with_astart(scrambled_cube, model):
    model.eval()
    
    # --- SPEED TUNING HYPERPARAMETER ---
    # 1.0 = Standard A* (Perfect, but incredibly slow for depths > 6)
    # 2.0+ = Weighted A* (Runs up to 1000x faster, near-optimal paths)
    HEURISTIC_WEIGHT = 2.2 

    all_actions = []
    for side in RubiksCubeSimulator.SIDES:
        for direction in ['left', 'right']:
            all_actions.append((side, direction))

    queue = []
    counter = 0
    
    start_tensor = encode_cube_state(scrambled_cube).unsqueeze(0)
    with torch.no_grad():
        predicted_h = model(start_tensor).item()
    
    # Push initial node
    heapq.heappush(queue, (predicted_h * HEURISTIC_WEIGHT, counter, 0, copy.deepcopy(scrambled_cube), []))
    
    visited_states = set()
    visited_states.add(get_state_string(scrambled_cube))

    print("\nExecuting Strategic Accelerated Weighted A* Search...")
    
    # Safety breakout loop threshold
    max_evaluations = 50000 
    eval_count = 0

    while queue:
        eval_count += 1
        if eval_count > max_evaluations:
            print(f"⚠️ Search limit reached ({max_evaluations} states checked). Breaking to prevent RAM freeze.")
            return None

        f, _, g, current_cube, path = heapq.heappop(queue)

        if current_cube.is_solved():
            print(f"🎉 Solution optimized in {g} moves! (Evaluated {eval_count} total states)")
            return path

        for side, direction in all_actions:
            next_cube = copy.deepcopy(current_cube)
            next_cube.move(side, direction)
            state_str = get_state_string(next_cube)

            if state_str in visited_states:
                continue
            
            visited_states.add(state_str)

            next_tensor = encode_cube_state(next_cube).unsqueeze(0)
            with torch.no_grad():
                h = model(next_tensor).item()
            
            next_g = g + 1
            # Apply Heuristic Weighting here to prioritize deeper progress
            next_f = next_g + (h * HEURISTIC_WEIGHT)
            next_path = path + [(side, direction)]

            counter += 1
            heapq.heappush(queue, (next_f, counter, next_g, next_cube, next_path))

    print("❌ Strategy failed. Search space exhausted.")
    return None


# --- Execution Test Run ---
if __name__ == "__main__":
    # 1. Instantiate the Brain
    nn_brain = RubiksDQNRefined()
    
    # 2. Train it via Autodidactic Iteration (teach it the concept of distance)
    train_value_network(nn_brain, epochs=300, batch_size=32)

    # 3. Create a fresh puzzle and scramble it
    puzzle = RubiksCubeSimulator()
    print("\nScrambling the cube...")
    scramble_history = puzzle.scramble(steps=5) # 5 steps for quick demo execution
    puzzle.display()

    # 4. Let A* Search look at the whole picture and solve it
    solution_path = solve_cube_with_astart(puzzle, nn_brain)
    
    print("\nRequired Move Sequence to Solve:")
    for step, (side, direction) in enumerate(solution_path, 1):
        print(f"Step {step}: Turn Side [{side}] -> Direction: {direction}")

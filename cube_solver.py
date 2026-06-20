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
def train_value_network(model, max_epochs=5000, batch_size=64):
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
    criterion = nn.MSELoss()
    model.train()

    all_actions = []
    for side in RubiksCubeSimulator.SIDES:
        for direction in ['left', 'right']:
            all_actions.append((side, direction))

    # Plateau tracking arrays
    best_loss = float('inf')
    plateau_counter = 0
    patience_limit = 4  # Stop if the loss rolls around the same baseline 4 check-intervals in a row

    print("Starting Deep Autodidactic Loop with Dynamic Plateau Tracking...")
    for epoch in range(max_epochs):
        states_batch = []
        targets_batch = []

        for _ in range(batch_size):
            env = RubiksCubeSimulator()
            # Constrain search layer boundaries (1 to 6 steps)
            scramble_steps = random.randint(1, 6)
            
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
        
        # Check training variance behavior every 50 epochs
        if epoch % 50 == 0:
            print(f"🔄 Epoch {epoch:04d} | Processing Error Baseline: {current_loss:.4f}")
            
            # Monitor if loss is rolling/converging instead of dropping significantly
            if current_loss < best_loss - 0.1:
                best_loss = current_loss
                plateau_counter = 0  # Loss dropped notably, reset patience
            else:
                plateau_counter += 1 # Loss is fluctuating within the same baseline margin
                
            if plateau_counter >= patience_limit and epoch > 100:
                print(f"🎯 Convergence Plateau reached at epoch {epoch} (Loss: {current_loss:.4f}). Optimization complete!")
                break

# --- 2. Strategic A* Solver (The Global Picture Engine) ---
def solve_cube_with_astart(scrambled_cube, model):
    model.eval()
    all_actions = []
    for side in RubiksCubeSimulator.SIDES:
        for direction in ['left', 'right']:
            all_actions.append((side, direction))

    # Priority queue storing elements as: (F_score, unique_counter, steps_taken, current_cube_object, path_taken)
    # Priority Queue automatically bubbles up the lowest F_score
    queue = []
    counter = 0
    
    # Evaluate starting state
    start_state_tensor = encode_cube_state(scrambled_cube).unsqueeze(0)
    with torch.no_grad():
        predicted_h = model(start_state_tensor).item()
    
    # Initial F = 0 actual moves + predicted_h remaining
    heapq.heappush(queue, (predicted_h, counter, 0, copy.deepcopy(scrambled_cube), []))
    
    # Track visited states to eliminate loops completely
    visited_states = set()
    visited_states.add(get_state_string(scrambled_cube))

    print("\nExecuting Strategic A* Search...")
    while queue:
        f, _, g, current_cube, path = heapq.heappop(queue)

        # Strategic check: Have we solved it?
        if current_cube.is_solved():
            print(f"🎉 Solution found globally optimized in {g} moves!")
            return path

        # Try all 12 possible next moves strategically
        for side, direction in all_actions:
            next_cube = copy.deepcopy(current_cube)
            next_cube.move(side, direction)
            state_str = get_state_string(next_cube)

            if state_str in visited_states:
                continue # Skip loops entirely
            
            visited_states.add(state_str)

            # Let neural network evaluate the strategic value of this transition
            next_tensor = encode_cube_state(next_cube).unsqueeze(0)
            with torch.no_grad():
                h = model(next_tensor).item()
            
            # Update scores
            next_g = g + 1
            next_f = next_g + h
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

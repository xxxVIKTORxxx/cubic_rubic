import random

def configure_and_scramble(cube_instance, min_moves=5, max_moves=20):
    """
    Manages custom scramble sequences with dynamic safety limiters,
    hardware requirement warnings, and input boundaries.
    """
    # Absolute threshold before RAM exhaustion during graph searches
    HARD_CEILING = 20 

    if max_moves > HARD_CEILING:
        print("\n" + "!" * 50)
        print(f"⚠️ WARNING: Your max scramble moves ({max_moves}) exceeds the safe threshold ({HARD_CEILING}).")
        print("Solving a cube of this complexity with an unoptimized search tree requires:")
        print("👉 Approximately 16GB to 32GB of system RAM.")
        print(f"👉 Processing time could grow exponentially.")
        print(f"🔒 Lowering max_moves automatically to {HARD_CEILING} for safety.")
        print("💡 To bypass this, edit the HARD_CEILING manually inside 'cube_scrambler.py'.")
        print("!" * 50 + "\n")
        
        # Enforce safety constraint
        max_moves = HARD_CEILING

    if min_moves > max_moves:
        min_moves = max_moves

    # Determine randomized trajectory depth
    total_steps = random.randint(min_moves, max_moves)
    
    # Calculate Expected Processing Metrics
    expected_seconds = 0.1 if total_steps <= 5 else (5.0 if total_steps <= 10 else 45.0)
    print(f"⏱️ Expected Strategic Resolution Time: ~{expected_seconds} seconds (Based on {total_steps} shuffles)")

    # Execute scramble loops
    all_actions = []
    for side in cube_instance.SIDES:
        for direction in ['left', 'right']:
            all_actions.append((side, direction))

    scramble_history = []
    for _ in range(total_steps):
        side, direction = random.choice(all_actions)
        cube_instance.move(side, direction)
        scramble_history.append((side, direction))

    return scramble_history
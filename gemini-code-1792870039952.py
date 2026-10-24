import heapq

# Define the goal state (0 represents the empty/blank space)
GOAL_STATE = (1, 2, 3, 4, 5, 6, 7, 8, 0)

def get_neighbors(state):
    """Generates all valid neighboring states from the current state."""
    neighbors = []
    zero_index = state.index(0)
    r, c = divmod(zero_index, 3)
    
    # Possible movements: Up, Down, Left, Right
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < 3 and 0 <= nc < 3:
            new_zero_index = nr * 3 + nc
            state_list = list(state)
            # Swap the blank space with the adjacent tile
            state_list[zero_index], state_list[new_zero_index] = state_list[new_zero_index], state_list[zero_index]
            neighbors.append(tuple(state_list))
            
    return neighbors

def heuristic_misplaced(state):
    """Heuristic 1: Counts the number of tiles out of place."""
    return sum(1 for i in range(9) if state[i] != 0 and state[i] != GOAL_STATE[i])

def heuristic_manhattan(state):
    """Heuristic 2: Calculates the sum of Manhattan distances for all tiles."""
    distance = 0
    for i, val in enumerate(state):
        if val != 0:
            target_idx = GOAL_STATE.index(val)
            r1, c1 = divmod(i, 3)
            r2, c2 = divmod(target_idx, 3)
            distance += abs(r1 - r2) + abs(c1 - c2)
    return distance

def a_star_search(initial_state, heuristic_func):
    """Performs A* Search using the specified heuristic function."""
    initial_h = heuristic_func(initial_state)
    # Priority queue stores tuples of: (f_score, g_score, current_state, path)
    pq = [(initial_h, 0, initial_state, [])]
    closed_set = {}
    
    generated_nodes = 1  # Initial state is generated
    
    while pq:
        f, g, current_state, path = heapq.heappop(pq)
        
        if current_state == GOAL_STATE:
            return len(path), generated_nodes, path + [current_state]
            
        if current_state in closed_set and closed_set[current_state] <= g:
            continue
        closed_set[current_state] = g
        
        for neighbor in get_neighbors(current_state):
            generated_nodes += 1
            new_g = g + 1
            if neighbor not in closed_set or new_g < closed_set.get(neighbor, float('inf')):
                new_h = heuristic_func(neighbor)
                heapq.heappush(pq, (new_g + new_h, new_g, neighbor, path + [current_state]))
                
    return None, generated_nodes, None
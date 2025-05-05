import os
import numpy as np
import random
import pickle
import time

# === Hyperparameters ===
GRID_SIZE = 5
NUM_AGENTS = 4
ALPHA = 1e-3            # learning rate
GAMMA = 0.95           # discount factor
EPSILON_START = 1.0    # initial exploration rate
EPSILON_MIN = 0.1      # minimum exploration rate (higher for sustained exploration)
EPSILON_DECAY = 0.9999 # slower decay for prolonged exploration
MAX_EPISODE_STEPS = 25 # match evaluation horizon
STEP_BUDGET = 1_500_000  # total agent-steps budget

# === Reward weights ===
REWARD_STEP      = -0.01  # per-step penalty
REWARD_PICKUP    = +2.0  # pickup reward
REWARD_DELIVERY  = +7.0  # delivery reward (amplified)
REWARD_COLLISION = -15.0 # collision penalty
SHAPING_COEFF    = 0.2   # potential-based shaping coefficient

# === Training flags ===
USE_SENSOR = True      # disable sensor to reduce state-space initially
USE_OFFJOB_TRAINING = True
USE_CENTRAL_CLOCK = True

# Precompute all (A,B) pairs for curriculum sampling
ALL_PAIRS = [((x, y), (u, v))
             for x in range(GRID_SIZE) for y in range(GRID_SIZE)
             for u in range(GRID_SIZE) for v in range(GRID_SIZE)
             if (x, y) != (u, v)]

coverage = {pair: 0 for pair in ALL_PAIRS}

def randomize_locations():
    """
    Sample (A,B) inversely proportional to coverage for curriculum.
    """
    weights = [1.0 / (coverage[pair] + 1)**2 for pair in ALL_PAIRS]
    pair = random.choices(ALL_PAIRS, weights=weights, k=1)[0]
    coverage[pair] += 1
    return pair


class Agent:
    def __init__(self, agent_id, shared_q):
        self.id = agent_id
        self.q_table = shared_q
        self.epsilon = EPSILON_START
        self.reset(None, False)

    def reset(self, start_pos, carrying):
        self.pos = start_pos
        self.carrying = carrying
        self.last_state = None
        self.last_action = None

    def get_state(self, grid, A_loc, B_loc):
        x, y = self.pos
        c = int(self.carrying)
        mask = 0
        if USE_SENSOR:
            dirs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
            for i, (dx,dy) in enumerate(dirs):
                nx, ny = x+dx, y+dy
                if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                    cell = grid[nx][ny]
                    if cell:
                        for occ_id, occ_carry in cell:
                            if occ_id != self.id and occ_carry != c:
                                mask |= (1 << i)
                                break
        dxA, dyA = A_loc[0] - x, A_loc[1] - y
        dxB, dyB = B_loc[0] - x, B_loc[1] - y
        return (x, y, c, mask, dxA, dyA, dxB, dyB)

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(4)
        q = self.q_table.setdefault(state, np.zeros(4))
        return int(np.argmax(q))

    def update_q(self, state, action, reward, next_state):
        q = self.q_table.setdefault(state, np.zeros(4))
        q_next = self.q_table.setdefault(next_state, np.zeros(4))
        q[action] += ALPHA * (reward + GAMMA * np.max(q_next) - q[action])


def step_all_agents(agents, grid, A_loc, B_loc, train=True):
    proposals = []
    for agent in agents:
        # record old distance for shaping
        if agent.carrying:
            old_dist = abs(agent.pos[0]-B_loc[0]) + abs(agent.pos[1]-B_loc[1])
        else:
            old_dist = abs(agent.pos[0]-A_loc[0]) + abs(agent.pos[1]-A_loc[1])
        state = agent.get_state(grid, A_loc, B_loc)
        action = agent.choose_action(state)
        dx, dy = [(-1,0),(1,0),(0,-1),(0,1)][action]
        nx, ny = agent.pos[0] + dx, agent.pos[1] + dy
        if not (0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE):
            nx, ny = agent.pos
        proposals.append((agent, agent.pos, (nx, ny), action, state, old_dist, agent.carrying))

    # detect head-on collisions
    collisions = set()
    for i, (ai, old_i, new_i, _, _, _, _) in enumerate(proposals):
        for j, (aj, old_j, new_j, _, _, _, _) in enumerate(proposals):
            if i < j and old_i == new_j and old_j == new_i and ai.carrying != aj.carrying:
                if old_i not in [A_loc, B_loc] and old_j not in [A_loc, B_loc]:
                    collisions.add((ai.id, aj.id))

    new_grid = [[[] for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    total_reward = 0.0
    for agent, old_pos, new_pos, action, state, old_dist, carried_before in proposals:
        # per-step + shaping
        reward = REWARD_STEP
        if carried_before:
            new_dist = abs(new_pos[0]-B_loc[0]) + abs(new_pos[1]-B_loc[1])
        else:
            new_dist = abs(new_pos[0]-A_loc[0]) + abs(new_pos[1]-A_loc[1])
        reward += SHAPING_COEFF * (old_dist - new_dist)
        # collision penalty
        if any(agent.id in pair for pair in collisions):
            reward += REWARD_COLLISION
        agent.pos = new_pos
        # pickup/delivery
        if not agent.carrying and agent.pos == A_loc:
            agent.carrying = True
            reward += REWARD_PICKUP
        elif agent.carrying and agent.pos == B_loc:
            agent.carrying = False
            reward += REWARD_DELIVERY
        new_grid[agent.pos[0]][agent.pos[1]].append((agent.id, agent.carrying))
        next_state = agent.get_state(new_grid, A_loc, B_loc)
        if train:
            agent.update_q(state, action, reward, next_state)
        total_reward += reward
    return new_grid, len(collisions), total_reward


import time, random, pickle

def train(num_episodes=10000, collision_budget=3999, time_budget=600):
    shared_q = {}
    agents = [Agent(i, shared_q) for i in range(NUM_AGENTS)]
    total_collisions = 0
    total_steps      = 0
    start_time = time.time()
    coverage = {}

    for ep in range(1, num_episodes+1):
        ep_start_time = time.time()
        ep_collisions = 0
        ep_steps      = 0

        # 1) sample (A,B) and track coverage
        A_loc, B_loc = randomize_locations()
        coverage[(A_loc, B_loc)] = coverage.get((A_loc, B_loc), 0) + 1

        # 2) reset grid
        grid = [[[] for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]

        # 3) pick start positions
        if USE_OFFJOB_TRAINING:
            starts = [random.choice([A_loc, B_loc]) for _ in range(NUM_AGENTS)]
        else:
            starts = [A_loc] * (NUM_AGENTS//2) + [B_loc] * (NUM_AGENTS - NUM_AGENTS//2)
            random.shuffle(starts)

        # 4) reset each agent so that agent.pos is never None
        for agent, start in zip(agents, starts):
            initial_carry = (start == A_loc)
            agent.reset(start, initial_carry)
            grid[start[0]][start[1]].append((agent.id, agent.carrying))

        # --- LOGGING: Print episode and agent starts like evaluation ---
        print(f"\n[Train] Episode {ep}: A={A_loc}, B={B_loc}")
        for agent, start in zip(agents, starts):
            loc_str = "A_loc" if start == A_loc else "B_loc"
            print(f"    Agent {agent.id} starts at {loc_str} ({start})")
        # -------------------------------------------------------------

        # --- Advanced per-agent cycle step tracking (A→B→A or B→A→B) ---
        cycle_start = {ag.id: None for ag in agents}
        visited_other = {ag.id: False for ag in agents}
        completion_step = {ag.id: None for ag in agents}

        for step in range(1, MAX_EPISODE_STEPS + 1):
            ep_steps += 1
            order = agents if USE_CENTRAL_CLOCK else random.sample(agents, len(agents))
            grid, collisions, *_ = step_all_agents(order, grid, A_loc, B_loc)
            ep_collisions += collisions
            total_collisions += collisions  # <-- ADD THIS LINE

            for ag, start in zip(agents, starts):
                # Mark the start of the cycle
                if cycle_start[ag.id] is None:
                    cycle_start[ag.id] = step
                # Determine the "other" location
                other_loc = B_loc if start == A_loc else A_loc
                # Mark if agent has visited the other location
                if not visited_other[ag.id] and ag.pos == other_loc:
                    visited_other[ag.id] = True
                # If agent has returned to start after visiting other, cycle is complete
                if (visited_other[ag.id] and ag.pos == start and
                    completion_step[ag.id] is None and step > cycle_start[ag.id]):
                    completion_step[ag.id] = step - cycle_start[ag.id] + 1

            # If all agents have completed a cycle, stop early
            if all(completion_step[ag.id] is not None for ag in agents):
                break
        # -------------------------------------------------------------

        # Log per-agent cycle steps
        for ag in agents:
            s = completion_step[ag.id]
            if s is not None and s <= MAX_EPISODE_STEPS:
                print(f"  Agent {ag.id} completed cycle in {s} steps")
            elif s is not None:
                print(f"  Agent {ag.id} completed cycle in {s} steps (exceeds {MAX_EPISODE_STEPS})")
            else:
                print(f"  Agent {ag.id} did not complete a full cycle in {MAX_EPISODE_STEPS} steps")

        total_steps += ep_steps

        # 6) decay epsilon
        for agent in agents:
            agent.epsilon = max(EPSILON_MIN, agent.epsilon * EPSILON_DECAY)

        # 7) log this episode
        ep_time = time.time() - ep_start_time
        print(f"[Train] Ep {ep:5d} → steps: {ep_steps:3d}, "
              f"collisions: {ep_collisions:2d}, time: {ep_time:5.2f}s")

        # 8) optional exploration burst
        if ep % 10000 == 0 and ep <= num_episodes - 10000:
            for agent in agents:
                agent.epsilon = EPSILON_START
            print(f"*** Exploration burst at ep {ep} ***")

        if total_collisions > collision_budget:
            print(f"Collision budget exceeded at episode {ep}. Stopping training.")
            break
        if (time.time() - start_time) > time_budget:
            print(f"Time budget exceeded at episode {ep}. Stopping training.")
            break

    # end episodes loop

    # report least-trained pairs
    least = sorted(coverage.items(), key=lambda kv: kv[1])[:5]
    print("Least-trained (A,B) pairs and counts:", least)

    # save Q-table
    with open("q_table.pkl", "wb") as f:
        pickle.dump(shared_q, f)

    # final summary
    total_time = time.time() - start_time
    print(f"\nTraining complete.")
    print(f"  Total episodes run : {ep}")
    print(f"  Total agent-steps  : {total_steps}")
    print(f"  Total collisions   : {total_collisions}")
    print(f"  Total wall-time    : {total_time:.2f} seconds")

    return shared_q



def evaluate(q_table, scenarios=None, max_steps=25):
    if scenarios is None:
        scenarios = ALL_PAIRS
    total = len(scenarios)
    successes = 0
    print("Starting evaluation with per-episode logging...")

    for idx, (A_loc, B_loc) in enumerate(scenarios, start=1):
        print(f"\nEpisode {idx}/{total}: A={A_loc}, B={B_loc}")
        grid = [[[] for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        agents = [Agent(i, q_table) for i in range(NUM_AGENTS)]
        agent_starts = [random.choice([A_loc, B_loc]) for _ in range(NUM_AGENTS)]
        for agent, start in zip(agents, agent_starts):
            initial_carry = (start == A_loc)
            agent.reset(start, initial_carry)
            agent.epsilon = 0.0
            grid[start[0]][start[1]].append((agent.id, agent.carrying))

        # Log agent starting positions
        for agent, start in zip(agents, agent_starts):
            loc_str = "A_loc" if start == A_loc else "B_loc"
            print(f"    Agent {agent.id} starts at {loc_str} ({start})")

        # For each agent, track:
        # - If they started at A, when they first reach B (cycle_start)
        # - When they complete B→A→B (completion_step)
        picked_up = {ag.id: False for ag in agents}
        cycle_start = {ag.id: None for ag in agents}
        completion_step = {ag.id: None for ag in agents}
        collision_at = None

        for step in range(1, max_steps * 2 + 1):  # Allow up to 2x steps for agents starting at A
            grid, collisions, _ = step_all_agents(agents, grid, A_loc, B_loc, train=False)
            if collisions > 0:
                collision_at = step
                break
            for ag, start in zip(agents, agent_starts):
                # If agent started at A and reaches B for the first time, mark cycle_start
                if start == A_loc and cycle_start[ag.id] is None and ag.pos == B_loc:
                    cycle_start[ag.id] = step
                    picked_up[ag.id] = False  # Reset pickup for the cycle
                # If agent started at B, cycle_start is 1
                if start == B_loc:
                    cycle_start[ag.id] = 1
                # Only count pickup/delivery after cycle_start
                if cycle_start[ag.id] is not None and step >= cycle_start[ag.id]:
                    # Pickup at A
                    if not picked_up[ag.id] and ag.pos == A_loc and ag.carrying:
                        picked_up[ag.id] = True
                    # Delivery at B after pickup
                    if picked_up[ag.id] and not ag.carrying and ag.pos == B_loc and completion_step[ag.id] is None:
                        completion_step[ag.id] = step - cycle_start[ag.id] + 1  # Steps taken in cycle
            # If all agents have completed the cycle, stop
            if all(completion_step[ag.id] is not None for ag in agents):
                break

        for ag in agents:
            s = completion_step[ag.id]
            if collision_at is not None:
                print(f"  Agent {ag.id} did not finish due to collision at step {collision_at}")
            elif s is not None and s <= max_steps:
                print(f"  Agent {ag.id} succeeded in B→A→B cycle in {s} steps")
            elif s is not None:
                print(f"  Agent {ag.id} completed cycle in {s} steps (exceeds {max_steps})")
            else:
                print(f"  Agent {ag.id} failed to complete B→A→B cycle within {max_steps} steps")

        episode_success = (
            collision_at is None and
            all(completion_step[ag.id] is not None and completion_step[ag.id] <= max_steps for ag in agents)
        )
        status = "SUCCESS" if episode_success else "FAILURE"
        print(f"Episode {idx} Result: {status}")
        if episode_success:
            successes += 1

    rate = 100 * successes / total
    print(f"\nOverall success: {successes}/{total} episodes → {rate:.2f}%")
    return rate


if __name__ == '__main__':
    q_table = train()
    final_rate = evaluate(q_table)
    print(f"Final evaluation success rate: {final_rate:.2f}%")
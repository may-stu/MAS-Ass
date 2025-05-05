import os
import numpy as np
import random
import pickle
import time

# === Hyperparameters ===
GRID_SIZE       = 5
NUM_AGENTS      = 4
ALPHA           = 0.1            # learning rate
GAMMA           = 0.97           # discount factor
EPSILON_START   = 1.0            # initial exploration rate
EPSILON_MIN     = 0.1            # minimum exploration rate
EPSILON_DECAY   = 0.9999         # slower decay for prolonged exploration
MAX_EPISODE_STEPS = 25           # match evaluation horizon
STEP_BUDGET     = 1_500_000      # total agent-steps budget

# === Reward weights ===
REWARD_STEP      = -0.01          # per-step penalty
REWARD_PICKUP    = +2.0          # pickup reward
REWARD_DELIVERY  = +7.0          # delivery reward (amplified)
REWARD_COLLISION = -15.0         # collision penalty
SHAPING_COEFF    = 0.2           # potential-based shaping coefficient

# === Training flags ===
USE_SENSOR         = True       # disable sensor to reduce state-space initially
USE_OFFJOB_TRAINING = True
USE_CENTRAL_CLOCK   = True

# Precompute all (A,B) pairs for curriculum sampling
ALL_PAIRS = [((x, y), (u, v))
             for x in range(GRID_SIZE) for y in range(GRID_SIZE)
             for u in range(GRID_SIZE) for v in range(GRID_SIZE)
             if (x, y) != (u, v)]
coverage = {pair: 0 for pair in ALL_PAIRS}

def randomize_locations():
    """
    Sample (A,B) inversely proportional to coverage for curriculum sampling.
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
        if agent.carrying:
            old_dist = abs(agent.pos[0] - B_loc[0]) + abs(agent.pos[1] - B_loc[1])
        else:
            old_dist = abs(agent.pos[0] - A_loc[0]) + abs(agent.pos[1] - A_loc[1])
        state = agent.get_state(grid, A_loc, B_loc)
        action = agent.choose_action(state)
        dx, dy = [(-1,0),(1,0),(0,-1),(0,1)][action]
        nx, ny = agent.pos[0] + dx, agent.pos[1] + dy
        if not (0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE):
            nx, ny = agent.pos
        proposals.append((agent, agent.pos, (nx, ny), action, state, old_dist))

    collisions = set()
    for i, (ai, old_i, new_i, *_ ) in enumerate(proposals):
        for j, (aj, old_j, new_j, *_ ) in enumerate(proposals):
            if i < j and old_i == new_j and old_j == new_i and ai.id != aj.id:
                if old_i not in [A_loc, B_loc] and old_j not in [A_loc, B_loc]:
                    collisions.add((ai.id, aj.id))

    new_grid = [[[] for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    total_reward = 0.0
    for agent, _, new_pos, action, state, old_dist in proposals:
        reward = REWARD_STEP + SHAPING_COEFF * (old_dist - (
            abs(new_pos[0] - (B_loc[0] if agent.carrying else A_loc[0])) +
            abs(new_pos[1] - (B_loc[1] if agent.carrying else A_loc[1]))
        ))
        if any(agent.id in pair for pair in collisions):
            reward += REWARD_COLLISION
        agent.pos = new_pos
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


def train(num_episodes=100000, collision_budget=4000, time_budget=600):
    shared_q = {}
    agents = [Agent(i, shared_q) for i in range(NUM_AGENTS)]
    total_collisions = 0
    start_time = time.time()
    step_count = 0
    rewards = []

    for ep in range(1, num_episodes + 1):
        A_loc, B_loc = randomize_locations()
        grid = [[[] for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        starts = [A_loc] * (NUM_AGENTS // 2) + [B_loc] * (NUM_AGENTS - NUM_AGENTS // 2)
        random.shuffle(starts)
        for agent, start in zip(agents, starts):
            agent.reset(start, start == A_loc)
            grid[start[0]][start[1]].append((agent.id, agent.carrying))

        ep_reward = 0.0
        done = False
        for _ in range(MAX_EPISODE_STEPS):
            order = random.sample(agents, len(agents))
            grid, collisions, r = step_all_agents(order, grid, A_loc, B_loc, train=True)
            ep_reward += r
            total_collisions += collisions
            step_count += NUM_AGENTS
            if total_collisions > collision_budget or (time.time() - start_time) > time_budget or step_count >= STEP_BUDGET:
                done = True
                break
        rewards.append(ep_reward)
        for agent in agents:
            agent.epsilon = max(EPSILON_MIN, agent.epsilon * EPSILON_DECAY)
        if ep % 1000 == 0:
            print(f"[Train] Episode {ep}: avg reward = {np.mean(rewards[-1000:]):.2f}")
        if ep % 5000 == 0 and ep <= num_episodes - 5000:
            for agent in agents:
                agent.epsilon = EPSILON_START
            print(f"*** Exploration burst at ep {ep} ***")
        if done:
            print(f"Stopped training at ep {ep}")
            break

    with open("q_table.pkl", "wb") as f:
        pickle.dump(shared_q, f)
    return shared_q

def evaluate(q_table, scenarios=None, max_steps=25):
    scenarios = scenarios or ALL_PAIRS
    successes = 0
    for idx, (A_loc, B_loc) in enumerate(scenarios, start=1):
        grid = [[[] for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        agents = [Agent(i, q_table) for i in range(NUM_AGENTS)]
        for agent in agents:
            agent.reset(B_loc, False)
            agent.epsilon = 0.0
            grid[B_loc[0]][B_loc[1]].append((agent.id, agent.carrying))
        collision_at = None
        completion = {ag.id: False for ag in agents}
        for step in range(max_steps):
            grid, collisions, _ = step_all_agents(agents, grid, A_loc, B_loc, train=False)
            if collisions:
                collision_at = step + 1
                break
            for ag in agents:
                if not ag.carrying and ag.pos == B_loc:
                    completion[ag.id] = True
            if all(completion.values()):
                break
        if not collision_at and all(completion.values()):
            successes += 1
    rate = 100 * successes / len(scenarios)
    print(f"Final evaluation success rate: {rate:.2f}%")
    return rate


if __name__ == '__main__':
    q_table = train()
    final_rate = evaluate(q_table)
    print(f"Final evaluation success rate: {final_rate:.2f}%")
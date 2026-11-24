import asyncio
import heapq
import math
import random
import pygame

COLS = 19
ROWS = 15
CELL_SIZE = 44
MAZE_W = COLS * CELL_SIZE
MAZE_H = ROWS * CELL_SIZE
SIDEBAR_W = 356
WIDTH = MAZE_W + SIDEBAR_W
HEIGHT = MAZE_H
FPS = 60

DIRS = {
    "U": (0, -1),
    "D": (0, 1),
    "L": (-1, 0),
    "R": (1, 0),
}
OPPOSITE = {"U": "D", "D": "U", "L": "R", "R": "L"}
DIR_LIST = ["U", "D", "L", "R"]

BG_DARK = (14, 15, 22)
FLOOR_COL = (28, 30, 42)
FLOOR_ALT = (33, 36, 50)
WALL_COL = (72, 78, 104)
PANEL_BG = (19, 21, 30)
PANEL_BORDER = (52, 57, 78)
TEXT_MAIN = (232, 236, 245)
TEXT_DIM = (148, 156, 180)
AMBER = (255, 198, 72)
CRIMSON = (235, 64, 84)
CYAN = (76, 214, 235)
EMERALD = (68, 224, 148)
PURPLE = (168, 108, 240)
YELLOW = (248, 222, 82)


class Cell:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.walls = {"U": True, "D": True, "L": True, "R": True}
        self.visited = False


class Maze:
    def __init__(self, cols, rows):
        self.cols = cols
        self.rows = rows
        self.grid = [[Cell(x, y) for y in range(rows)] for x in range(cols)]
        self.generate()

    def in_bounds(self, x, y):
        return 0 <= x < self.cols and 0 <= y < self.rows

    def generate(self):
        for x in range(self.cols):
            for y in range(self.rows):
                self.grid[x][y] = Cell(x, y)

        start = self.grid[0][0]
        start.visited = True
        stack = [start]

        while stack:
            current = stack[-1]
            neighbors = []
            for d, (dx, dy) in DIRS.items():
                nx, ny = current.x + dx, current.y + dy
                if self.in_bounds(nx, ny) and not self.grid[nx][ny].visited:
                    neighbors.append((d, self.grid[nx][ny]))

            if neighbors:
                direction, nxt = random.choice(neighbors)
                current.walls[direction] = False
                nxt.walls[OPPOSITE[direction]] = False
                nxt.visited = True
                stack.append(nxt)
            else:
                stack.pop()

        braid_count = int(self.cols * self.rows * 0.14)
        for _ in range(braid_count):
            x = random.randint(1, self.cols - 2)
            y = random.randint(1, self.rows - 2)
            cell = self.grid[x][y]
            closed = [d for d, w in cell.walls.items() if w]
            if closed:
                d = random.choice(closed)
                dx, dy = DIRS[d]
                nxt = self.grid[x + dx][y + dy]
                cell.walls[d] = False
                nxt.walls[OPPOSITE[d]] = False

    def neighbors(self, pos):
        x, y = pos
        cell = self.grid[x][y]
        result = []
        for d, (dx, dy) in DIRS.items():
            if not cell.walls[d]:
                nx, ny = x + dx, y + dy
                if self.in_bounds(nx, ny):
                    result.append((nx, ny))
        return result

    def can_move(self, pos, direction):
        x, y = pos
        if not self.in_bounds(x, y):
            return False
        if self.grid[x][y].walls[direction]:
            return False
        dx, dy = DIRS[direction]
        return self.in_bounds(x + dx, y + dy)

    def has_line_of_sight(self, a, b):
        ax, ay = a
        bx, by = b
        if ax == bx:
            step = 1 if by > ay else -1
            direction = "D" if step == 1 else "U"
            cy = ay
            while cy != by:
                if self.grid[ax][cy].walls[direction]:
                    return False
                cy += step
            return True
        if ay == by:
            step = 1 if bx > ax else -1
            direction = "R" if step == 1 else "L"
            cx = ax
            while cx != bx:
                if self.grid[cx][ay].walls[direction]:
                    return False
                cx += step
            return True
        return False


class AStarPathfinder:
    @staticmethod
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    @staticmethod
    def find_path(maze, start, goal, cost_map=None):
        if start == goal:
            return [], 0.0, 1

        open_heap = []
        heapq.heappush(open_heap, (0.0, 0.0, start))
        came_from = {}
        g_score = {start: 0.0}
        closed = set()
        nodes_explored = 0

        while open_heap:
            _, _, current = heapq.heappop(open_heap)
            if current in closed:
                continue
            closed.add(current)
            nodes_explored += 1

            if current == goal:
                path = []
                node = current
                while node in came_from:
                    path.append(node)
                    node = came_from[node]
                path.reverse()
                return path, g_score[goal], nodes_explored

            for nxt in maze.neighbors(current):
                if nxt in closed:
                    continue
                step_cost = cost_map.get(nxt, 1.0) if cost_map else 1.0
                tentative_g = g_score[current] + step_cost
                if tentative_g < g_score.get(nxt, float("inf")):
                    came_from[nxt] = current
                    g_score[nxt] = tentative_g
                    f_score = tentative_g + AStarPathfinder.heuristic(nxt, goal)
                    heapq.heappush(open_heap, (f_score, tentative_g, nxt))

        return [], float("inf"), nodes_explored


class MarkovPredictor:
    def __init__(self):
        self.counts = {d: {t: 0 for t in DIR_LIST} for d in DIR_LIST}
        self.last_dir = None
        self.predicted_dir = None
        self.predicted_prob = 0.0
        self.total_transitions = 0
        self.total_predictions = 0
        self.correct_predictions = 0

    def record_move(self, direction):
        if self.predicted_dir is not None:
            self.total_predictions += 1
            if direction == self.predicted_dir:
                self.correct_predictions += 1

        if self.last_dir is not None:
            self.counts[self.last_dir][direction] += 1
            self.total_transitions += 1

        self.last_dir = direction
        self.predicted_dir, self.predicted_prob = self.predict_next(direction)

    def get_distribution(self, from_dir):
        if from_dir is None:
            return {d: 0.25 for d in DIR_LIST}
        row = self.counts[from_dir]
        total = sum(row.values())
        if total == 0:
            return {d: 0.25 for d in DIR_LIST}
        return {d: row[d] / total for d in DIR_LIST}

    def predict_next(self, from_dir):
        dist = self.get_distribution(from_dir)
        best_dir = max(DIR_LIST, key=lambda d: dist[d])
        return best_dir, dist[best_dir]

    def accuracy(self):
        if self.total_predictions == 0:
            return 0.0
        return (self.correct_predictions / self.total_predictions) * 100.0


class Player:
    def __init__(self, start_pos):
        self.grid_pos = start_pos
        self.visual_pos = [float(start_pos[0]), float(start_pos[1])]
        self.hp = 100
        self.max_hp = 100
        self.score = 0
        self.lanterns_collected = 0
        self.move_cooldown = 0.0
        self.slow_timer = 0.0
        self.root_timer = 0.0
        self.invuln_timer = 0.0
        self.facing = "R"

    def update(self, dt):
        self.move_cooldown = max(0.0, self.move_cooldown - dt)
        self.slow_timer = max(0.0, self.slow_timer - dt)
        self.root_timer = max(0.0, self.root_timer - dt)
        self.invuln_timer = max(0.0, self.invuln_timer - dt)

        speed = 14.0 * dt
        self.visual_pos[0] += (self.grid_pos[0] - self.visual_pos[0]) * min(1.0, speed)
        self.visual_pos[1] += (self.grid_pos[1] - self.visual_pos[1]) * min(1.0, speed)

    def try_move(self, direction, maze):
        if self.move_cooldown > 0.0 or self.root_timer > 0.0:
            return False
        self.facing = direction
        if maze.can_move(self.grid_pos, direction):
            dx, dy = DIRS[direction]
            self.grid_pos = (self.grid_pos[0] + dx, self.grid_pos[1] + dy)
            self.move_cooldown = 0.25 if self.slow_timer > 0.0 else 0.13
            return True
        return False


class GhostNPC:
    def __init__(self, start_pos, name="Wisp"):
        self.name = name
        self.grid_pos = start_pos
        self.visual_pos = [float(start_pos[0]), float(start_pos[1])]
        self.state = "PATROL"
        self.target = start_pos
        self.path = []
        self.path_cost = 0.0
        self.nodes_explored = 0
        self.last_known_player = None
        self.search_timer = 0.0
        self.move_timer = 0.0
        self.replan_timer = 0.0

    def pick_patrol_target(self, maze):
        x = random.randint(0, maze.cols - 1)
        y = random.randint(0, maze.rows - 1)
        return (x, y)

    def update(self, dt, maze, player, cost_map):
        self.move_timer -= dt
        self.replan_timer -= dt

        dist = AStarPathfinder.heuristic(self.grid_pos, player.grid_pos)
        can_see = (dist <= 6 and maze.has_line_of_sight(self.grid_pos, player.grid_pos)) or dist <= 3

        if self.state == "PATROL":
            if can_see:
                self.state = "CHASE"
                self.replan_timer = 0.0
            elif not self.path or self.grid_pos == self.target:
                self.target = self.pick_patrol_target(maze)
                self.path, self.path_cost, self.nodes_explored = AStarPathfinder.find_path(
                    maze, self.grid_pos, self.target, cost_map
                )

        elif self.state == "CHASE":
            if can_see or dist <= 4:
                self.last_known_player = player.grid_pos
                if self.replan_timer <= 0.0:
                    self.target = player.grid_pos
                    self.path, self.path_cost, self.nodes_explored = AStarPathfinder.find_path(
                        maze, self.grid_pos, self.target, cost_map
                    )
                    self.replan_timer = 0.42
            else:
                self.state = "SEARCH"
                self.search_timer = 3.2
                self.target = self.last_known_player or player.grid_pos
                self.path, self.path_cost, self.nodes_explored = AStarPathfinder.find_path(
                    maze, self.grid_pos, self.target, cost_map
                )

        elif self.state == "SEARCH":
            self.search_timer -= dt
            if can_see:
                self.state = "CHASE"
                self.replan_timer = 0.0
            elif self.search_timer <= 0.0:
                self.state = "PATROL"
                self.path = []
            elif not self.path:
                adj = maze.neighbors(self.grid_pos)
                if adj:
                    self.target = random.choice(adj)
                    self.path, self.path_cost, self.nodes_explored = AStarPathfinder.find_path(
                        maze, self.grid_pos, self.target, cost_map
                    )

        step_delay = 0.27 if self.state == "CHASE" else 0.36
        if self.move_timer <= 0.0 and self.path:
            self.grid_pos = self.path.pop(0)
            self.move_timer = step_delay

        lerp = min(1.0, 10.0 * dt)
        self.visual_pos[0] += (self.grid_pos[0] - self.visual_pos[0]) * lerp
        self.visual_pos[1] += (self.grid_pos[1] - self.visual_pos[1]) * lerp


class BossNPC:
    def __init__(self, start_pos):
        self.grid_pos = start_pos
        self.visual_pos = [float(start_pos[0]), float(start_pos[1])]
        self.state = "AMBUSH"
        self.target = start_pos
        self.path = []
        self.path_cost = 0.0
        self.nodes_explored = 0
        self.replan_interval = 0.48
        self.replan_timer = 0.0
        self.move_timer = 0.0
        self.predicted_cell = start_pos
        self.candidates = []
        self.decision_reason = "Initializing"
        self.trail = []

    def project_player_path(self, maze, player_pos, markov, steps=4):
        dist = markov.get_distribution(markov.last_dir)
        ordered_dirs = sorted(DIR_LIST, key=lambda d: dist[d], reverse=True)
        curr = player_pos
        projected = []
        cumulative_prob = 1.0

        for _ in range(steps):
            moved = False
            for d in ordered_dirs:
                if maze.can_move(curr, d):
                    dx, dy = DIRS[d]
                    nxt = (curr[0] + dx, curr[1] + dy)
                    if nxt not in projected:
                        p = max(0.15, dist[d])
                        cumulative_prob = max(0.15, cumulative_prob * (0.55 + 0.45 * p))
                        projected.append((nxt, cumulative_prob))
                        curr = nxt
                        moved = True
                        break
            if not moved:
                break

        return projected

    def evaluate_ambush(self, maze, player, markov, cost_map):
        dist_to_player = AStarPathfinder.heuristic(self.grid_pos, player.grid_pos)
        if dist_to_player <= 2:
            self.state = "STRIKE"
            self.target = player.grid_pos
            self.predicted_cell = player.grid_pos
            self.path, self.path_cost, self.nodes_explored = AStarPathfinder.find_path(
                maze, self.grid_pos, self.target, cost_map
            )
            self.candidates = [
                {
                    "cell": player.grid_pos,
                    "prob": 1.0,
                    "cost": self.path_cost,
                    "utility": 100.0 - 4.5 * self.path_cost,
                    "selected": True,
                }
            ]
            self.decision_reason = "Within strike range"
            return

        projected = self.project_player_path(maze, player.grid_pos, markov, steps=4)
        candidate_probs = {}

        if projected:
            self.predicted_cell = projected[0][0]
            for cell, prob in projected:
                candidate_probs[cell] = max(candidate_probs.get(cell, 0.0), prob)
            anchor = projected[-1][0]
            for nb in maze.neighbors(anchor):
                if nb != player.grid_pos and nb not in candidate_probs:
                    candidate_probs[nb] = projected[-1][1] * 0.72
        else:
            self.predicted_cell = player.grid_pos

        if not candidate_probs:
            candidate_probs[player.grid_pos] = 0.5

        evaluated = []
        total_explored = 0
        w_prob = 100.0
        w_dist = 4.5

        for cell, prob in list(candidate_probs.items())[:5]:
            path, cost, explored = AStarPathfinder.find_path(maze, self.grid_pos, cell, cost_map)
            total_explored += explored
            utility = (w_prob * prob) - (w_dist * cost)
            evaluated.append(
                {
                    "cell": cell,
                    "prob": prob,
                    "cost": cost,
                    "utility": utility,
                    "path": path,
                    "selected": False,
                }
            )

        evaluated.sort(key=lambda item: item["utility"], reverse=True)
        best = evaluated[0]

        if best["utility"] < -15.0:
            self.state = "HUNT"
            path, cost, explored = AStarPathfinder.find_path(maze, self.grid_pos, player.grid_pos, cost_map)
            self.target = player.grid_pos
            self.path = path
            self.path_cost = cost
            self.nodes_explored = total_explored + explored
            self.decision_reason = "All ambush utilities too low; direct hunt"
        else:
            self.state = "AMBUSH"
            best["selected"] = True
            self.target = best["cell"]
            self.path = best["path"]
            self.path_cost = best["cost"]
            self.nodes_explored = total_explored
            self.decision_reason = (
                f"Best utility {best['utility']:.1f} (P={best['prob']*100:.0f}%, C={best['cost']:.0f})"
            )

        self.candidates = evaluated

    def update(self, dt, maze, player, markov, cost_map, force_replan=False):
        self.replan_timer -= dt
        self.move_timer -= dt

        if force_replan or self.replan_timer <= 0.0:
            self.evaluate_ambush(maze, player, markov, cost_map)
            self.replan_timer = self.replan_interval

        step_delay = 0.24 if self.state == "STRIKE" else 0.28
        if self.move_timer <= 0.0 and self.path:
            self.grid_pos = self.path.pop(0)
            self.move_timer = step_delay

        lerp = min(1.0, 11.0 * dt)
        self.visual_pos[0] += (self.grid_pos[0] - self.visual_pos[0]) * lerp
        self.visual_pos[1] += (self.grid_pos[1] - self.visual_pos[1]) * lerp

        px = self.visual_pos[0] * CELL_SIZE + CELL_SIZE / 2
        py = self.visual_pos[1] * CELL_SIZE + CELL_SIZE / 2
        self.trail.append((px, py, 1.0))
        self.trail = [(x, y, a - dt * 2.2) for x, y, a in self.trail if a - dt * 2.2 > 0]


class Game:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Lanterns of Hollowkeep — Haunted Maze AI")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()

        self.font_xs = pygame.font.SysFont("consolas", 12)
        self.font_sm = pygame.font.SysFont("consolas", 14)
        self.font_md = pygame.font.SysFont("consolas", 18, bold=True)
        self.font_lg = pygame.font.SysFont("georgia", 32, bold=True)
        self.font_xl = pygame.font.SysFont("georgia", 44, bold=True)

        self.probs = {
            "lantern": 0.08,
            "elixir": 0.05,
            "trap": 0.07,
            "web": 0.10,
        }
        self.slider_rects = {}
        self.start_btn_rect = pygame.Rect(WIDTH // 2 - 120, HEIGHT - 110, 240, 52)

        self.running = True
        self.mode = "MENU"
        self.debug_mode = True
        self.elapsed = 0.0
        self.status_banner = ""
        self.banner_timer = 0.0
        self.particles = []

        self.maze = None
        self.player = None
        self.markov = None
        self.ghosts = []
        self.boss = None
        self.exit_pos = (COLS - 1, ROWS - 1)
        self.lanterns = set()
        self.total_lanterns = 0
        self.elixirs = set()
        self.traps = set()
        self.webs = set()
        self.cost_map = {}

    def start_new_game(self):
        self.maze = Maze(COLS, ROWS)
        self.player = Player((0, 0))
        self.markov = MarkovPredictor()
        self.exit_pos = (COLS - 1, ROWS - 1)
        self.elapsed = 0.0
        self.particles = []
        self.status_banner = "Collect all Soul Lanterns to unseal the Exit"
        self.banner_timer = 3.5

        self.lanterns = set()
        self.elixirs = set()
        self.traps = set()
        self.webs = set()

        candidates = [
            (x, y)
            for x in range(COLS)
            for y in range(ROWS)
            if (x, y) not in {(0, 0), (1, 0), (0, 1), self.exit_pos}
        ]
        random.shuffle(candidates)

        for pos in candidates:
            roll = random.random()
            p1 = self.probs["lantern"]
            p2 = p1 + self.probs["elixir"]
            p3 = p2 + self.probs["trap"]
            p4 = p3 + self.probs["web"]
            if roll < p1:
                self.lanterns.add(pos)
            elif roll < p2:
                self.elixirs.add(pos)
            elif roll < p3:
                self.traps.add(pos)
            elif roll < p4:
                self.webs.add(pos)

        while len(self.lanterns) < 4:
            pos = random.choice(candidates)
            self.elixirs.discard(pos)
            self.traps.discard(pos)
            self.webs.discard(pos)
            self.lanterns.add(pos)

        self.total_lanterns = len(self.lanterns)
        self.cost_map = {pos: 2.0 for pos in self.webs}

        far_ghost = [p for p in candidates if AStarPathfinder.heuristic((0, 0), p) >= 9]
        far_boss = [p for p in candidates if AStarPathfinder.heuristic((0, 0), p) >= 13]

        g1_pos = random.choice(far_ghost) if far_ghost else (COLS // 2, ROWS // 2)
        g2_pool = [p for p in far_ghost if AStarPathfinder.heuristic(g1_pos, p) >= 6]
        g2_pos = random.choice(g2_pool) if g2_pool else (COLS - 2, 1)
        boss_pos = random.choice(far_boss) if far_boss else (COLS - 1, ROWS - 2)

        self.ghosts = [GhostNPC(g1_pos, "Wisp A"), GhostNPC(g2_pos, "Wisp B")]
        self.boss = BossNPC(boss_pos)
        self.boss.evaluate_ambush(self.maze, self.player, self.markov, self.cost_map)
        self.mode = "PLAYING"

    def spawn_particles(self, grid_pos, color, count=14):
        cx = grid_pos[0] * CELL_SIZE + CELL_SIZE / 2
        cy = grid_pos[1] * CELL_SIZE + CELL_SIZE / 2
        for _ in range(count):
            ang = random.uniform(0, math.tau)
            spd = random.uniform(30, 110)
            self.particles.append([cx, cy, math.cos(ang) * spd, math.sin(ang) * spd, 0.55, color])

    def set_banner(self, text, duration=2.2):
        self.status_banner = text
        self.banner_timer = duration

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if self.mode != "MENU":
                        self.mode = "MENU"
                    else:
                        self.running = False
                        return

                if self.mode == "MENU":
                    if event.key in (pygame.K_RETURN, pygame.K_SPACE):
                        self.start_new_game()

                elif self.mode == "PLAYING":
                    if event.key in (pygame.K_F1, pygame.K_TAB):
                        self.debug_mode = not self.debug_mode
                    elif event.key == pygame.K_r:
                        self.start_new_game()
                    elif event.key == pygame.K_m:
                        self.mode = "MENU"

                elif self.mode in ("WIN", "LOSE"):
                    if event.key == pygame.K_r:
                        self.start_new_game()
                    elif event.key == pygame.K_m:
                        self.mode = "MENU"

            if self.mode == "MENU" and event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                if self.start_btn_rect.collidepoint(mx, my):
                    self.start_new_game()
                for key, (minus_r, plus_r) in self.slider_rects.items():
                    if minus_r.collidepoint(mx, my):
                        self.probs[key] = max(0.01 if key == "lantern" else 0.0, round(self.probs[key] - 0.02, 2))
                    elif plus_r.collidepoint(mx, my):
                        self.probs[key] = min(0.25, round(self.probs[key] + 0.02, 2))

    def update_playing(self, dt):
        self.elapsed += dt
        self.banner_timer = max(0.0, self.banner_timer - dt)

        keys = pygame.key.get_pressed()
        move_dir = None
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            move_dir = "U"
        elif keys[pygame.K_s] or keys[pygame.K_DOWN]:
            move_dir = "D"
        elif keys[pygame.K_a] or keys[pygame.K_LEFT]:
            move_dir = "L"
        elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            move_dir = "R"

        player_moved = False
        if move_dir and self.player.try_move(move_dir, self.maze):
            self.markov.record_move(move_dir)
            player_moved = True

        self.player.update(dt)

        pos = self.player.grid_pos
        if pos in self.lanterns:
            self.lanterns.remove(pos)
            self.player.lanterns_collected += 1
            self.player.score += 100
            self.spawn_particles(pos, AMBER, 16)
            if not self.lanterns:
                self.set_banner("All Soul Lanterns claimed! Reach the Exit Portal!", 4.0)

        if pos in self.elixirs:
            self.elixirs.remove(pos)
            self.player.hp = min(self.player.max_hp, self.player.hp + 25)
            self.player.score += 40
            self.spawn_particles(pos, EMERALD, 14)
            self.set_banner("Blood Elixir restored +25 HP", 1.8)

        if pos in self.traps and self.player.invuln_timer <= 0.0:
            self.traps.remove(pos)
            self.player.hp = max(0, self.player.hp - 20)
            self.player.slow_timer = 2.0
            self.player.invuln_timer = 0.8
            self.spawn_particles(pos, CRIMSON, 18)
            self.set_banner("Bone Trap! -20 HP & Slowed", 1.8)

        if pos in self.webs and player_moved:
            self.webs.remove(pos)
            self.cost_map.pop(pos, None)
            self.player.root_timer = 0.45
            self.spawn_particles(pos, PURPLE, 12)
            self.set_banner("Snared in Shadow Web!", 1.2)

        for ghost in self.ghosts:
            ghost.update(dt, self.maze, self.player, self.cost_map)
            if ghost.grid_pos == self.player.grid_pos and self.player.invuln_timer <= 0.0:
                self.player.hp = max(0, self.player.hp - 15)
                self.player.invuln_timer = 1.1
                self.spawn_particles(self.player.grid_pos, CYAN, 18)
                self.set_banner(f"{ghost.name} struck you! -15 HP", 1.6)

        self.boss.update(dt, self.maze, self.player, self.markov, self.cost_map, force_replan=player_moved)
        if self.boss.grid_pos == self.player.grid_pos:
            self.player.hp = 0
            self.mode = "LOSE"
            return

        if self.player.hp <= 0:
            self.mode = "LOSE"
            return

        if not self.lanterns and self.player.grid_pos == self.exit_pos:
            self.player.score += int(self.player.hp * 5)
            self.mode = "WIN"
            return

        next_particles = []
        for p in self.particles:
            p[0] += p[2] * dt
            p[1] += p[3] * dt
            p[4] -= dt
            if p[4] > 0:
                next_particles.append(p)
        self.particles = next_particles

    def cell_center(self, pos):
        return (int(pos[0] * CELL_SIZE + CELL_SIZE / 2), int(pos[1] * CELL_SIZE + CELL_SIZE / 2))

    def draw_maze(self):
        for x in range(COLS):
            for y in range(ROWS):
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                col = FLOOR_COL if (x + y) % 2 == 0 else FLOOR_ALT
                pygame.draw.rect(self.screen, col, rect)

        for pos in self.webs:
            cx, cy = self.cell_center(pos)
            pygame.draw.circle(self.screen, (52, 34, 72), (cx, cy), CELL_SIZE // 2 - 4)
            for ang in range(0, 360, 45):
                rad = math.radians(ang)
                ex = cx + int(math.cos(rad) * (CELL_SIZE // 2 - 5))
                ey = cy + int(math.sin(rad) * (CELL_SIZE // 2 - 5))
                pygame.draw.line(self.screen, (130, 96, 175), (cx, cy), (ex, ey), 1)
            pygame.draw.circle(self.screen, (145, 110, 195), (cx, cy), 9, 1)

        for pos in self.traps:
            cx, cy = self.cell_center(pos)
            pts = [
                (cx, cy - 11),
                (cx + 9, cy + 8),
                (cx - 9, cy + 8),
            ]
            pygame.draw.polygon(self.screen, (185, 52, 72), pts)
            pygame.draw.polygon(self.screen, (240, 140, 150), pts, 2)

        pulse = 0.5 + 0.5 * math.sin(self.elapsed * 5.0)
        for pos in self.lanterns:
            cx, cy = self.cell_center(pos)
            glow_r = int(12 + pulse * 4)
            glow = pygame.Surface((glow_r * 2 + 4, glow_r * 2 + 4), pygame.SRCALPHA)
            pygame.draw.circle(glow, (255, 195, 70, 75), (glow_r + 2, glow_r + 2), glow_r)
            self.screen.blit(glow, (cx - glow_r - 2, cy - glow_r - 2))
            pygame.draw.polygon(
                self.screen,
                AMBER,
                [(cx, cy - 9), (cx + 7, cy), (cx, cy + 9), (cx - 7, cy)],
            )
            pygame.draw.circle(self.screen, (255, 245, 210), (cx, cy), 3)

        for pos in self.elixirs:
            cx, cy = self.cell_center(pos)
            pygame.draw.circle(self.screen, EMERALD, (cx, cy + 2), 7)
            pygame.draw.rect(self.screen, (200, 255, 230), (cx - 3, cy - 9, 6, 5))

        ex, ey = self.cell_center(self.exit_pos)
        unlocked = len(self.lanterns) == 0
        exit_col = EMERALD if unlocked else (95, 92, 115)
        pygame.draw.circle(self.screen, exit_col, (ex, ey), int(14 + (pulse * 3 if unlocked else 0)), 3)
        pygame.draw.circle(self.screen, exit_col, (ex, ey), 7, 0 if unlocked else 2)

        for x in range(COLS):
            for y in range(ROWS):
                cell = self.maze.grid[x][y]
                px, py = x * CELL_SIZE, y * CELL_SIZE
                if cell.walls["U"]:
                    pygame.draw.line(self.screen, WALL_COL, (px, py), (px + CELL_SIZE, py), 3)
                if cell.walls["D"]:
                    pygame.draw.line(self.screen, WALL_COL, (px, py + CELL_SIZE), (px + CELL_SIZE, py + CELL_SIZE), 3)
                if cell.walls["L"]:
                    pygame.draw.line(self.screen, WALL_COL, (px, py), (px, py + CELL_SIZE), 3)
                if cell.walls["R"]:
                    pygame.draw.line(self.screen, WALL_COL, (px + CELL_SIZE, py), (px + CELL_SIZE, py + CELL_SIZE), 3)

    def draw_debug_overlay(self):
        for cand in self.boss.candidates:
            cx, cy = cand["cell"]
            rect = pygame.Rect(cx * CELL_SIZE + 4, cy * CELL_SIZE + 4, CELL_SIZE - 8, CELL_SIZE - 8)
            surf = pygame.Surface((rect.w, rect.h), pygame.SRCALPHA)
            if cand["selected"]:
                surf.fill((255, 215, 64, 95))
                border = YELLOW
            else:
                surf.fill((255, 160, 64, 48))
                border = (210, 145, 70)
            self.screen.blit(surf, rect.topleft)
            pygame.draw.rect(self.screen, border, rect, 2)
            u_txt = self.font_xs.render(f"{cand['utility']:.0f}", True, TEXT_MAIN)
            self.screen.blit(u_txt, (rect.x + 4, rect.y + 4))

        if self.boss.predicted_cell:
            px, py = self.cell_center(self.boss.predicted_cell)
            pygame.draw.circle(self.screen, YELLOW, (px, py), 15, 2)

        for ghost in self.ghosts:
            if ghost.path:
                pts = [self.cell_center(ghost.grid_pos)] + [self.cell_center(p) for p in ghost.path]
                pygame.draw.lines(self.screen, EMERALD, False, pts, 2)
                tx, ty = self.cell_center(ghost.target)
                pygame.draw.circle(self.screen, EMERALD, (tx, ty), 6, 2)

        if self.boss.path:
            pts = [self.cell_center(self.boss.grid_pos)] + [self.cell_center(p) for p in self.boss.path]
            for i in range(len(pts) - 1):
                pygame.draw.line(self.screen, CRIMSON, pts[i], pts[i + 1], 3)
            tx, ty = self.cell_center(self.boss.target)
            pygame.draw.rect(self.screen, CRIMSON, (tx - 7, ty - 7, 14, 14), 2)

    def draw_entities(self):
        for x, y, a in self.boss.trail:
            s = pygame.Surface((16, 16), pygame.SRCALPHA)
            pygame.draw.circle(s, (235, 64, 84, int(110 * a)), (8, 8), max(2, int(7 * a)))
            self.screen.blit(s, (int(x) - 8, int(y) - 8))

        px = int(self.player.visual_pos[0] * CELL_SIZE + CELL_SIZE / 2)
        py = int(self.player.visual_pos[1] * CELL_SIZE + CELL_SIZE / 2)
        light = pygame.Surface((110, 110), pygame.SRCALPHA)
        pygame.draw.circle(light, (255, 200, 90, 38), (55, 55), 52)
        self.screen.blit(light, (px - 55, py - 55))
        player_col = (255, 255, 255) if self.player.invuln_timer > 0 and int(self.elapsed * 16) % 2 == 0 else AMBER
        pygame.draw.circle(self.screen, player_col, (px, py), 12)
        pygame.draw.circle(self.screen, (40, 28, 12), (px, py), 12, 2)
        dx, dy = DIRS[self.player.facing]
        pygame.draw.circle(self.screen, (255, 250, 220), (px + dx * 6, py + dy * 6), 3)

        for ghost in self.ghosts:
            gx = int(ghost.visual_pos[0] * CELL_SIZE + CELL_SIZE / 2)
            gy = int(ghost.visual_pos[1] * CELL_SIZE + CELL_SIZE / 2)
            g_col = CRIMSON if ghost.state == "CHASE" else YELLOW if ghost.state == "SEARCH" else CYAN
            surf = pygame.Surface((32, 32), pygame.SRCALPHA)
            pygame.draw.circle(surf, (*g_col, 190), (16, 14), 11)
            pygame.draw.polygon(surf, (*g_col, 190), [(5, 14), (27, 14), (25, 27), (19, 22), (16, 27), (12, 22), (7, 27)])
            self.screen.blit(surf, (gx - 16, gy - 16))
            pygame.draw.circle(self.screen, BG_DARK, (gx - 4, gy - 3), 2)
            pygame.draw.circle(self.screen, BG_DARK, (gx + 4, gy - 3), 2)

            if self.debug_mode:
                tag = self.font_xs.render(f"{ghost.name}:{ghost.state}", True, g_col)
                bg = pygame.Rect(gx - tag.get_width() // 2 - 3, gy - 28, tag.get_width() + 6, 15)
                pygame.draw.rect(self.screen, (10, 12, 18), bg)
                self.screen.blit(tag, (bg.x + 3, bg.y + 1))

        bx = int(self.boss.visual_pos[0] * CELL_SIZE + CELL_SIZE / 2)
        by = int(self.boss.visual_pos[1] * CELL_SIZE + CELL_SIZE / 2)
        pygame.draw.polygon(
            self.screen,
            CRIMSON,
            [(bx, by - 15), (bx + 14, by + 11), (bx, by + 6), (bx - 14, by + 11)],
        )
        pygame.draw.polygon(
            self.screen,
            (255, 185, 195),
            [(bx, by - 15), (bx + 14, by + 11), (bx, by + 6), (bx - 14, by + 11)],
            2,
        )
        pygame.draw.circle(self.screen, YELLOW, (bx - 4, by - 2), 2)
        pygame.draw.circle(self.screen, YELLOW, (bx + 4, by - 2), 2)

        if self.debug_mode:
            btag = self.font_xs.render(f"BOSS:{self.boss.state} -> {self.boss.target}", True, CRIMSON)
            bg = pygame.Rect(bx - btag.get_width() // 2 - 3, by - 31, btag.get_width() + 6, 15)
            pygame.draw.rect(self.screen, (10, 12, 18), bg)
            self.screen.blit(btag, (bg.x + 3, bg.y + 1))

        for x, y, _, _, life, col in self.particles:
            pygame.draw.circle(self.screen, col, (int(x), int(y)), max(1, int(4 * life)))

    def draw_sidebar(self):
        panel = pygame.Rect(MAZE_W, 0, SIDEBAR_W, HEIGHT)
        pygame.draw.rect(self.screen, PANEL_BG, panel)
        pygame.draw.line(self.screen, PANEL_BORDER, (MAZE_W, 0), (MAZE_W, HEIGHT), 2)

        x = MAZE_W + 16
        y = 14

        title = self.font_md.render("LANTERNS OF HOLLOWKEEP", True, AMBER)
        self.screen.blit(title, (x, y))
        y += 28

        hp_ratio = self.player.hp / self.player.max_hp
        pygame.draw.rect(self.screen, (42, 44, 58), (x, y, SIDEBAR_W - 32, 16), border_radius=4)
        pygame.draw.rect(
            self.screen,
            EMERALD if hp_ratio > 0.45 else CRIMSON,
            (x, y, int((SIDEBAR_W - 32) * hp_ratio), 16),
            border_radius=4,
        )
        hp_txt = self.font_xs.render(f"HP: {self.player.hp}/{self.player.max_hp}", True, TEXT_MAIN)
        self.screen.blit(hp_txt, (x + 8, y + 2))
        y += 26

        stats = [
            f"Lanterns: {self.player.lanterns_collected}/{self.total_lanterns}   Score: {self.player.score}",
            f"Time: {self.elapsed:05.1f}s   Debug [F1/TAB]: {'ON' if self.debug_mode else 'OFF'}",
            "Controls: WASD/Arrows | R: Reset | M: Menu",
        ]
        for line in stats:
            self.screen.blit(self.font_xs.render(line, True, TEXT_DIM), (x, y))
            y += 17

        y += 6
        pygame.draw.line(self.screen, PANEL_BORDER, (x, y), (WIDTH - 16, y), 1)
        y += 10

        if not self.debug_mode:
            hint = self.font_sm.render("Press F1 or TAB for AI Debug Telemetry", True, CYAN)
            self.screen.blit(hint, (x, y))
            y += 28
            legend = [
                (AMBER, "Soul Lantern — Collect all to unlock exit"),
                (EMERALD, "Blood Elixir — Restores +25 HP"),
                (CRIMSON, "Bone Trap — -20 HP and slows"),
                (PURPLE, "Shadow Web — Snares & +1 A* terrain cost"),
                (CYAN, "Wisp Stalker — 3-State FSM + A*"),
                (CRIMSON, "The Weaver (Boss) — Markov Ambush AI"),
            ]
            for col, label in legend:
                pygame.draw.circle(self.screen, col, (x + 6, y + 7), 5)
                self.screen.blit(self.font_xs.render(label, True, TEXT_MAIN), (x + 18, y))
                y += 22
            return

        self.screen.blit(self.font_sm.render("AI DEBUG TELEMETRY", True, EMERALD), (x, y))
        y += 22

        for ghost in self.ghosts:
            line1 = f"{ghost.name}: {ghost.state:<6} -> {ghost.target}"
            line2 = f"  Path: {len(ghost.path)} steps | Explored: {ghost.nodes_explored}"
            self.screen.blit(self.font_xs.render(line1, True, CYAN), (x, y))
            y += 15
            self.screen.blit(self.font_xs.render(line2, True, TEXT_DIM), (x, y))
            y += 17

        y += 4
        self.screen.blit(self.font_sm.render("BOSS PREDICTIVE AMBUSH", True, CRIMSON), (x, y))
        y += 20
        boss_lines = [
            f"State: {self.boss.state}  Target: {self.boss.target}",
            f"Replan: {max(0.0, self.boss.replan_timer):.2f}s/{self.boss.replan_interval:.2f}s  Explored: {self.boss.nodes_explored}",
            f"Why: {self.boss.decision_reason}",
        ]
        for line in boss_lines:
            self.screen.blit(self.font_xs.render(line, True, TEXT_MAIN), (x, y))
            y += 16

        y += 4
        self.screen.blit(self.font_xs.render("Ambush Candidates (Cell | P | Cost | Util):", True, YELLOW), (x, y))
        y += 16
        for cand in self.boss.candidates[:4]:
            mark = "*" if cand["selected"] else " "
            c_txt = (
                f"{mark}{str(cand['cell']):<8} P:{cand['prob']*100:4.0f}% "
                f"C:{cand['cost']:4.1f} U:{cand['utility']:5.1f}"
            )
            col = YELLOW if cand["selected"] else TEXT_DIM
            self.screen.blit(self.font_xs.render(c_txt, True, col), (x, y))
            y += 15

        y += 8
        pygame.draw.line(self.screen, PANEL_BORDER, (x, y), (WIDTH - 16, y), 1)
        y += 10

        self.screen.blit(self.font_sm.render("MARKOV 1ST-ORDER MODEL", True, AMBER), (x, y))
        y += 20
        pred_d = self.markov.predicted_dir or "-"
        pred_p = self.markov.predicted_prob * 100.0
        m_lines = [
            f"Last: {self.markov.last_dir or '-'} -> Pred: {pred_d} ({pred_p:.0f}%)",
            f"Accuracy: {self.markov.accuracy():.1f}% ({self.markov.correct_predictions}/{self.markov.total_predictions}) | N={self.markov.total_transitions}",
        ]
        for line in m_lines:
            self.screen.blit(self.font_xs.render(line, True, TEXT_MAIN), (x, y))
            y += 16

        y += 4
        header = "FROM\\TO   U    D    L    R"
        self.screen.blit(self.font_xs.render(header, True, TEXT_DIM), (x, y))
        y += 15
        for r_dir in DIR_LIST:
            row = self.markov.counts[r_dir]
            row_str = f"   {r_dir}    {row['U']:3d}  {row['D']:3d}  {row['L']:3d}  {row['R']:3d}"
            col = AMBER if r_dir == self.markov.last_dir else TEXT_MAIN
            self.screen.blit(self.font_xs.render(row_str, True, col), (x, y))
            y += 15

    def draw_menu(self):
        self.screen.fill(BG_DARK)
        title = self.font_xl.render("Lanterns of Hollowkeep", True, AMBER)
        sub = self.font_sm.render("Procedural DFS Maze  |  FSM + A*  |  Markov Predictive Boss", True, TEXT_DIM)
        self.screen.blit(title, (WIDTH // 2 - title.get_width() // 2, 58))
        self.screen.blit(sub, (WIDTH // 2 - sub.get_width() // 2, 118))

        box = pygame.Rect(WIDTH // 2 - 300, 175, 600, 320)
        pygame.draw.rect(self.screen, PANEL_BG, box, border_radius=10)
        pygame.draw.rect(self.screen, PANEL_BORDER, box, 2, border_radius=10)

        hdr = self.font_md.render("WORLD GENERATION PROBABILITIES", True, TEXT_MAIN)
        self.screen.blit(hdr, (box.x + 28, box.y + 22))

        items = [
            ("lantern", "Reward: Soul Lanterns", AMBER),
            ("elixir", "Reward: Blood Elixirs (+25 HP)", EMERALD),
            ("trap", "Hazard: Bone Traps (-20 HP, Slow)", CRIMSON),
            ("web", "Hazard: Shadow Webs (Snare, Cost 2)", PURPLE),
        ]

        self.slider_rects = {}
        y = box.y + 72
        for key, label, col in items:
            pygame.draw.circle(self.screen, col, (box.x + 36, y + 14), 6)
            self.screen.blit(self.font_sm.render(label, True, TEXT_MAIN), (box.x + 54, y + 6))

            minus_r = pygame.Rect(box.right - 170, y, 32, 28)
            plus_r = pygame.Rect(box.right - 58, y, 32, 28)
            self.slider_rects[key] = (minus_r, plus_r)

            pygame.draw.rect(self.screen, (45, 50, 70), minus_r, border_radius=5)
            pygame.draw.rect(self.screen, (45, 50, 70), plus_r, border_radius=5)
            self.screen.blit(self.font_md.render("-", True, TEXT_MAIN), (minus_r.x + 11, minus_r.y + 4))
            self.screen.blit(self.font_md.render("+", True, TEXT_MAIN), (plus_r.x + 10, plus_r.y + 4))

            pct = self.font_md.render(f"{int(round(self.probs[key] * 100)):2d}%", True, col)
            self.screen.blit(pct, (box.right - 122, y + 4))
            y += 58

        pygame.draw.rect(self.screen, AMBER, self.start_btn_rect, border_radius=8)
        btn_txt = self.font_md.render("ENTER THE CRYPT", True, BG_DARK)
        self.screen.blit(
            btn_txt,
            (
                self.start_btn_rect.centerx - btn_txt.get_width() // 2,
                self.start_btn_rect.centery - btn_txt.get_height() // 2,
            ),
        )
        foot = self.font_xs.render("Press ENTER/SPACE or click button to generate maze  |  F1/TAB toggles AI Debug", True, TEXT_DIM)
        self.screen.blit(foot, (WIDTH // 2 - foot.get_width() // 2, HEIGHT - 38))

    def draw_end_overlay(self):
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((10, 10, 16, 185))
        self.screen.blit(overlay, (0, 0))

        won = self.mode == "WIN"
        heading = self.font_xl.render("ESCAPED HOLLOWKEEP" if won else "AMBUSHED IN THE DARK", True, EMERALD if won else CRIMSON)
        self.screen.blit(heading, (WIDTH // 2 - heading.get_width() // 2, HEIGHT // 2 - 110))

        summary = [
            f"Final Score: {self.player.score}   |   Lanterns: {self.player.lanterns_collected}/{self.total_lanterns}",
            f"Time Survived: {self.elapsed:.1f}s   |   Boss Markov Accuracy: {self.markov.accuracy():.1f}%",
            "Press R to Regenerate & Play Again   |   M for Probability Menu   |   ESC for Menu",
        ]
        y = HEIGHT // 2 - 20
        for line in summary:
            t = self.font_sm.render(line, True, TEXT_MAIN)
            self.screen.blit(t, (WIDTH // 2 - t.get_width() // 2, y))
            y += 32

    def draw_playing(self):
        self.screen.fill(BG_DARK)
        self.draw_maze()
        if self.debug_mode:
            self.draw_debug_overlay()
        self.draw_entities()
        self.draw_sidebar()

        if self.banner_timer > 0.0 and self.status_banner:
            msg = self.font_sm.render(self.status_banner, True, TEXT_MAIN)
            box = pygame.Rect(MAZE_W // 2 - msg.get_width() // 2 - 14, 12, msg.get_width() + 28, 28)
            pygame.draw.rect(self.screen, (16, 18, 28), box, border_radius=6)
            pygame.draw.rect(self.screen, AMBER, box, 1, border_radius=6)
            self.screen.blit(msg, (box.x + 14, box.y + 6))

    async def run(self):
        while self.running:
            dt = min(0.05, self.clock.tick(FPS) / 1000.0)
            self.handle_events()

            if self.mode == "MENU":
                self.draw_menu()
            elif self.mode == "PLAYING":
                self.update_playing(dt)
                self.draw_playing()
            else:
                self.draw_playing()
                self.draw_end_overlay()

            pygame.display.flip()
            await asyncio.sleep(0)

        pygame.quit()


if __name__ == "__main__":
    asyncio.run(Game().run())

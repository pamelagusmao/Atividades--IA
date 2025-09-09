import time
import heapq
from collections import deque
import numpy as np
import matplotlib.pyplot as plt

# =============================
# Configuração inicial
# =============================

START = (0, 0)     # ponto inicial (linha, coluna)
GOAL = (14, 14)    # ponto final (linha, coluna)


def parse_map(lines):
    """Converte lista de 15 strings em grid 15x15 com verificação de erros."""
    if len(lines) != 15:
        raise ValueError(f"ERRO: o mapa deve ter 15 linhas, mas tem {len(lines)}")
    grid = np.zeros((15, 15), dtype=int)
    for r, line in enumerate(lines):
        if len(line) != 15:
            raise ValueError(f"ERRO: a linha {r+1} deve ter 15 caracteres, mas tem {len(line)}")
        for c, ch in enumerate(line):
            if ch not in (".", "#"):
                raise ValueError(f"ERRO: caractere inválido '{ch}' na linha {r+1}, coluna {c+1}")
            grid[r, c] = 1 if ch == "#" else 0
    return grid


# =============================
# Funções auxiliares
# =============================

def neighbors(pos):
    """Retorna vizinhos válidos (cima, baixo, esquerda, direita)."""
    r, c = pos
    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        nr, nc = r+dr, c+dc
        if 0 <= nr < 15 and 0 <= nc < 15:
            yield (nr, nc)


def reconstruct_path(came_from, current):
    """Reconstrói o caminho percorrendo os pais até a origem."""
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


# =============================
# Algoritmo BFS
# =============================

def bfs(grid, start=START, goal=GOAL):
    """Busca em largura (BFS)."""
    t0 = time.perf_counter()
    q = deque([start])
    visited = {start}
    came_from = {}
    expanded = 0

    while q:
        cur = q.popleft()
        expanded += 1

        if cur == goal:
            path = reconstruct_path(came_from, cur)
            dt = (time.perf_counter() - t0) * 1000
            return True, path, expanded, dt

        for nb in neighbors(cur):
            r, c = nb
            if grid[r, c] == 1:  # obstáculo
                continue
            if nb not in visited:
                visited.add(nb)
                came_from[nb] = cur
                q.append(nb)

    dt = (time.perf_counter() - t0) * 1000
    return False, [], expanded, dt


# =============================
# Algoritmo A*
# =============================

def manhattan(a, b):
    """Distância Manhattan (heurística)."""
    return abs(a[0]-b[0]) + abs(a[1]-b[1])


def astar(grid, start=START, goal=GOAL):
    """Algoritmo A* (com heurística Manhattan)."""
    t0 = time.perf_counter()
    open_heap = []
    heapq.heappush(open_heap, (manhattan(start, goal), 0, start))
    came_from = {}
    g = {start: 0}
    closed = set()
    expanded = 0

    while open_heap:
        f, gcost, cur = heapq.heappop(open_heap)

        if cur in closed:
            continue
        closed.add(cur)
        expanded += 1

        if cur == goal:
            path = reconstruct_path(came_from, cur)
            dt = (time.perf_counter() - t0) * 1000
            return True, path, expanded, dt

        for nb in neighbors(cur):
            r, c = nb
            if grid[r, c] == 1:
                continue
            tentative = g[cur] + 1
            if nb not in g or tentative < g[nb]:
                g[nb] = tentative
                came_from[nb] = cur
                fscore = tentative + manhattan(nb, goal)
                heapq.heappush(open_heap, (fscore, tentative, nb))

    dt = (time.perf_counter() - t0) * 1000
    return False, [], expanded, dt


# =============================
# Visualização
# =============================

def visualize(grid, path=None, title="Mapa"):
    """Mostra o mapa 15x15 com obstáculos e caminho."""
    arr = grid.copy()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(arr, origin='upper', cmap="gray_r", interpolation='none')
    ax.set_title(title)
    ax.set_xticks(range(15))
    ax.set_yticks(range(15))
    ax.grid(True, linewidth=0.5)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    if path:
        ys = [p[0] for p in path]
        xs = [p[1] for p in path]
        ax.plot(xs, ys, linewidth=2, color="blue")
        ax.scatter([0, 14], [0, 14], s=80, c="red")  # início e fim
    plt.show()


# =============================
# Mapas
# =============================

MAP_A = [
    "..#.##.....#..#",
    "..#.#.........#",
    "....#.....#..#.",
    "...............",
    "....###...#.#..",
    "...#.......#...",
    "..#.......#...#",
    "#..............",
    "#.........##...",
    "........#......",
    ".#.#....#..##.#",
    ".....#.........",
    ".#.#..#.###..#.",
    "....#.#...#...#",
    "#....#.#..###.."
]

MAP_B = [
    ".....#.###.#...",
    "........###.#..",
    "...#.#...#.....",
    "..##..#.....#.#",
    "#......#.....#.",
    "........###.#.#",
    ".......##..##.#",
    "#.#....#...#...",
    "##............#",
    "..###.#.....#..",
    "#....##........",
    "...........##..",
    ".............##",
    "...#...#.#.###.",
    "##...........#."
]

MAP_C = [
    "..##.....###...",
    ".#...#...##....",
    "###.#...##...##",
    "..#.##.#.......",
    "#....#....##...",
    "...#..######..#",
    "#.##..#......#.",
    ".##..##......#.",
    "#......#....##.",
    "...#..#####.#.#",
    "##...#....#.##.",
    "##....#..#..##.",
    "...##..#......#",
    "#.##.###..#..#.",
    "#.............."
]

grid_A = parse_map(MAP_A)
grid_B = parse_map(MAP_B)
grid_C = parse_map(MAP_C)


# =============================
# Execução e comparação
# =============================

def run_and_compare(name, grid):
    print(f"\n=== {name} ===")

    for alg_name, alg_fn in [("BFS", bfs), ("A*", astar)]:
        try:
            found, path, expanded, ms = alg_fn(grid)
            print(f"\n{alg_name}:")
            if found:
                print(f" - Caminho encontrado: {path}")
                print(f" - Tamanho do caminho: {len(path)}")
            else:
                print(" - ERRO: não foi possível encontrar caminho!")
            print(f" - Nós expandidos: {expanded}")
            print(f" - Tempo: {ms:.3f} ms")
            visualize(grid, path if found else None, f"{name} — {alg_name}")
        except Exception as e:
            print(f"ERRO ao executar {alg_name} em {name}: {e}")


if __name__ == "__main__":
    run_and_compare("Mapa A", grid_A)
    run_and_compare("Mapa B", grid_B)
    run_and_compare("Mapa C", grid_C)




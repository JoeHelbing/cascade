"""Lightweight toroidal MultiGrid replacing mesa.space.MultiGrid."""


class MultiGrid:
    """
    A 2D toroidal grid where each cell can hold multiple agents.
    Uses a 2D list for O(1) cell access (matching mesa's internal structure).
    """

    def __init__(self, width, height, torus=True):
        self.width = width
        self.height = height
        self.torus = torus
        # 2D list indexed by [x][y], each cell is a list of agents
        self._grid = [[[] for _ in range(height)] for _ in range(width)]
        # Track empty cells for fast random placement
        self._empties = set((x, y) for x in range(width) for y in range(height))
        # Neighborhood cache
        self._neighborhood_cache = {}

    @property
    def empties(self):
        return self._empties

    def place_agent(self, agent, pos):
        """Place an agent at the given position."""
        x, y = pos
        if self.torus:
            x, y = x % self.width, y % self.height
            pos = (x, y)
        self._grid[x][y].append(agent)
        agent.pos = pos
        self._empties.discard(pos)

    def remove_agent(self, agent):
        """Remove an agent from the grid."""
        x, y = agent.pos
        cell = self._grid[x][y]
        cell.remove(agent)
        if not cell:
            self._empties.add((x, y))
        agent.pos = None

    def move_agent(self, agent, new_pos):
        """Move an agent to a new position."""
        ox, oy = agent.pos
        nx, ny = new_pos
        if self.torus:
            nx, ny = nx % self.width, ny % self.height

        # Remove from old cell
        old_cell = self._grid[ox][oy]
        old_cell.remove(agent)
        if not old_cell:
            self._empties.add((ox, oy))

        # Add to new cell
        new_pos = (nx, ny)
        self._grid[nx][ny].append(agent)
        self._empties.discard(new_pos)

        agent.pos = new_pos

    def is_cell_empty(self, pos):
        """Check if a cell has no agents."""
        return not self._grid[pos[0]][pos[1]]

    def get_neighborhood(self, pos, moore=True, include_center=False, radius=1):
        """Get all cell coordinates within radius (Moore or Von Neumann)."""
        cache_key = (pos, moore, include_center, radius)
        cached = self._neighborhood_cache.get(cache_key)
        if cached is not None:
            return cached

        x, y = pos
        w, h = self.width, self.height
        cells = []
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if not moore and abs(dx) + abs(dy) > radius:
                    continue
                cells.append(((x + dx) % w, (y + dy) % h))

        if not include_center:
            cells.remove(pos)

        self._neighborhood_cache[cache_key] = cells
        return cells

    def get_cell_list_contents(self, cell_list):
        """Get all agents in the given list of cells."""
        grid = self._grid
        contents = []
        extend = contents.extend
        for x, y in cell_list:
            cell = grid[x][y]
            if cell:
                extend(cell)
        return contents

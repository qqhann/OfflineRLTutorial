# @title Environment Code
import sys
from typing import Dict, List, Optional
import numpy as np

EMPTY = 110
WALL = 111
START = 112
REWARD = 113
OUT_OF_BOUNDS = 114
LAVA = 118

TILES = {EMPTY, WALL, START, REWARD, LAVA}

STR_MAP = {"O": EMPTY, "#": WALL, "S": START, "R": REWARD, "L": LAVA}

RENDER_DICT = {v: k for k, v in STR_MAP.items()}
RENDER_DICT[EMPTY] = " "
RENDER_DICT[START] = " "


class GridSpec:
    def __init__(self, w: int, h: int):
        self.__data = np.zeros((w, h), dtype=np.int32)
        self.__w: int = w
        self.__h: int = h

    def __setitem__(self, key, val):
        self.__data[key] = val

    def __getitem__(self, key):
        if self.out_of_bounds(key):
            raise NotImplementedError("Out of bounds:" + str(key))
        return self.__data[tuple(key)]

    def out_of_bounds(self, wh):
        """Return true if x, y is out of bounds"""
        w, h = wh
        if w < 0 or w >= self.__w:
            return True
        if h < 0 or h >= self.__h:
            return True
        return False

    def get_neighbors(self, k, xy=False):
        """Return values of up, down, left, and right tiles"""
        if not xy:
            k = self.idx_to_xy(k)
        offsets = [
            np.array([0, -1]),
            np.array([0, 1]),
            np.array([-1, 0]),
            np.array([1, 0]),
        ]
        neighbors = [
            self[k + offset] if (not self.out_of_bounds(k + offset)) else OUT_OF_BOUNDS
            for offset in offsets
        ]
        return neighbors

    def get_value(self, k, xy=False):
        """Return values of up, down, left, and right tiles"""
        if not xy:
            k = self.idx_to_xy(k)
        return self[k]

    def find(self, value):
        return np.array(np.where(self.spec == value)).T

    @property
    def spec(self):
        return self.__data

    @property
    def width(self):
        return self.__w

    def __len__(self):
        return self.__w * self.__h

    @property
    def height(self):
        return self.__h

    def idx_to_xy(self, idx):
        if hasattr(idx, "__len__"):  # array
            x = idx % self.__w
            y = np.floor(idx / self.__w).astype(np.int32)
            xy = np.c_[x, y]
            return xy
        else:
            return np.array([idx % self.__w, int(np.floor(idx / self.__w))])

    def xy_to_idx(self, key):
        shape = np.array(key).shape
        if len(shape) == 1:
            return key[0] + key[1] * self.__w
        elif len(shape) == 2:
            return key[:, 0] + key[:, 1] * self.__w
        else:
            raise NotImplementedError()

    def __hash__(self):
        data = (self.__w, self.__h) + tuple(self.__data.reshape([-1]).tolist())
        return hash(data)


def spec_from_string(s: str, valmap: Dict[str, int] = STR_MAP) -> GridSpec:
    if s.endswith("\\"):
        s = s[:-1]
    rows = s.split("\\")
    rowlens = np.array([len(row) for row in rows])
    assert np.all(rowlens == rowlens[0])
    w, h = len(rows[0]), len(rows)

    gs = GridSpec(w, h)
    for i in range(h):
        for j in range(w):
            gs[j, i] = valmap[rows[i][j]]
    return gs


def spec_from_sparse_locations(w: int, h: int, tile_to_locs) -> GridSpec:
    """
    Example usage:
    >> spec_from_sparse_locations(10, 10, {START: [(0,0)], REWARD: [(7,8), (8,8)]})
    """
    gs = GridSpec(w, h)
    for tile_type in tile_to_locs:
        locs = np.array(tile_to_locs[tile_type])
        for i in range(locs.shape[0]):
            gs[tuple(locs[i])] = tile_type
    return gs


def local_spec(map, xpnt) -> GridSpec:
    """
    >>> local_spec("yOy\\\\Oxy", xpnt=(5,5))
    array([[4, 4],
           [6, 4],
           [6, 5]])
    """
    Y = 0
    X = 1
    O = 2
    valmap = {"y": Y, "x": X, "O": O}
    gs = spec_from_string(map, valmap=valmap)
    ys = gs.find(Y)
    x = gs.find(X)
    result = ys - x + np.array(xpnt)
    return result


ACT_NOOP = 0
ACT_UP = 1
ACT_DOWN = 2
ACT_LEFT = 3
ACT_RIGHT = 4
ACT_DICT = {
    ACT_NOOP: [0, 0],
    ACT_UP: [0, -1],
    ACT_LEFT: [-1, 0],
    ACT_RIGHT: [+1, 0],
    ACT_DOWN: [0, +1],
}
ACT_TO_STR = {
    ACT_NOOP: "NOOP",
    ACT_UP: "UP",
    ACT_LEFT: "LEFT",
    ACT_RIGHT: "RIGHT",
    ACT_DOWN: "DOWN",
}


class TransitionModel:
    def __init__(self, gridspec: GridSpec, eps=0.2):
        self.gs = gridspec
        self.eps = eps

    def get_aprobs(self, s, a) -> np.ndarray:
        # TODO: could probably output a matrix over all states...
        legal_moves = self.__get_legal_moves(s)
        p = np.zeros(len(ACT_DICT))
        p[legal_moves] = self.eps / (len(legal_moves))
        if a in legal_moves:
            p[a] += 1.0 - self.eps
        else:
            # p = np.array([1.0,0,0,0,0])  # NOOP
            p[ACT_NOOP] += 1.0 - self.eps
        return p

    def __get_legal_moves(self, s) -> List[int]:
        xy = np.array(self.gs.idx_to_xy(s))
        moves = [
            move
            for move in ACT_DICT
            if not self.gs.out_of_bounds(xy + ACT_DICT[move])
            and self.gs[xy + ACT_DICT[move]] != WALL
        ]
        return moves


OBS_ONEHOT = "onehot"
OBS_RANDOM = "random"
OBS_SMOOTH = "smooth"


class GridEnv:
    def __init__(
        self, gridspec: GridSpec, teps=0.0, observation_type=OBS_ONEHOT, dim_obs=8
    ):
        super(GridEnv, self).__init__()
        self.num_states = len(gridspec)
        self.num_actions = 5
        self.obs_type = observation_type
        self.gs = gridspec
        self.model = TransitionModel(gridspec, eps=teps)
        self._transition_matrix: Optional[np.ndarray] = None
        self._transition_matrix = self.transition_matrix()

        if self.obs_type == OBS_RANDOM:
            self.dim_obs = dim_obs
            self.obs_matrix = np.random.randn(self.num_states, self.dim_obs)
        elif self.obs_type == OBS_SMOOTH:
            self.dim_obs = dim_obs
            self.obs_matrix = np.random.randn(self.num_states, self.dim_obs)
            trans_matrix = np.sum(self._transition_matrix, axis=1) / self.num_actions
            for k in range(10):
                cur_obs_mat = self.obs_matrix[:, :]
                for state in range(self.num_states):
                    new_obs = trans_matrix[state].dot(cur_obs_mat)
                    self.obs_matrix[state] = new_obs
        else:
            self.dim_obs = self.gs.width + self.gs.height

    def observation(self, s):
        if self.obs_type == OBS_ONEHOT:
            xy_vec = np.zeros(self.gs.width + self.gs.height)
            xy = self.gs.idx_to_xy(s)
            xy_vec[xy[0]] = 1.0
            xy_vec[xy[1] + self.gs.width] = 1.0
            return xy_vec
        elif self.obs_type == OBS_RANDOM or self.obs_type == OBS_SMOOTH:
            return self.obs_matrix[s]
        else:
            raise ValueError("Invalid obs type %s" % self.obs_type)

    def reward(self, s, a, ns):
        """
        Returns the reward (float)
        """
        tile_type = self.gs[self.gs.idx_to_xy(s)]
        if tile_type == REWARD:
            return 1
        elif tile_type == LAVA:
            return -1
        else:
            return 0

    def transitions(self, s, a):
        """
        Returns a dictionary of next_state (int) -> prob (float)
        """
        tile_type = self.gs[self.gs.idx_to_xy(s)]
        # if tile_type == LAVA: # Lava gets you stuck
        #    return {s: 1.0}
        if tile_type == WALL:
            return {s: 1.0}

        aprobs = self.model.get_aprobs(s, a)
        t_dict = {}
        for sa in range(5):
            if aprobs[sa] > 0:
                next_s = self.gs.idx_to_xy(s) + ACT_DICT[sa]
                next_s_idx = self.gs.xy_to_idx(next_s)
                t_dict[next_s_idx] = t_dict.get(next_s_idx, 0.0) + aprobs[sa]
        return t_dict

    def initial_state_distribution(self):
        start_idxs = np.array(np.where(self.gs.spec == START)).T
        num_starts = start_idxs.shape[0]
        initial_distribution = {}
        for i in range(num_starts):
            initial_distribution[self.gs.xy_to_idx(start_idxs[i])] = 1.0 / num_starts
        return initial_distribution

    def step_stateless(self, s, a, verbose=False):
        probs = self.transitions(s, a).items()
        ns_idx = np.random.choice(range(len(probs)), p=[p[1] for p in probs])
        ns = probs[ns_idx][0]
        rew = self.reward(s, a, ns)
        return ns, rew

    def step(self, a, verbose=False):
        ns, r = self.step_stateless(self.__state, a, verbose=verbose)
        self.__state = ns
        return ns, r, False, {}

    def reset(self):
        init_distr = self.initial_state_distribution().items()
        start_idx = np.random.choice(len(init_distr), p=[p[1] for p in init_distr])
        self.__state = init_distr[start_idx][0]
        self._timestep = 0
        return start_idx

    def render(self, close=False, ostream=sys.stdout):
        if close:
            return

        state = self.__state
        ostream.write("-" * (self.gs.width + 2) + "\n")
        for h in range(self.gs.height):
            ostream.write("|")
            for w in range(self.gs.width):
                if self.gs.xy_to_idx((w, h)) == state:
                    ostream.write("*")
                else:
                    val = self.gs[w, h]
                    ostream.write(RENDER_DICT[val])
            ostream.write("|\n")
        ostream.write("-" * (self.gs.width + 2) + "\n")

    def transition_matrix(self):
        if self._transition_matrix is None:
            transition_matrix = np.zeros(
                (self.num_states, self.num_actions, self.num_states)
            )
            for s in range(self.num_states):
                for a in range(self.num_actions):
                    for ns, prob in self.transitions(s, a).items():
                        transition_matrix[s, a, ns] = prob
            self._transition_matrix = transition_matrix
        return self._transition_matrix

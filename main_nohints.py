from enum import Enum
import random
from collections import deque
from operator import itemgetter

random.seed(11)

TURN_CNT = 300
MARGIN = 5
FLOOR_LEN = 30
DANGER_CORNER_WIDTH = 3

MAXINT = 9223372036854775807


class PointDiff:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, other):
        if not isinstance(other, PointDiff):
            return False
        return (self.x == other.x) and (self.y == other.y)

    def __hash__(self):
        return hash((self.x, self.y))

    def __repr__(self):
        return f"PointDiff({self.x}, {self.y})"


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        next_x = self.x + other.x
        next_y = self.y + other.y
        return Point(next_x, next_y)

    def __sub__(self, other):
        diff_x = self.x - other.x
        diff_y = self.y - other.y
        return PointDiff(diff_x, diff_y)

    def __eq__(self, other):
        if not isinstance(other, Point):
            return False
        return (self.x == other.x) and (self.y == other.y)

    def __hash__(self):
        return hash((self.x, self.y))

    def __repr__(self):
        return f"({self.x}, {self.y})"


move_actions_table = {
    PointDiff(-1, 0): "U",
    PointDiff(1, 0): "D",
    PointDiff(0, -1): "L",
    PointDiff(0, 1): "R",
    PointDiff(0, 0): ".",
}

move_char_to_diff = {v: k for k, v in move_actions_table.items()}

blockade_conv_table = {
    PointDiff(-1, 0): "u",
    PointDiff(1, 0): "d",
    PointDiff(0, -1): "l",
    PointDiff(0, 1): "r",
}

neighbour_diffs = list(blockade_conv_table.keys())


def cal_distance_points(p1, p2):
    """Manhattan distance"""
    return abs(p1.x - p2.x) + abs(p1.y - p2.y)


def cal_distance(animal1, animal2):
    return cal_distance_points(animal1.point, animal2.point)


def shortest_path_distance(p1, p2):
    pass


class Tile(Enum):
    EMPTY = 0
    WALL = 1  # 周囲の壁
    PARTITION = 2
    NOTPARTITION = 3
    DANGER = 4
    NUISANCE_GOAL = 5

    def __str__(self):
        return str(self.value)


class Floor:
    def __init__(self, floor_len, margin):
        self.tiles = []

        # 上5行
        for _ in range(margin):
            row = [Tile.WALL] * (floor_len + margin * 2)
            self.tiles.append(row)

        # 真ん中30行
        for _ in range(floor_len):
            row = [Tile.WALL] * margin + [Tile.EMPTY] * floor_len + [Tile.WALL] * margin
            self.tiles.append(row)

        # 下5行
        for _ in range(margin):
            row = [Tile.WALL] * (floor_len + margin * 2)
            self.tiles.append(row)

    def __repr__(self):
        tiles_txt = ""
        for row in self.tiles:
            row_txt = "# "  # printする際に頭に＃があるとコメント扱いされる
            for tile in row:
                row_txt += str(tile.value)
            tiles_txt += row_txt + "\n"
        return tiles_txt

    def get_tile(self, point):
        row = point.x
        col = point.y
        return self.tiles[row][col]

    def is_safe(self, point):
        return self.get_tile(point) not in [Tile.WALL, Tile.PARTITION]

    def is_safe_completely(self, point):
        return self.get_tile(point) not in [Tile.WALL, Tile.PARTITION, Tile.DANGER]

    def update_tile(self, point, tile):
        row = point.x
        col = point.y
        self.tiles[row][col] = tile
        if tile in [Tile.WALL, Tile.PARTITION, Tile.DANGER]:
            self.update_danger(point, tile)

    def update_danger(self, point, tile):
        # 行くべきでない場所を更新する
        # 何かしらで埋められた場合は、行くべきでない場所が増えたか確認し更新する
        for diff in neighbour_diffs:
            # 更新する場所の周囲がdangerになる可能性
            # 下記のpointがwallかpartitionになる。
            danger_cand = point + diff
            emptys = self.neighbor_empty(danger_cand)
            # 周囲を3つ以上WALLかPARTITIONかDANGERに囲まれていてemptyならdangerに変更
            if len(emptys) == 1 and (self.get_tile(danger_cand) == Tile.EMPTY):
                # WARNING: 無限ループ
                self.update_tile(danger_cand, Tile.DANGER)

    def neighbor_empty(self, point):
        """受け取ったpointの隣でwall, partition, dangerでないものを返す
        これが1つしか返さなければ、与えたpointの唯一の通路になる
        """
        emptys = []
        for diff in neighbour_diffs:
            neighbour = point + diff
            if self.get_tile(neighbour) not in [Tile.WALL, Tile.PARTITION, Tile.DANGER]:
                emptys.append(neighbour)
        return emptys


# CAUTION: update_tile dangerでエラーにならないように、MARGINは 2以上である必要。
floor = Floor(FLOOR_LEN, MARGIN)
assert len(floor.tiles) == FLOOR_LEN + MARGIN * 2
for row in floor.tiles:
    assert len(row) == FLOOR_LEN + MARGIN * 2


class Steps(dict):
    def __init__(self, U=0, D=0, L=0, R=0, *arg, **kw):
        super(Steps, self).__init__(*arg, **kw)
        self["U"] = U
        self["D"] = D
        self["L"] = L
        self["R"] = R

    def __repr__(self):
        return f"({self['U']}, {self['D']}, {self['L']}, {self['R']})"

    def __add__(self, other):
        next_steps = Steps()
        for key in ["U", "D", "L", "R"]:
            next_steps[key] = self[key] + other[key]
        return next_steps

    def __eq__(self, other):
        if not isinstance(other, Steps):
            return False

        for k in self.keys():
            if self[k] != other[k]:
                return False
        return True

    def __hash__(self):
        return hash((self["U"], self["D"], self["L"], self["R"]))


def cal_steps(start, goal):
    diff = goal - start
    steps = Steps()
    # 上下
    if diff.x > 0:
        steps["D"] = diff.x
    else:
        # 正の数にする
        steps["U"] = -diff.x

    # 左右
    if diff.y > 0:
        steps["R"] = diff.y
    else:
        steps["L"] = -diff.y

    return steps


def get_dirs_priority(start, goal):
    steps = cal_steps(start, goal)

    dirs_diff = []
    # cntが大きい方角順
    for dir_str, cnt in sorted(steps.items(), key=itemgetter(1), reverse=True):
        if cnt > 0:
            dirs_diff.append(move_char_to_diff[dir_str])

    return dirs_diff


class VisitedFloor(Floor):
    def __init__(self, floor_len, margin):
        self.floor_len = floor_len
        self.margin = margin
        self.routes = self.create()

    def create(self):
        counts = []
        square_len = self.floor_len + self.margin * 2
        for _ in range(square_len):
            row = [None] * square_len
            counts.append(row)
        return counts

    def visit(self, point, route):
        row = point.x
        col = point.y
        self.routes[row][col] = route

    def get_route(self, point):
        row = point.x
        col = point.y
        return self.routes[row][col]

    def is_visited(self, point):
        """Noneでなければ訪れたことがある"""
        return self.get_route(point) != None


def solve_route(start, goal, floor):  # type: ignore
    """ゴールまでの経路のpointをリストで返す
    STARTは含まず、GOALは含む。
    STARTとGOALが同じ場合は空のリストを返す。
    """
    visited = VisitedFloor(floor_len=FLOOR_LEN, margin=MARGIN)

    route = []
    visited.visit(start, route)

    q = deque([start])

    while q:
        now = q.popleft()
        if now == goal:
            return visited.get_route(now)

        for diff in neighbour_diffs:
            neighbour = now + diff

            if (not visited.is_visited(neighbour)) & floor.is_safe(neighbour):
                q.append(neighbour)
                route = visited.get_route(now) + [neighbour]
                visited.visit(neighbour, route)

                # pathが長すぎる場合は囲われていると見なす
                # TODO: 妥当なやり方
                threshold = 100
                if len(route) >= threshold:
                    return None
    return None


class PartitionCands(Floor):
    def __init__(self, MARGIN):
        self.tiles = self.create_empty_tiles(MARGIN)

    def create_empty_tiles(self, MARGIN):
        tiles = []
        square_len = FLOOR_LEN + MARGIN * 2
        for _ in range(square_len):
            row = [Tile.EMPTY] * square_len
            tiles.append(row)
        return tiles

    def update_tile(self, point, tile):
        row = point.x
        col = point.y
        self.tiles[row][col] = tile

    def refresh(self, humans, pets):
        self.tiles = self.create_empty_tiles(MARGIN)

        # 開始時点の人がいる場所はNOTPARTITION
        for human in humans:
            self.update_tile(human.point, Tile.NOTPARTITION)

            # 開始時点に人がDANGERにいる場合、そのDANGERに隣接するDANGERとEMPTY(このEMPTYは1つのみか)
            # にはNOTPARTITION
            def recur_adjacent_danger(point):
                if floor.get_tile(point) == Tile.DANGER:
                    self.update_tile(point, Tile.NOTPARTITION)
                    for neighbour_diff in neighbour_diffs:
                        neighbour = point + neighbour_diff
                        recur_adjacent_danger(neighbour)
                elif floor.get_tile(point) == Tile.EMPTY:
                    # EMPTYの場合はNOTPARTITIONにして、再帰は行わない。
                    self.update_tile(point, Tile.NOTPARTITION)

        # 開始時点にペットがいる場所と隣接する場所はNOTPARTITION
        for pet in pets:
            self.update_tile(pet.point, Tile.NOTPARTITION)

            # 隣接する場所
            for neighbour_diff in neighbour_diffs:
                neighbour_point = pet.point + neighbour_diff
                self.update_tile(neighbour_point, Tile.NOTPARTITION)


partition_cands = PartitionCands(MARGIN)


class HumansCount(Floor):
    def __init__(self, margin, humans):
        self.margin = margin
        self.counts = self.create_zeros()
        self.update_human_counts(humans)

    def create_zeros(self):
        counts = []
        square_len = FLOOR_LEN + self.margin * 2
        for _ in range(square_len):
            row = [0] * square_len
            counts.append(row)
        return counts

    def add_one(self, point):
        row = point.x
        col = point.y
        self.counts[row][col] += 1

    def update_human_counts(self, humans):
        for human in humans:
            self.add_one(human.point)

    def get_cnt(self, point):
        row = point.x
        col = point.y
        return self.counts[row][col]


class CanGoFloor(Floor):
    def __init__(self, margin):
        self.margin = margin
        self.counts = self.create_zeros()

    def create_zeros(self):
        counts = []
        square_len = FLOOR_LEN + self.margin * 2
        for _ in range(square_len):
            row = [0] * square_len
            counts.append(row)
        return counts

    def add_one(self, point):
        row = point.x
        col = point.y
        self.counts[row][col] += 1

    def write_weight(self, point, weight):
        row = point.x
        col = point.y
        self.counts[row][col] = weight

    def is_visited(self, point):
        row = point.x
        col = point.y
        return self.counts[row][col] > 0


class FloorForExpect(Floor):
    def __init__(self, margin):
        self.margin = margin
        self.counts = self.create_zeros()

    def create_zeros(self):
        counts = []
        square_len = FLOOR_LEN + self.margin * 2
        for _ in range(square_len):
            row = [0] * square_len
            counts.append(row)
        return counts

    def add(self, point, number):
        row = point.x
        col = point.y
        self.counts[row][col] += number


class VisitedSteps(Floor):
    def __init__(self, margin):
        self.margin = margin
        self.cells = self.create_nones()

    def create_nones(self):
        counts = []
        square_len = FLOOR_LEN + self.margin * 2
        for _ in range(square_len):
            row = [set() for _ in range(square_len)]
            counts.append(row)
        return counts

    def visits(self, point, steps):
        row = point.x
        col = point.y
        self.cells[row][col].add(steps)

    def is_visited_with_steps(self, point, steps):
        row = point.x
        col = point.y
        return steps in self.cells[row][col]


assert len(partition_cands.tiles) == FLOOR_LEN + MARGIN * 2
for row in partition_cands.tiles:
    assert len(row) == FLOOR_LEN + MARGIN * 2


class Kind(Enum):
    COW = 1
    PIG = 2
    RABBIT = 3
    DOG = 4
    CAT = 5


kind_to_block_dist = {
    Kind.COW: 3,
    Kind.PIG: 4,
    Kind.RABBIT: 5,
    Kind.DOG: 4,
    Kind.CAT: 4,
}

kind_to_action_cnt = {
    Kind.COW: 1,
    Kind.PIG: 2,
    Kind.RABBIT: 3,
    Kind.DOG: 2,
    Kind.CAT: 2,
}


class PetStatus(Enum):
    NORMAL = 0
    DEAD = 1


class Pet:
    def __init__(
        self,
        id,
        kind,
        point,
        expected_dir=None,
        status=PetStatus.NORMAL,
        can_go_floor=None,
    ):
        self.id = id
        self.kind = kind
        self.action_cnt = kind_to_action_cnt[kind]
        self.point = point
        self.expected_dir = expected_dir if expected_dir else point  # 初期値は現在地を入れる
        self.expected_route = []
        self.status = status
        self.can_go_floor = can_go_floor

    def move(self, action_char):
        diff = move_char_to_diff[action_char]
        next_point = self.point + diff
        if floor.get_tile(next_point) not in [Tile.WALL, Tile.PARTITION]:
            self.point = next_point

    def __eq__(self, other):
        if not isinstance(other, Pet):
            return False
        return self.id == other.id

    def __hash__(self):
        return self.id

    def __repr__(self):
        return f"Pet({self.id}, {self.kind}, {self.point}, {self.status})"

    # 最新のpartitionを把握したいため、humanごとに実行したい
    # 4^n 回計算必要なのはきついか？→最大4^6＝4096としてみるか
    # def update_expected_point(self, rest_action):
    #     floor_for_expect = FloorForExpect(MARGIN)

    #     def recur(point, rest_action, prob):
    #         if rest_action == 0:
    #             floor_for_expect.add(point, prob)
    #             return

    #         can_go_neighbours = []
    #         for neighbour_diff in neighbour_diffs:
    #             neighbour = point + neighbour_diff
    #             if floor.get_tile(neighbour) not in [Tile.WALL, Tile.PARTITION]:
    #                 can_go_neighbours.append(neighbour)

    #         prob /= len(can_go_neighbours)

    #         for can_go in can_go_neighbours:
    #             recur(can_go, rest_action-1, prob)

    #     rest_action
    #     recur(self.point, rest_action)

    def update_status(self, humans):
        # TODO: free判定をちゃんとやるか、閾値変更
        THRESHOLD_POINTS = 250
        can_go_cnt = 0
        self.can_go_floor = CanGoFloor(MARGIN)
        humans_count = HumansCount(MARGIN, humans)

        # can_go_cnt を数える
        # 下記の再帰が終わったタイミングで can_go_cnt が THRESHOLD未満で、
        # humanもその範囲にいないなら is_catched
        # humanがいるなら free。 can_go_cntがTHRESHOLD以上なら free。
        def free_dfs(point, weight):
            nonlocal can_go_cnt
            if self.can_go_floor.is_visited(point):
                return None
            self.can_go_floor.write_weight(point, weight)

            # TODO: 人間も一緒に閉じ込められてしまった場合
            if humans_count.get_cnt(point) > 0:
                return True

            can_go_cnt += 1
            if can_go_cnt >= THRESHOLD_POINTS:
                # 閾値を超えたら終了
                return True

            for neighbour_diff in neighbour_diffs:
                neighbour = point + neighbour_diff
                if floor.get_tile(neighbour) not in [Tile.WALL, Tile.PARTITION]:
                    check = free_dfs(neighbour, weight * 0.5)
                    if check:
                        return True

        initial_weight = 1
        check = free_dfs(self.point, initial_weight)
        if check is not True:
            # checkはNoneのことがある。その場合はFalse
            self.status = PetStatus.DEAD

        print(f"# {self.__repr__()}, can_go_count: {can_go_cnt}, check: {check}")

    def update_expected_dir(self):
        mean_x = 0
        mean_y = 0

        tile_len = len(self.can_go_floor.counts)
        for row in range(tile_len):
            for col in range(tile_len):
                weight = self.can_go_floor.counts[row][col]
                if weight > 0:
                    mean_x += row * weight
                    mean_y += col * weight

        self.expected_dir = Point(round(mean_x), round(mean_y))

    def update_expected_route(self):
        self.expected_route = solve_route(self.point, self.expected_dir, floor)

    def is_free(self):
        return self.status == PetStatus.NORMAL


class HumanStatus(Enum):
    NORMAL = 0
    GETOUT = 1
    NUISANCE = 2
    DEAD = 3


# 大きすぎると、行けるところが少ないと置かなくなる
HUMAN_FREE_POINTS_THRESHOLD = 100


class Human:
    def __init__(
        self,
        id,
        point,
        status=HumanStatus.NORMAL,
        team=None,
        role=None,
        target=None,
        block_dist=3,
        next_blockade=None,
        next_move=None,
        route=None,
        get_out_route=None,
        solve_route_turn=0,
        nuisance_goal=None,
        nuisance_route=None,
    ):
        self.id = id
        self.point = point
        self.status = status
        self.team = team
        self.role = role
        self.target = target
        self.block_dist = block_dist
        self.next_blockade = next_blockade
        self.next_move = next_move
        self.route = route if route else deque()
        self.get_out_route = get_out_route if get_out_route else deque()
        self.solve_route_turn = solve_route_turn
        self.nuisance_goal = nuisance_goal
        self.nuisance_route = nuisance_route if nuisance_route else deque()

    def __repr__(self):
        return f"Human({self.id}, {self.point}, target: {self.target}, status:{self.status}, next_move:{self.next_move}, next_blockade: {self.next_blockade}, route: {self.route}, nuisance: {self.nuisance_route})"

    def select_target(self, pets):
        # self.target = self.team.target  # type:ignore

        # 最も近いペットをターゲットにする
        # TODO: 囲われているペットは無視する
        # TODO: 最短経路を考慮すべきか？
        # TODO: 全員使った場合の挙動は？
        # nearest_pet = None
        # min_distance = MAXINT
        pets_priority_distances = []

        # DogとCatの優先順位を下げる
        kind_to_priority = {
            Kind.COW: 1,
            Kind.PIG: 1,
            Kind.RABBIT: 1,
            Kind.DOG: 2,
            Kind.CAT: 2,
        }

        for pet in pets:
            if pet.is_free():
                distance = cal_distance_points(self.point, pet.point)
                priority = kind_to_priority[pet.kind]
                pets_priority_distances.append((pet, priority, distance))

        # priority無効化
        pets_priority_distances = sorted(pets_priority_distances, key=lambda x: x[2])

        # TODO: 全て捕まえた時の挙動
        if len(pets_priority_distances) > 0:
            self.target = pets_priority_distances[0][0]

            # petのkindによってblockする距離を変える
            self.block_dist = kind_to_block_dist[self.target.kind]  # type: ignore
        else:
            # 全て捕まえたとき
            self.target = None
            self.block_dist = None

    def next_action_char(self):
        if self.next_blockade:
            diff = self.next_blockade - self.point
            return blockade_conv_table[diff]

        if self.next_move:
            # 進む先のtileはPartition候補から消す
            partition_cands.update_tile(self.next_move, Tile.NOTPARTITION)

            next_diff = self.next_move - self.point
            move_char = move_actions_table[next_diff]

            return move_char

        # 取れる行動がなければ何もしない
        return "."

    def set_status(self):
        if self.status == HumanStatus.NUISANCE:
            # 当初のgoalについたならNORMALに戻す
            if self.point == self.nuisance_goal:
                self.status = HumanStatus.NORMAL

                # NORMALに戻ったならrouteを引き直す
                self.solve_route_turn = -1
                self.route = deque()

        # elseにしてないのは、上でNORMALになる場合があるから
        if self.status != HumanStatus.NUISANCE:
            # DANGERにいるなら脱出モード
            if floor.get_tile(self.point) == Tile.DANGER:
                self.status = HumanStatus.GETOUT
            else:
                self.status = HumanStatus.NORMAL

    def refresh(self):
        self.next_blockade = None
        self.next_move = None

    def think_route(self, turn):
        # humanの solve_turn を過ぎていたら solveし直す
        # routeの次が通行不能になってる場合もsolveし直す
        # Refactor

        print(f"# {self.target}")

        route = solve_route(self.point, self.target.point, floor)
        if route is None:
            self.route = deque()
        else:
            self.route = deque(route)

        route_len = len(self.route)
        if route_len >= 6:
            # 遠いならそのルートの半分の長さまではそのままでいく。
            self.solve_route_turn = turn + route_len // 2
        else:
            # 近いなら毎ターンsolveする
            self.solve_route_turn = turn + 1

    def think_to_get_out(self):
        route = self.get_route_to_empty()
        if route:
            self.get_out_route = deque(route)
        else:
            # 逃げ出す道がないなら死んでいる
            print(f"# Died by get_out: {self}")
            self.status = HumanStatus.DEAD

    def think_to_nuisance_route(self):
        route = solve_route(self.point, self.nuisance_goal, floor)
        print(f"# nuisance_route: {self.point}, {self.nuisance_goal}, {route}")
        if route:
            self.nuisance_route = deque(route)
        else:
            # 逃げ出す道がないなら死んでいる
            print(f"# Died by nuisance: {self}")
            self.status = HumanStatus.DEAD

    def get_route_to_empty(self):  # type: ignore
        """EMPTYまでの経路のpointをリストで返す
        STARTは含まず、GOALは含む。
        STARTとGOALが同じ場合は空のリストを返す。
        """
        visited = VisitedFloor(floor_len=FLOOR_LEN, margin=MARGIN)

        route = []

        start = self.point
        visited.visit(start, route)

        q = deque([start])

        while q:
            now = q.popleft()
            # その場所がEMPTYなら完了
            if floor.get_tile(now) == Tile.EMPTY:
                return visited.get_route(now)

            for diff in neighbour_diffs:
                neighbour = now + diff

                if (not visited.is_visited(neighbour)) and (
                    floor.get_tile(neighbour) in [Tile.DANGER, Tile.EMPTY]
                ):
                    q.append(neighbour)
                    route = visited.get_route(now) + [neighbour]
                    visited.visit(neighbour, route)

        return None

    def sort_directions(self):
        """directionを選ぶ関数
        絶対値が大きいdirectionを選びやすい

        TODO: ソートの精度の向上
        TODO: 四方を少しはランダムに選ぶようにする
        TODO: 近づきすぎない
        TODO: 待機をどう扱うか
        #"""

        directions = []

        distance = len(self.route)
        random.shuffle(neighbour_diffs)

        if distance <= self.block_dist:
            # self.block_dist離れている時はランダムに動くためrouteがずれる可能性がある。
            self.route = deque()
            # 次のturnにrouteを算出する
            self.solve_route_turn = -1

        if distance == 0:
            # ランダム
            # TODO: 最適な場所にすすむ。なお、別に距離0になることはほとんどなさそう。
            directions += neighbour_diffs

        elif distance == 1:
            # すでに離れている方向の優先度を上げる
            directions += [self.point - self.target.point] + neighbour_diffs  # type: ignore

        elif 2 <= distance <= self.block_dist:
            # 十分近いので待機を優先
            directions += neighbour_diffs
        else:
            # 4より大きい場合
            # 近づく
            # Refactor
            if len(self.route) >= 2:
                next_point = self.route[0]
                direction = next_point - self.point
                directions += [direction]

            directions += neighbour_diffs

        return directions

    def is_free_if_blockade(self, hypothesis_point):
        # TODO: free判定をちゃんとやるか、閾値変更
        can_go_cnt = 0
        visited = CanGoFloor(MARGIN)

        # can_go_cnt を数える
        def free_dfs(point):
            nonlocal can_go_cnt
            if visited.is_visited(point):
                return None
            visited.add_one(point)

            can_go_cnt += 1
            if can_go_cnt >= HUMAN_FREE_POINTS_THRESHOLD:
                # 閾値を超えたら終了
                return True

            for neighbour_diff in neighbour_diffs:
                neighbour = point + neighbour_diff

                # 置く仮定の場所は無いものとする
                if neighbour == hypothesis_point:
                    continue

                if floor.get_tile(neighbour) not in [Tile.WALL, Tile.PARTITION]:
                    check = free_dfs(neighbour)
                    if check:
                        return True

        _ = free_dfs(self.point)

        # 行ける場所が多いなら free
        return can_go_cnt >= HUMAN_FREE_POINTS_THRESHOLD

    def decide_next_action(self, humans):
        if self.status == HumanStatus.NUISANCE:
            # 邪魔なので出ていく
            # TODO:毎回考える必要あるか？
            self.think_to_nuisance_route()
            if len(self.nuisance_route) > 0 and (
                floor.get_tile(self.nuisance_route[0])
                not in [
                    Tile.WALL,
                    Tile.PARTITION,
                ]
            ):
                self.next_move = self.nuisance_route.popleft()
                return

        if self.status == HumanStatus.GETOUT:
            # TODO:毎回考える必要あるか？
            self.think_to_get_out()
            if len(self.get_out_route) > 0 and (
                floor.get_tile(self.get_out_route[0])
                not in [
                    Tile.WALL,
                    Tile.PARTITION,
                ]
            ):
                self.next_move = self.get_out_route.popleft()
                return

        # TODO: 2しか離れてなくても、遠いところに置くことは可能。
        # DANGERにいるときは置ける場所が唯一の通路のため置いてはいけない。
        distance_between_human_target = len(self.route)
        if (3 <= distance_between_human_target <= self.block_dist) and (
            floor.get_tile(self.point) != Tile.DANGER
        ):

            blockade_cand = self.route[0]

            # その位置に壁やpartitionがなく、人やペットの制約もなければ、partitionを立てる
            # TODO: dangerなら置いても良くないか？
            if (floor.get_tile(blockade_cand) == Tile.EMPTY) and (
                partition_cands.get_tile(blockade_cand) == Tile.EMPTY
            ):

                # 置いても自分がfreeなら
                # TODO: 自分は待機するか
                if self.is_free_if_blockade(blockade_cand):
                    # 置いても全humanがfreeなら置く
                    # CAUTION: 時間かかりすぎな可能性
                    # 置くとfreeじゃなくなるhumanは邪魔者としてメモする
                    nuisances = []
                    for human in humans:
                        if human.id == self.id:
                            # 自分はfreeであることを確認済みのため無視
                            continue
                        if not human.is_free_if_blockade(blockade_cand):
                            nuisances.append(human)

                    # 邪魔者がいなければblockadeする
                    print(f"# nuisances: {nuisances}")
                    if len(nuisances) == 0:
                        self.next_blockade = blockade_cand
                        floor.update_tile(self.next_blockade, Tile.PARTITION)
                        return
                    else:
                        # 邪魔者に自分のところを一旦のgoalとさせる（置こうとしたhumanは置いてもfreeなので最短の安全場所）
                        for human in nuisances:
                            human.status = HumanStatus.NUISANCE
                            human.nuisance_goal = self.point
                            human.nuisance_route = deque()
                            print(f"# {self} said '{human} is nuisance!!!'")

                            # 該当場所をnuisance_goalにする
                            floor.update_tile(self.point, Tile.NUISANCE_GOAL)

        # 移動先の優先順位付
        directions = self.sort_directions()

        for direction in directions:
            move_to_cand = self.point + direction
            if floor.get_tile(move_to_cand) in [
                Tile.WALL,
                Tile.PARTITION,
                Tile.DANGER,
            ]:
                pass
            else:
                self.next_move = move_to_cand

                # 進む先がDANGERなら、元々いるところ(human.point)が唯一の通路だった。そのためそこはNOTPARTITIONにする。
                # TODO: 意味あるか不明
                if floor.get_tile(self.next_move) == Tile.DANGER:
                    floor.update_tile(self.point, Tile.NOTPARTITION)

                break

        # Refactor: routeを参照するタイミングと削除するタイミングが違うのがわかりにくい
        if (len(self.route) > 0) and (self.next_move == self.route[0]):
            _ = self.route.popleft()


class Team:
    def __init__(self, humans, target=None):
        self.humans = humans
        self.target = target
        self.set_role()

    def __repr__(self):
        return f"Team({self.humans}, {self.target})"

    def set_role(self):
        for i, human in enumerate(self.humans):
            human.role = i % 4

    def select_target(self, pets):
        """平均して最も近いpetを選ぶ"""
        # TODO: 囲われているペットは無視する
        # TODO: 最短経路を考慮すべきか？
        # TODO: 全部囲われた時の挙動は？

        pet_distance_sum = {pet: 0 for pet in pets}

        for pet in pets:
            if pet.is_free():
                for human in self.humans:  # type: ignore
                    distance = cal_distance(human, pet)
                    pet_distance_sum[pet] += distance

        nearest_distance_sum = MAXINT
        for pet, distance_sum in pet_distance_sum.items():
            if distance_sum < nearest_distance_sum:
                self.target = pet
                nearest_distance_sum = distance_sum
        print(f"# target: {self.target}")


def initial_input():
    N = int(input())
    pets = []
    for i in range(N):
        x, y, t = map(lambda x: int(x), input().split())
        pet = Pet(id=i + 1, kind=Kind(t), point=Point(x - 1 + MARGIN, y - 1 + MARGIN))
        pets.append(pet)

    M = int(input())
    humans = []
    for i in range(M):
        x, y = map(lambda x: int(x), input().split())
        human = Human(id=i + 1, point=Point(x - 1 + MARGIN, y - 1 + MARGIN))
        humans.append(human)

    return N, pets, M, humans


def main():
    N, pets, M, humans = initial_input()

    team = Team(humans)
    for human in humans:
        human.team = team

    for turn in range(TURN_CNT):
        print(f"# turn: {turn}")
        action_str = ""
        partition_cands.refresh(humans, pets)

        # 死んでるpetがいるか確認
        for pet in pets:
            pet.update_status(humans)

        # 死んでるpetをtargetにしているhumanがいたらNoneにする
        for human in humans:
            # Noneの場合もあるためhuman.targetで確認している
            if human.target and (human.target.is_free() is False):
                human.target = None

        # team.select_target(pets)

        for human in humans:
            human.refresh()

        for human in humans:

            human.select_target(pets)

            if human.target is None:
                # すべて捕まえたとき
                action_str += "."
            else:
                human.set_status()

                # routeがPartitionで埋まってしまったなら、そのrouteを削除する。routeはこのターンに引き直す。
                if (len(human.route) > 0) and (
                    Tile.PARTITION in [floor.get_tile(p) for p in human.route]
                ):
                    human.route = deque()
                    human.solve_route_turn = turn

                # turn数に応じて、もしくはrouteがなければ、routeを引き直す
                if (human.solve_route_turn <= turn) or (len(human.route) == 0):
                    human.think_route(turn)

                human.decide_next_action(humans)

                print(f"#{human}")
                action_char = human.next_action_char()

                print(f"# {human}")
                action_str += action_char

        print(f"# {floor}")

        # 人間の行動を出力
        print(action_str, flush=True)

        # 人間の位置を更新
        for human in humans:
            if human.next_move:
                human.point = human.next_move

        # ペットの行動を受け取る
        pet_action_str_list = input().split()
        for pet, actions in zip(pets, pet_action_str_list):
            for action_char in actions:
                pet.move(action_char)


if __name__ == "__main__":
    main()

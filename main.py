from __future__ import annotations
from dataclasses import dataclass
from shutil import move
from typing import List, Union, Optional, Dict, Set
from enum import Enum
import random
import copy
from collections import OrderedDict
from operator import itemgetter

random.seed(11)

TURN_CNT = 300
MARGIN = 5
FLOOR_LEN = 30
DANGER_CORNER_WIDTH = 3

MAXINT = 9223372036854775807


@dataclass
class PointDiff:
    x: int
    y: int

    def __hash__(self):
        return hash((self.x, self.y))


@dataclass
class Point:
    x: int
    y: int

    def __add__(self, other: Union[Point, PointDiff]):
        next_x = self.x + other.x
        next_y = self.y + other.y
        return Point(next_x, next_y)

    def __sub__(self, other: Point):
        diff_x = self.x - other.x
        diff_y = self.y - other.y
        return PointDiff(diff_x, diff_y)

    def __hash__(self) -> int:
        return hash((self.x, self.y))


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


def cal_distance_points(p1: Point, p2: Point):
    """Manhattan distance"""
    return abs(p1.x - p2.x) + abs(p1.y - p2.y)


def cal_distance(animal1: Human, animal2: Human):
    return cal_distance_points(animal1.point, animal2.point)


def shortest_path_distance(p1: Point, p2: Point):
    pass


class Tile(Enum):
    EMPTY = 0
    WALL = 1  # 周囲の壁
    PARTITION = 2
    NOTUSE = 3
    DANGER = 4

    def __str__(self):
        return str(self.value)


class Floor:
    def __init__(self, floor_len, margin):
        self.tiles: List[List[Tile]] = []

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

        # 角はDANGER
        # for row in self.tiles[MARGIN : MARGIN + DANGER_CORNER_WIDTH]:
        #     for col in range(MARGIN, MARGIN + DANGER_CORNER_WIDTH):
        #         self.tiles[row][col] = Tile.DANGER

    def __repr__(self):
        tiles_txt = ""
        for row in self.tiles:
            row_txt = ""
            for tile in row:
                row_txt += str(tile.value)
            tiles_txt += row_txt + "\n"
        return tiles_txt

    def get_tile(self, point: Point):
        row = point.x
        col = point.y
        return self.tiles[row][col]

    def update_tile(self, point: Point, tile: Tile):
        row = point.x
        col = point.y
        self.tiles[row][col] = tile

        # # 行くべきでない場所を更新する
        # # 何かしらで埋められた場合は、行くべきでない場所が増えたか確認し更新する
        # if tile != Tile.EMPTY:
        #     for diff in neighbour_diffs:
        #         danger_cand = point + diff
        #         cnt = self.count_neighbor_filled(danger_cand)
        #         # 周囲を3つ以上何かに囲まれていてemptyならdangerに変更
        #         if (cnt >= 3) & (self.get_tile(point) == Tile.EMPTY):
        #             # WARNING: 無限ループ
        #             self.update_tile(danger_cand, Tile.DANGER)

    def count_neighbor_filled(self, point: Point):
        """argumentの隣がいくつEMPTYでないかを数える"""
        cnt = 0
        for diff in neighbour_diffs:
            neighbour = point + diff
            if self.get_tile(neighbour) != Tile.EMPTY:
                cnt += 1
        return cnt


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


def get_dirs_priority(start, goal) -> List[PointDiff]:
    steps = cal_steps(start, goal)

    dirs_diff: List[PointDiff] = []
    # cntが大きい方角順
    for dir_str, cnt in sorted(steps.items(), key=itemgetter(1), reverse=True):
        if cnt > 0:
            dirs_diff.append(move_char_to_diff[dir_str])

    return dirs_diff


def solve_route(start, goal, floor) -> List[Point]:
    """startからgoalまでの経路のPointのListを返す
    経路長は最短であることを前提とする。
    経路が見つからなければ空のListを返す。

    TODO: 経路長が最短でない場合
    """

    visited: Set[Point] = set()

    def dfs(start, goal, steps) -> List[Point]:
        nonlocal visited
        if start == goal:
            return [goal]

        for step_dir, step_cnt in steps.items():
            if step_cnt == 0:
                continue

            diff = move_char_to_diff[step_dir]
            next_point = start + diff

            if (next_point not in visited) & (
                floor.get_tile(next_point) not in [Tile.WALL, Tile.PARTITION]
            ):
                next_steps = copy.deepcopy(steps)
                next_steps[step_dir] -= 1
                result = dfs(next_point, goal, next_steps)
                if len(result) > 0:
                    return [start] + result

        visited.add(start)

        return []

    steps = cal_steps(start, goal)
    return dfs(start, goal, steps)


class PartitionCands(Floor):
    def __init__(self, MARGIN):
        self.tiles: List[List[Tile]] = self.create_empty_tiles(MARGIN)

    def create_empty_tiles(self, MARGIN):
        tiles = []
        square_len = FLOOR_LEN + MARGIN * 2
        for _ in range(square_len):
            row = [Tile.EMPTY] * square_len
            tiles.append(row)
        return tiles

    def refresh(self, humans, pets):
        self.tiles = self.create_empty_tiles(MARGIN)

        # 開始時点の人がいる場所はNOTUSE
        for human in humans:
            self.update_tile(human.point, Tile.NOTUSE)

        # 開始時点にペットがいる場所と隣接する場所はNOTUSE
        for pet in pets:
            self.update_tile(pet.point, Tile.NOTUSE)

            # 隣接する場所
            for neighbour_diff in neighbour_diffs:
                neighbour_point = pet.point + neighbour_diff
                self.update_tile(neighbour_point, Tile.NOTUSE)


partition_cands = PartitionCands(MARGIN)

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


@dataclass
class Pet:
    id: int
    kind: Kind
    point: Point

    def move(self, action_char):
        diff = move_char_to_diff[action_char]
        next_point = self.point + diff
        if floor.get_tile(next_point) not in [Tile.WALL, Tile.PARTITION]:
            self.point = next_point

    def __hash__(self) -> int:
        return self.id

    # def is_catched(self, humans):
    #     can_go_cnt = 0
    #     visited = Floor()

    #     humans_floor = Floor()
    #     for human
    #     def dfs(point):


@dataclass
class Human:
    id: int
    point: Point
    team: Optional[Team] = None
    role: Optional[int] = None
    target: Optional[Pet] = None
    block_dist: int = 3
    next_blockade: Optional[Point] = None
    next_move: Optional[Point] = None

    def select_target(self, pets):
        self.target = self.team.target  # type:ignore

        # petのkindによってblockする距離を変える
        self.block_dist = kind_to_block_dist[self.target.kind]  # type: ignore

        # # 最も近いペットをターゲットにする
        # # TODO: 囲われているペットは無視する
        # # TODO: 最短経路を考慮すべきか？
        # nearest_pet = None
        # min_distance = 100
        # for pet in pets:
        #     d = cal_distance_points(self.point, pet.point)
        #     if d < min_distance:
        #         nearest_pet = pet
        # self.target = nearest_pet

    def next_action_char(self):
        if self.next_blockade:
            diff = self.next_blockade - self.point
            return blockade_conv_table[diff]

        if self.next_move:
            next_diff = self.next_move - self.point
            move_char = move_actions_table[next_diff]
            return move_char

        # 取れる行動がなければ何もしない
        return "."

    def refresh(self, pets):
        self.next_blockade = None
        self.next_move = None
        self.select_target(pets)

    def sort_directions(self) -> List[PointDiff]:
        """directionを選ぶ関数
        絶対値が大きいdirectionを選びやすい

        # TODO: ソートの精度の向上
        # TODO: 四方を少しはランダムに選ぶようにする
        # TODO: 近づきすぎない
        TODO: 待機をどう扱うか
        #"""

        directions: List[PointDiff] = []
        diff_to_target: PointDiff = self.target.point - self.point  # type: ignore

        # # roleに応じて、ターゲットの上下左右に寄らせる
        # role_dir: PointDiff = list(blockade_conv_table.keys())[self.role]  # type: ignore
        # if role_dir.x != 0:
        #     # targetとの相対位置と担当が異なる場合
        #     # H→P　といた時、 diff_to_target は (0, 1)。
        #     # このHの担当が (0, -1) なら、現状正しい。
        #     # そのため掛け算したときに符号が正なら修正する必要
        #     if diff_to_target.x * role_dir.x > 0:
        #         # 担当方向に進める
        #         directions.append(role_dir)
        # else:
        #     # targetとの相対位置と担当が異なる場合
        #     # TODO: refactor
        #     if diff_to_target.y * role_dir.y < 0:
        #         # 担当方向に進める
        #         directions.append(role_dir)

        distance = abs(diff_to_target.x) + abs(diff_to_target.y)
        random.shuffle(neighbour_diffs)
        if distance == 0:
            # ランダム
            directions += neighbour_diffs

        elif distance == 1:
            # すでに離れている方向の優先度を上げる
            directions += [self.point - self.target.point] + neighbour_diffs  # type: ignore

        elif 2 <= distance <= 4:
            # 十分近いので待機を優先
            # TODO: targetが捕まっているパターン
            directions += [PointDiff(0, 0)] + neighbour_diffs
        else:
            # 4より大きい場合
            # 近づく
            path = solve_route(self.point, self.target.point, floor)  # type: ignore
            if len(path) >= 2:
                next_point = path[1]
                direction = next_point - self.point
                # print(f"# {next_point}, {self.point}, {direction}")
                directions += [direction]

            directions += neighbour_diffs

            # if random.randint(0, distance) < abs(diff_to_target.x):
            #     directions += [
            #         PointDiff(1 if diff_to_target.x > 0 else -1, 0)
            #     ] + neighbour_diffs
            # else:
            #     directions += [
            #         PointDiff(0, 1 if diff_to_target.y > 0 else -1)
            #     ] + neighbour_diffs

        return directions


@dataclass
class Team:
    def __init__(self, humans: List[Human], target: Optional[Pet] = None):
        self.humans = humans
        self.target = target
        self.set_role()

    def set_role(self):
        for i, human in enumerate(self.humans):
            human.role = i % 4

    def select_target(self, pets):
        """平均して最も近いpetを選ぶ"""
        # TODO: 囲われているペットは無視する
        # TODO: 最短経路を考慮すべきか？

        pet_distance_sum: Dict[Pet, int] = {pet: 0 for pet in pets}

        for pet in pets:
            for human in self.humans:  # type: ignore
                distance = cal_distance(human, pet)
                pet_distance_sum[pet] += distance

        nearest_distance_sum = MAXINT
        for pet, distance_sum in pet_distance_sum.items():
            if distance_sum < nearest_distance_sum:
                self.target = pet


def initial_input():
    N = int(input())
    pets: List[Pet] = []
    for i in range(N):
        x, y, t = map(lambda x: int(x), input().split())
        pet = Pet(id=i + 1, kind=Kind(t), point=Point(x - 1 + MARGIN, y - 1 + MARGIN))
        pets.append(pet)

    M = int(input())
    humans: List[Human] = []
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
        action_str = ""
        partition_cands.refresh(humans, pets)

        team.select_target(pets)

        # humanの意志をリフレッシュ
        for human in humans:
            human.refresh(pets)

        for human in humans:
            # ターゲットを決める
            # TODO: 一度決めたらターゲットは当分更新しないべき？
            human.select_target(pets)

            distance_between_human_target = cal_distance(human, human.target)

            # TODO: 2しか離れてなくても、遠いところに置くことは可能。
            if 3 <= distance_between_human_target <= human.block_dist:
                # 優先度高いものほど左にする
                # get_dirs_priority
                blockade_dirs = get_dirs_priority(human.point, human.target.point)
                blockade_cands: List[Point] = [
                    human.point + dir for dir in blockade_dirs
                ]

                for blockade_cand in blockade_cands:
                    # その位置に壁やpartitionがなく、人やペットの制約もなければ、partitionを立てる
                    if (floor.get_tile(blockade_cand) == Tile.EMPTY) & (
                        partition_cands.get_tile(blockade_cand) == Tile.EMPTY
                    ):
                        human.next_blockade = blockade_cand
                        floor.update_tile(human.next_blockade, Tile.PARTITION)
                        break

            else:
                # 移動先の優先順位付
                directions = human.sort_directions()

                for direction in directions:
                    move_to_cand = human.point + direction
                    if floor.get_tile(move_to_cand) != Tile.EMPTY:
                        pass
                    else:
                        human.next_move = move_to_cand
                        # 進む先のtileはPartition候補から消す
                        partition_cands.update_tile(human.next_move, Tile.NOTUSE)
                        break

            action_char = human.next_action_char()
            action_str += action_char

            for human in humans:
                print(
                    f"# human id:{human.id}, role:{human.role}, target:{human.target}"
                )

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

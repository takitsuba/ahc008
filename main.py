from __future__ import annotations
from dataclasses import dataclass
from shutil import move
from typing import List, Union, Optional, Dict
from enum import Enum
import random
from collections import OrderedDict

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

    def get_tile(self, point: Point):
        row = point.x
        col = point.y
        return self.tiles[row][col]

    def update_tile(self, point: Point, tile: Tile):
        row = point.x
        col = point.y
        self.tiles[row][col] = tile

        # 行くべきでない場所を更新する
        # 何かしらで埋められた場合は、行くべきでない場所が増えたか確認し更新する
        if tile != Tile.EMPTY:
            for diff in neighbour_diffs:
                danger_cand = point + diff
                cnt = self.count_neighbor_filled(danger_cand)
                # 周囲を3つ以上何かに囲まれていてemptyならdangerに変更
                if (cnt >= 3) & (self.get_tile(point) == Tile.EMPTY):
                    # WARNING: 無限ループ
                    self.update_tile(danger_cand, Tile.DANGER)

    def count_neighbor_filled(self, point: Point):
        """argumentの隣がいくつEMPTYでないかを数える"""
        cnt = 0
        for diff in neighbour_diffs:
            neighbour = point + diff
            if self.get_tile(neighbour) != Tile.EMPTY:
                cnt += 1
        return cnt


floor = Floor(FLOOR_LEN, MARGIN)
assert len(floor.tiles) == FLOOR_LEN + MARGIN * 2
for row in floor.tiles:
    assert len(row) == FLOOR_LEN + MARGIN * 2


# def solve_route(start: Point, goal: Point):
#     def dfs(start, target, path, visited = set()):
#         path.append(start)
#         visited.add(start)
#         if start == goal:
#             return path
#         for neighbour in adj_list[start]:
#             if neighbour not in visited:
#                 result = dfs(adj_list, neighbour, target, path, visited)
#                 if result is not None:
#                     return result
#         path.pop()
#         return None


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


@dataclass
class Pet:
    id: int
    kind: Kind
    point: Point

    def move(self, action_char):
        diff = move_char_to_diff[action_char]
        self.point += diff

    def __hash__(self) -> int:
        return self.id


@dataclass
class Human:
    id: int
    point: Point
    team: Optional[Team] = None
    role: Optional[int] = None
    target: Optional[Pet] = None
    next_blockade: Optional[Point] = None
    next_move: Optional[Point] = None

    def select_target(self, pets):
        self.target = self.team.target  # type:ignore
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

        # roleに応じて、ターゲットの上下左右に寄らせる
        role_dir: PointDiff = list(blockade_conv_table.keys())[self.role]  # type: ignore
        if role_dir.x != 0:
            # targetとの相対位置と担当が異なる場合
            # H→P　といた時、 diff_to_target は (0, 1)。
            # このHの担当が (0, -1) なら、現状正しい。
            # そのため掛け算したときに符号が正なら修正する必要
            if diff_to_target.x * role_dir.x > 0:
                # 担当方向に進める
                directions.append(role_dir)
        else:
            # targetとの相対位置と担当が異なる場合
            # TODO: refactor
            if diff_to_target.y * role_dir.y < 0:
                # 担当方向に進める
                directions.append(role_dir)

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
            if random.randint(0, distance) < abs(diff_to_target.x):
                directions += [
                    PointDiff(1 if diff_to_target.x > 0 else -1, 0)
                ] + neighbour_diffs
            else:
                directions += [
                    PointDiff(0, 1 if diff_to_target.y > 0 else -1)
                ] + neighbour_diffs

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

            # 距離が3なら2のところを埋める
            # 8パターンなら総当たりで良いと考えた
            distance_between_human_target = cal_distance(human, human.target)
            if distance_between_human_target == 3:
                # 優先度高いものほど左にする
                blockade_cands = []

                # TODO: 回転すれば省略できそう
                # 斜めの方が優先度高い
                # TODO: diffのクラス
                diagonal_diffs = [[1, 1], [1, -1], [-1, 1], [-1, -1]]
                for diff in diagonal_diffs:
                    blockade_point = Point(
                        human.target.point.x + diff[0], human.target.point.y + diff[1]
                    )
                    if cal_distance_points(human.point, blockade_point) == 2:
                        blockade_cands.append(blockade_point)

                # TODO: 関数を作成する
                # x,y どちらかは同じ座標の場合
                same_axis_diffs = [[2, 0], [-2, 0], [0, 2], [0, -2]]
                for diff in same_axis_diffs:
                    blockade_point = Point(
                        human.target.point.x + diff[0], human.target.point.y + diff[1]
                    )
                    if cal_distance_points(human.point, blockade_point) == 1:
                        blockade_cands.append(blockade_point)

                    # import pdb; pdb.set_trace()

                for blockade_cand in blockade_cands:
                    # その位置に壁やpartitionがなく、人やペットの制約もなければ、partitionを立てる
                    if (floor.get_tile(blockade_cand) == Tile.EMPTY) & (
                        partition_cands.get_tile(blockade_cand) == Tile.EMPTY
                    ):
                        human.next_blockade = blockade_cand
                        floor.update_tile(human.next_blockade, Tile.PARTITION)
                        break

                # import pdb; pdb.set_trace()

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

                # import pdb; pdb.set_trace()

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

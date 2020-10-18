from itertools import product
from typing import List, Tuple

import pandas as pandas
from pandas import DataFrame
from pulp import *

# List[Tuple[素材数, 成功率(％)]]
table_r = [
    (1, 100),
    (1, 99),
    (1, 96),
    (2, 91),
    (2, 83),
    (2, 74),
    (2, 64),
    (3, 54),
    (3, 46),
    (3, 38),
    (3, 32),
    (4, 27),
    (4, 23),
    (4, 20),
    (4, 18),
    (5, 16),
    (5, 14),
    (5, 13),
    (5, 13)
]


def calc(table: List[Tuple[int, int]], start_level: int, goal_level: int, max_soul_count: int) -> None:
    """最適解を計算して出力する

    :param table:素材数・成功率のテーブル
    :param start_level:開始レベル
    :param goal_level:目標レベル
    :param max_soul_count:使用できるシラズのまもり魂の下図
    """

    # 入力チェック
    if start_level >= goal_level:
        print('【計算結果】')
        print('強化の必要なし.')
        return

    # 期待値テーブル(シラズの精花、シラズのまもり魂)を算出する
    data_df = DataFrame.from_records(table_r)
    data_df.columns = ['flower_count', 'percent']

    problem_df = DataFrame()
    for level in range(0, len(data_df)):
        data_record = data_df[level: level + 1]
        percent = data_record['percent'].values[0]
        flower_count = data_record['flower_count'].values[0]
        for soul_count in range(0, max_soul_count + 1):
            flower_exp = 100.0 / min((percent + 5 * soul_count), 100) * flower_count
            soul_exp = 100.0 / min((percent + 5 * soul_count), 100) * soul_count
            problem_df = problem_df.append({
                'level': level + 1,
                'soul_count': soul_count,
                'flower_exp': flower_exp,
                'soul_exp': soul_exp
            }, ignore_index=True)
    problem_df = problem_df[['level', 'soul_count', 'flower_exp', 'soul_exp']]

    # 変数(まもり魂使用数/回の選択肢)を定義
    pair_list = []
    for level in range(0, len(data_df)):
        for soul_count in range(0, max_soul_count + 1):
            pair_list.append((level, soul_count))
    problem_df['Var'] = [LpVariable('v%d_%d' % (i + 1, j), lowBound=0, upBound=1, cat=LpBinary) for i, j in pair_list]

    # 問題を定義
    problem_model = LpProblem(sense=LpMinimize)

    # 目的関数(精花の総和の最小化)を設定
    problem_model.setObjective(lpDot(problem_df['flower_exp'], problem_df['Var']))

    # 制約条件(各レベルにおける選択肢)を設定
    for level in range(0, len(data_df)):
        index_1 = level * (max_soul_count + 1)
        index_2 = index_1 + (max_soul_count + 1)
        if start_level <= level + 1 < goal_level:
            problem_model += lpSum(problem_df['Var'][index_1:index_2]) == 1
        else:
            problem_model += lpSum(problem_df['Var'][index_1:index_2]) == 0

    # 制約条件(魂使用回数の条件)を設定
    problem_model += lpDot(problem_df['Var'], problem_df['soul_exp']) <= max_soul_count

    # 解く
    print('-' * 80)
    problem_model.solve()
    print('-' * 80)

    # 結果を表示
    print(f'レベル{start_level}からレベル{goal_level}まで、まもり魂を{max_soul_count}個使う場合の分析結果：')
    print(f'・精花使用回数の期待値は、{round(value(problem_model.objective) * 10) / 10.0}個)')
    if problem_model.status == LpStatusOptimal:
        problem_df['Val'] = problem_df['Var'].apply(value)
        for level in range(0, len(data_df)):
            index_1 = level * (max_soul_count + 1)
            index_2 = index_1 + (max_soul_count + 1)
            sliced_value = problem_df['Val'][index_1: index_2].values
            for soul_count in range(0, max_soul_count + 1):
                if sliced_value[soul_count] > 0 and soul_count > 0:
                    percent = min(table_r[level][1] + soul_count * 5, 100)
                    print(f'・レベル{level + 1} では まもり魂を1回あたり{soul_count}個使用、成功率は{percent}％')
                    break
    else:
        print('解けませんでした')


if __name__ == '__main__':
    # print(calc(table_r, 1, 15, 0))
    # calc(table_r, 1, 15, 100)
    print(calc(table_r, 1, 20, 20))
    print(calc(table_r, 1, 20, 100))

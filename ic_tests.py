import os
import datetime as dt
import pandas as pd
from husfort.qutility import SFG
from regroups import CLibRegroups


def cal_ic_tests(pair: tuple[str, str], delay: int,
                 bgn_date: str, stp_date: str, factors: list[str],
                 regroups_dir: str) -> pd.Series:
    lib_regroup_reader = CLibRegroups(pair, delay, regroups_dir).get_lib_reader()
    df = lib_regroup_reader.read_by_conditions(conditions=[
        ("trade_date", ">=", bgn_date),
        ("trade_date", "<", stp_date),
    ], value_columns=["trade_date", "factor", "value"])
    pivot_df = pd.pivot_table(data=df, index="trade_date", columns="factor", values="value")
    corr_srs = pivot_df.corr(method="spearman")["diff_return"][factors]
    return corr_srs


def cal_ic_tests_pairs(instruments_pairs: list[tuple[str, str]], diff_ret_delays: list[int],
                       bgn_date: str, stp_date: str, factors: list[str],
                       regroups_dir: str, ic_tests_dir: str):
    ic_tests_dfs = []
    for pair in instruments_pairs:
        pair_id = f"{pair[0]}_{pair[1]}"
        pair_res = {}
        for delay in diff_ret_delays:
            pair_res[f"T{delay}"] = cal_ic_tests(pair, delay, bgn_date, stp_date, factors, regroups_dir)
        pair_df = pd.DataFrame(pair_res).reset_index()
        pair_df["pair"] = pair_id
        ic_tests_dfs.append(pair_df)
        print(f"{dt.datetime.now()} [INF] ic-tests for {SFG(pair_id)} calculated")
    ic_tests_df = pd.concat(ic_tests_dfs, axis=0, ignore_index=True)
    ic_tests_file = f"ic_tests.csv"
    ic_tests_path = os.path.join(ic_tests_dir, ic_tests_file)
    ic_tests_df.to_csv(ic_tests_path, index=False, float_format="%.8f")

    for delay in diff_ret_delays:
        delay_sum_df = pd.pivot_table(data=ic_tests_df, index="factor", columns="pair", values=f"T{delay}")
        delay_sum_file = f"ic_tests.sum.T{delay}.csv"
        delay_sum_path = os.path.join(ic_tests_dir, delay_sum_file)
        delay_sum_df.to_csv(delay_sum_path, float_format="%.8f")
    return 0

import os
import pandas as pd
from husfort.qsqlite import CLibMajorReturn


def cal_diff_returns(
        instru_a: str, instru_b: str,
        major_return_save_dir: str,
        bgn_date: str, stp_date: str,
        diff_returns_dir: str,
):
    lib_reader = CLibMajorReturn(instrument=instru_a, lib_save_dir=major_return_save_dir).get_lib_reader()
    return_df_a = lib_reader.read_by_conditions(
        conditions=[
            ("trade_date", ">=", bgn_date),
            ("trade_date", "<", stp_date),
        ], value_columns=["trade_date", "major_return"]
    ).set_index("trade_date").rename(mapper={"major_return": instru_a}, axis=1)
    lib_reader.close()

    lib_reader = CLibMajorReturn(instrument=instru_b, lib_save_dir=major_return_save_dir).get_lib_reader()
    return_df_b = lib_reader.read_by_conditions(
        conditions=[
            ("trade_date", ">=", bgn_date),
            ("trade_date", "<", stp_date),
        ], value_columns=["trade_date", "major_return"]
    ).set_index("trade_date").rename(mapper={"major_return": instru_b}, axis=1)
    lib_reader.close()

    diff_return_df = pd.merge(left=return_df_a, right=return_df_b, left_index=True, right_index=True, how="outer")
    if len(return_df_a) != len(return_df_b):
        print(f"... [ERR] length of {instru_a} != length of {instru_b}")
        raise ValueError
    if len(return_df_a) != len(diff_return_df):
        print(f"... [ERR] length of {instru_a} != length of diff returns")
        raise ValueError
    diff_return_df["diff_return"] = (diff_return_df[instru_a] - diff_return_df[instru_b]) * 0.5
    diff_return_file = f"diff_return.{instru_a}_{instru_b}.csv.gz"
    diff_return_path = os.path.join(diff_returns_dir, diff_return_file)
    diff_return_df.to_csv(diff_return_path, float_format="%.8f")
    return 0


def cal_diff_returns_groups(
        instruments_group: list[tuple[str, str]],
        major_return_save_dir: str,
        bgn_date: str, stp_date: str,
        diff_returns_dir: str,
):
    for instru_a, instru_b in instruments_group:
        cal_diff_returns(instru_a, instru_b, major_return_save_dir, bgn_date, stp_date, diff_returns_dir)
    return 0

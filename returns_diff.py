import pandas as pd
from husfort.qsqlite import CLibMajorReturn, CQuickSqliteLib, CLib1Tab1, CTable


class CLibDiffReturn(CQuickSqliteLib):
    def __init__(self, pair: tuple[str, str], lib_save_dir: str):
        self.instru_a, self.instru_b = pair
        lib_name = f"diff_return.{self.instru_a}_{self.instru_b}.db"
        super().__init__(lib_name, lib_save_dir)

    def get_lib_struct(self) -> CLib1Tab1:
        return CLib1Tab1(
            lib_name=self.lib_name,
            table=CTable(
                {
                    "table_name": "diff_return",
                    "primary_keys": {"trade_date": "TEXT"},
                    "value_columns": {
                        "instru_a": "REAL",
                        "instru_b": "REAL",
                        "diff_return": "REAL",
                    },
                }
            )
        )


def cal_diff_returns(
        pair: tuple[str, str],
        major_return_save_dir: str,
        run_mode: str, bgn_date: str, stp_date: str,
        diff_returns_dir: str,
):
    instru_a, instru_b = pair
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

    lib_writer = CLibDiffReturn(pair, diff_returns_dir).get_lib_writer(run_mode=run_mode)
    lib_writer.update(update_df=diff_return_df, using_index=True)
    lib_writer.commit()
    lib_writer.close()
    return 0


def cal_diff_returns_pairs(instruments_pairs: list[tuple[str, str]], **kwargs):
    for pair in instruments_pairs:
        cal_diff_returns(pair=pair, **kwargs)
    return 0

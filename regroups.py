from itertools import product
import datetime as dt
import pandas as pd
from husfort.qutility import SFG
from husfort.qcalendar import CCalendar
from husfort.qsqlite import CQuickSqliteLib, CLib1Tab1, CTable
from returns_diff import CLibDiffReturn
from exposures import CLibFactorExposure


class CLibRegroups(CQuickSqliteLib):
    def __init__(self, pair: tuple[str, str], delay: int, lib_save_dir: str):
        self.pair = pair
        self.instru_a, self.instru_b = self.pair
        lib_name = f"regroups.{self.instru_a}_{self.instru_b}.D{delay}.db"
        super().__init__(lib_name, lib_save_dir)

    def get_lib_struct(self) -> CLib1Tab1:
        return CLib1Tab1(
            lib_name=self.lib_name,
            table=CTable(
                {
                    "table_name": "regroups",
                    "primary_keys": {"trade_date": "TEXT", "factor": "TEXT"},
                    "value_columns": {"value": "REAL"},
                }
            )
        )


def cal_regroups(
        pair: tuple[str, str], delay: int,
        factors: list[str],
        run_mode: str, bgn_date: str, stp_date: str,
        diff_returns_dir: str,
        factors_exposure_dir: str,
        regroups_dir: str,
        calendar: CCalendar,

):
    instru_a, instru_b = pair
    pair_id = f"{instru_a}_{instru_b}"
    diff_ret_iter_dates = calendar.get_iter_list(bgn_date, stp_date)
    diff_ret_bgn_date, diff_ret_end_date = diff_ret_iter_dates[0], diff_ret_iter_dates[-1]
    exposure_bgn_date = calendar.get_next_date(diff_ret_bgn_date, -delay)
    exposure_end_date = calendar.get_next_date(diff_ret_end_date, -delay)
    exposure_iter_dates = ([calendar.get_next_date(diff_ret_bgn_date, t) for t in range(-delay, 0)]
                           + diff_ret_iter_dates[:-delay])
    bridge_df = pd.DataFrame({"exposure_dates": exposure_iter_dates, "test_ret_dates": diff_ret_iter_dates})

    factors_exposure_dfs = []
    for factor in factors:
        lib_reader = CLibFactorExposure(factor, factors_exposure_dir).get_lib_reader()
        factor_exposure_df = lib_reader.read_by_conditions(conditions=[
            ("trade_date", ">=", exposure_bgn_date),
            ("trade_date", "<=", exposure_end_date),
            ("pair", "=", pair_id),
        ], value_columns=["trade_date", "value"])
        factor_exposure_df["factor"] = factor
        factors_exposure_dfs.append(factor_exposure_df)
    factors_exposure_df = pd.concat(factors_exposure_dfs, axis=0, ignore_index=True)
    merged_df = pd.merge(left=bridge_df, right=factors_exposure_df,
                         left_on="exposure_dates", right_on="trade_date", how="right")
    exposure_df = merged_df[["test_ret_dates", "factor", "value"]].rename(
        mapper={"test_ret_dates": "trade_date"}, axis=1)

    lib_reader = CLibDiffReturn(pair, diff_returns_dir).get_lib_reader()
    diff_ret_df = lib_reader.read_by_conditions(conditions=[
        ("trade_date", ">=", diff_ret_bgn_date),
        ("trade_date", "<=", diff_ret_end_date),
    ], value_columns=["trade_date", "diff_return"]).rename(mapper={"diff_return": "value"}, axis=1)
    diff_ret_df["factor"] = "diff_return"
    diff_ret_df = diff_ret_df[["trade_date", "factor", "value"]]

    update_df = pd.concat([exposure_df, diff_ret_df], axis=0, ignore_index=True)
    lib_regroups_writer = CLibRegroups(pair, delay, regroups_dir).get_lib_writer(run_mode)
    lib_regroups_writer.update(update_df, using_index=False)
    lib_regroups_writer.commit()
    lib_regroups_writer.close()
    return 0


def cal_regroups_pairs(instruments_pairs: list[tuple[str, str]], diff_ret_delays: list[int], **kwargs):
    for (pair, delay) in product(instruments_pairs, diff_ret_delays):
        cal_regroups(pair, delay, **kwargs)
        print(f"{dt.datetime.now()} [INF] Pair {SFG(f'{pair[0]}-{pair[1]}-T{delay}')} is calculated")
    return 0

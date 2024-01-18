import pandas as pd
from husfort.qsqlite import CQuickSqliteLib, CLib1Tab1, CTable
from husfort.qcalendar import CCalendar
from returns_diff import CLibDiffReturn


class CLibFactorExposure(CQuickSqliteLib):
    def __init__(self, factor: str, lib_save_dir: str):
        self.factor = factor
        lib_name = f"factor_exposure.{factor}.db"
        super().__init__(lib_name, lib_save_dir)

    def get_lib_struct(self) -> CLib1Tab1:
        return CLib1Tab1(
            lib_name=self.lib_name,
            table=CTable(
                {
                    "table_name": self.factor,
                    "primary_keys": {"trade_date": "TEXT", "pair": "TEXT"},
                    "value_columns": {"value": "REAL"},
                }
            )
        )


class CFactorExposure(object):
    def __init__(self, factor: str, factors_exposure_dir: str, instruments_pairs: list[tuple[str, str]]):
        self.factor = factor
        self.factors_exposure_dir = factors_exposure_dir
        self.instruments_pairs = instruments_pairs

    def cal(self, bgn_date: str, stp_date: str, calendar: CCalendar) -> pd.DataFrame:
        pass

    @staticmethod
    def truncate_before_bgn(df: pd.DataFrame, bgn_date: str) -> pd.DataFrame:
        return df.truncate(before=bgn_date)

    def save(self, factor_exposure_df: pd.DataFrame, run_mode: str):
        lib_writer = CLibFactorExposure(self.factor, self.factors_exposure_dir).get_lib_writer(run_mode)
        lib_writer.update(update_df=factor_exposure_df, using_index=True)
        lib_writer.commit()
        lib_writer.close()
        return 0

    def main(self, run_mode: str, bgn_date: str, stp_date: str, calendar: CCalendar):
        factor_exposure_df = self.cal(bgn_date, stp_date, calendar)
        self.save(factor_exposure_df, run_mode)
        return 0


class CFactorExposureLagRet(CFactorExposure):
    def __init__(self, lag: int, diff_returns_dir: str, **kwargs):
        self.lag = lag
        self.diff_returns_dir = diff_returns_dir
        super().__init__(factor=f"L{lag:02d}", **kwargs)

    def cal(self, bgn_date: str, stp_date: str, calendar: CCalendar) -> pd.DataFrame:
        iter_dates = calendar.get_iter_list(bgn_date, stp_date)
        base_date = calendar.get_next_date(iter_dates[0], shift=-self.lag)
        dfs_list = []
        for (instru_a, instru_b) in self.instruments_pairs:
            pair = f"{instru_a}_{instru_b}"
            lib_diff_return_reader = CLibDiffReturn(instru_a, instru_b, self.diff_returns_dir).get_lib_reader()
            pair_df = lib_diff_return_reader.read_by_conditions(
                conditions=[
                    ("trade_date", ">=", base_date),
                    ("trade_date", "<", stp_date),
                ], value_columns=["trade_date", "diff_return"]
            ).set_index("trade_date")
            pair_df[self.factor] = pair_df["diff_return"].shift(1)
            pair_df.truncate(before=bgn_date)
            pair_df["pair"] = pair
            dfs_list.append(pair_df[["pair", self.factor]])
        df = pd.concat(dfs_list, axis=0, ignore_index=False)
        df.sort_values(by=["trade_date", "pair"], inplace=True)
        return self.truncate_before_bgn(df, bgn_date)

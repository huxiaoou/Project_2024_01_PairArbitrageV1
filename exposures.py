import datetime as dt
import numpy as np
import pandas as pd
from husfort.qutility import SFG
from husfort.qsqlite import CQuickSqliteLib, CLib1Tab1, CTable, CLibFactor
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
        """

        :param bgn_date:
        :param stp_date:
        :param calendar:
        :return: a pd.DataFrame, with index = list[str["YYYYMMDD"]], columns = ["pair", factor]
        """
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
        factor_exposure_df = self.truncate_before_bgn(factor_exposure_df, bgn_date)
        self.save(factor_exposure_df, run_mode)
        print(f"{dt.datetime.now()} [INF] Factor {SFG(self.factor)} is calculated")
        return 0


class _CFactorExposureEndogenous(CFactorExposure):
    def __init__(self, factor: str, diff_returns_dir: str, **kwargs):
        self.diff_returns_dir = diff_returns_dir
        super().__init__(factor=factor, **kwargs)

    def _get_base_date(self, bgn_date: str, calendar: CCalendar) -> str:
        pass

    def _get_diff_return(self, instru_a: str, instru_b: str, base_date: str, stp_date: str) -> pd.DataFrame:
        lib_diff_return_reader = CLibDiffReturn(instru_a, instru_b, self.diff_returns_dir).get_lib_reader()
        pair_df = lib_diff_return_reader.read_by_conditions(
            conditions=[
                ("trade_date", ">=", base_date),
                ("trade_date", "<", stp_date),
            ], value_columns=["trade_date", "diff_return"]
        ).set_index("trade_date")
        return pair_df

    def _cal_factor(self, diff_ret_srs: pd.Series) -> pd.Series:
        pass

    def cal(self, bgn_date: str, stp_date: str, calendar: CCalendar) -> pd.DataFrame:
        base_date = self._get_base_date(bgn_date, calendar)
        dfs_list = []
        for (instru_a, instru_b) in self.instruments_pairs:
            pair = f"{instru_a}_{instru_b}"
            pair_df = self._get_diff_return(instru_a, instru_b, base_date, stp_date)
            pair_df[self.factor] = self._cal_factor(pair_df["diff_return"])
            pair_df["pair"] = pair
            dfs_list.append(pair_df[["pair", self.factor]])
        df = pd.concat(dfs_list, axis=0, ignore_index=False)
        df.sort_values(by=["trade_date", "pair"], inplace=True)
        return df


class CFactorExposureLagRet(_CFactorExposureEndogenous):
    def __init__(self, lag: int, diff_returns_dir: str, **kwargs):
        self.lag = lag
        factor = f"LAG{lag:02d}"
        super().__init__(factor, diff_returns_dir, **kwargs)

    def _get_base_date(self, bgn_date: str, calendar: CCalendar) -> str:
        return calendar.get_next_date(bgn_date, -self.lag)

    def _cal_factor(self, diff_ret_srs: pd.Series) -> pd.Series:
        return diff_ret_srs.shift(self.lag)


class CFactorExposureEWM(_CFactorExposureEndogenous):
    def __init__(self, fast: int, slow: int, diff_returns_dir: str, fix_base_date: str, **kwargs):
        self.fix_base_date = fix_base_date
        self.fast, self.slow = fast, slow
        factor = f"F{int(fast * 100):02d}S{int(slow * 100):02d}"
        super().__init__(factor, diff_returns_dir, **kwargs)

    def _get_base_date(self, bgn_date: str, calendar: CCalendar) -> str:
        return self.fix_base_date

    def _cal_factor(self, diff_ret_srs: pd.Series) -> pd.Series:
        fast_srs = diff_ret_srs.ewm(alpha=self.fast, adjust=False).mean()
        slow_srs = diff_ret_srs.ewm(alpha=self.slow, adjust=False).mean()
        return fast_srs - slow_srs


class CFactorExposureVolatility(_CFactorExposureEndogenous):
    def __init__(self, win: int, k: int, diff_returns_dir: str, **kwargs):
        self.win, self.k = win, k
        factor = f"VTY{win:02d}K{k:02d}"
        super().__init__(factor, diff_returns_dir, **kwargs)

    def _get_base_date(self, bgn_date: str, calendar: CCalendar) -> str:
        return calendar.get_next_date(bgn_date, -self.win - self.k + 2)

    def _cal_factor(self, diff_ret_srs: pd.Series) -> pd.Series:
        volatility: pd.Series = diff_ret_srs.rolling(window=self.win).std() * np.sqrt(250)
        volatility_ma = volatility.rolling(window=self.k).mean()
        return volatility / volatility_ma - 1


class CFactorExposureTNR(_CFactorExposureEndogenous):
    def __init__(self, win: int, k: int, diff_returns_dir: str, **kwargs):
        self.win, self.k = win, k
        factor = f"TNR{win:02d}K{k:02d}"
        super().__init__(factor, diff_returns_dir, **kwargs)

    def _get_base_date(self, bgn_date: str, calendar: CCalendar) -> str:
        return calendar.get_next_date(bgn_date, -self.win - self.k + 2)

    def _cal_factor(self, diff_ret_srs: pd.Series) -> pd.Series:
        rng_sum_abs = diff_ret_srs.abs().rolling(window=self.win).sum()
        rng_abs_sum = diff_ret_srs.rolling(window=self.win).sum().abs()
        tnr: pd.Series = rng_abs_sum / rng_sum_abs
        tnr_ma = tnr.rolling(window=self.k).mean()
        return tnr / tnr_ma - 1


class _CFactorExposureFromInstruExposure(CFactorExposure):
    def __init__(self, factor: str, instru_factor_exposure_dir: str, **kwargs):
        self.instru_factor_exposure_dir = instru_factor_exposure_dir
        super().__init__(factor=factor, **kwargs)

    def cal(self, bgn_date: str, stp_date: str, calendar: CCalendar) -> pd.DataFrame:
        lib_instru_factor = CLibFactor(self.factor, self.instru_factor_exposure_dir).get_lib_reader()
        dfs_list = []
        for (instru_a, instru_b) in self.instruments_pairs:
            pair = f"{instru_a}_{instru_b}"
            instru_factor_exposure = {}
            for instru in [instru_a, instru_b]:
                instru_df = lib_instru_factor.read_by_conditions(
                    conditions=[
                        ("trade_date", ">=", bgn_date),
                        ("trade_date", "<", stp_date),
                        ("instrument", "=", instru)
                    ], value_columns=["trade_date", "value"]
                ).set_index("trade_date")
                instru_factor_exposure[instru] = instru_df["value"]
            pair_df = pd.DataFrame(instru_factor_exposure)
            pair_df[self.factor] = (pair_df[instru_a] - pair_df[instru_b]).fillna(0)
            pair_df["pair"] = pair
            dfs_list.append(pair_df[["pair", self.factor]])
        df = pd.concat(dfs_list, axis=0, ignore_index=False)
        df.sort_values(by=["trade_date", "pair"], inplace=True)
        return df


class CFactorExposureBasisa(_CFactorExposureFromInstruExposure):
    def __init__(self, win: int, instru_factor_exposure_dir: str, **kwargs):
        factor = f"BASISA{win:03d}"
        super().__init__(factor, instru_factor_exposure_dir, **kwargs)


class CFactorExposureCTP(_CFactorExposureFromInstruExposure):
    def __init__(self, win: int, instru_factor_exposure_dir: str, **kwargs):
        factor = f"CTP{win:03d}"
        super().__init__(factor, instru_factor_exposure_dir, **kwargs)


class CFactorExposureCVP(_CFactorExposureFromInstruExposure):
    def __init__(self, win: int, instru_factor_exposure_dir: str, **kwargs):
        factor = f"CVP{win:03d}"
        super().__init__(factor, instru_factor_exposure_dir, **kwargs)


class CFactorExposureCSP(_CFactorExposureFromInstruExposure):
    def __init__(self, win: int, instru_factor_exposure_dir: str, **kwargs):
        factor = f"CSP{win:03d}"
        super().__init__(factor, instru_factor_exposure_dir, **kwargs)


class CFactorExposureRSBR(_CFactorExposureFromInstruExposure):
    def __init__(self, win: int, instru_factor_exposure_dir: str, **kwargs):
        factor = f"RSBR{win:03d}"
        super().__init__(factor, instru_factor_exposure_dir, **kwargs)


class CFactorExposureRSLR(_CFactorExposureFromInstruExposure):
    def __init__(self, win: int, instru_factor_exposure_dir: str, **kwargs):
        factor = f"RSLR{win:03d}"
        super().__init__(factor, instru_factor_exposure_dir, **kwargs)


class CFactorExposureSKEW(_CFactorExposureFromInstruExposure):
    def __init__(self, win: int, instru_factor_exposure_dir: str, **kwargs):
        factor = f"SKEW{win:03d}"
        super().__init__(factor, instru_factor_exposure_dir, **kwargs)


class CFactorExposureMTM(_CFactorExposureFromInstruExposure):
    def __init__(self, win: int, instru_factor_exposure_dir: str, **kwargs):
        if win != 1:
            print(f"win = {win} for factor MTM is wrong")
            raise ValueError
        factor = f"MTM"
        super().__init__(factor, instru_factor_exposure_dir, **kwargs)


class CFactorExposureMTMS(_CFactorExposureFromInstruExposure):
    def __init__(self, win: int, instru_factor_exposure_dir: str, **kwargs):
        factor = f"MTMS{win:03d}"
        super().__init__(factor, instru_factor_exposure_dir, **kwargs)


class CFactorExposureTSA(_CFactorExposureFromInstruExposure):
    def __init__(self, win: int, instru_factor_exposure_dir: str, **kwargs):
        factor = f"TSA{win:03d}"
        super().__init__(factor, instru_factor_exposure_dir, **kwargs)


class CFactorExposureTSLD(_CFactorExposureFromInstruExposure):
    def __init__(self, win: int, instru_factor_exposure_dir: str, **kwargs):
        factor = f"TSLD{win:03d}"
        super().__init__(factor, instru_factor_exposure_dir, **kwargs)

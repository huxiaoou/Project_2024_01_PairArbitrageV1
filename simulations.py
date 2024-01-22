import datetime as dt
import multiprocessing as mp
import numpy as np
import pandas as pd
from itertools import product
from husfort.qutility import SFG
from husfort.qsqlite import CQuickSqliteLib, CLib1Tab1, CTable
from regroups import CLibRegroups


class CLibSimu(CQuickSqliteLib):
    def __init__(self, simu_id: str, lib_save_dir: str):
        self.simu_id = simu_id
        lib_name = f"simu.{self.simu_id}.db"
        super().__init__(lib_name, lib_save_dir)

    def get_lib_struct(self) -> CLib1Tab1:
        return CLib1Tab1(
            lib_name=self.lib_name,
            table=CTable(
                {
                    "table_name": "simu",
                    "primary_keys": {"trade_date": "TEXT"},
                    "value_columns": {
                        "signal": "REAL",
                        "rawRet": "REAL",
                        "dltWgt": "REAL",
                        "cost": "REAL",
                        "netRet": "REAL",
                        "cumNetRet": "REAL",
                    },
                }
            )
        )


class CQuickSimu(object):
    def __init__(self, simu_id: str, df: pd.DataFrame, sig: str, ret: str, cost_rate: float):
        """

        :param simu_id:
        :param df: has date-like index, with string format
        :param sig:
        :param ret:
        :param cost_rate:
        """
        self.simu_id = simu_id
        self.df = df
        self.sig = sig
        self.ret = ret
        self.cost_rate = cost_rate

    def __save(self, run_mode: str, simulations_dir: str):
        lib_simu_writer = CLibSimu(simu_id=self.simu_id, lib_save_dir=simulations_dir).get_lib_writer(run_mode)
        lib_simu_writer.update(update_df=self.df[[
            "signal", "rawRet", "dltWgt", "cost", "netRet", "cumNetRet"
        ]], using_index=True)
        lib_simu_writer.commit()
        lib_simu_writer.close()
        return 0

    def main(self, run_mode: str, simulations_dir: str):
        self.df["rawRet"] = self.df[self.sig] * self.df[self.ret]
        self.df["dltWgt"] = self.df[self.sig] - self.df[self.sig].shift(1).fillna(0)
        self.df["cost"] = self.df["dltWgt"].abs() * self.cost_rate
        self.df["netRet"] = self.df["rawRet"] - self.df["cost"]
        self.df["cumNetRet"] = self.df["netRet"].cumsum()
        self.__save(run_mode=run_mode, simulations_dir=simulations_dir)
        print(f"{dt.datetime.now()} [INF] simulation for {SFG(self.simu_id):>48s} calculated")
        return 0


def cal_simulations(pair: tuple[str, str], delay: int,
                    run_mode: str, bgn_date: str, stp_date: str, factors: list[str], cost_rate: float,
                    regroups_dir: str, simulations_dir: dir):
    pair_id = f"{pair[0]}_{pair[1]}"
    lib_regroup_reader = CLibRegroups(pair, delay, regroups_dir).get_lib_reader()
    df = lib_regroup_reader.read_by_conditions(conditions=[
        ("trade_date", ">=", bgn_date),
        ("trade_date", "<", stp_date),
    ], value_columns=["trade_date", "factor", "value"])
    pivot_df = pd.pivot_table(data=df, index="trade_date", columns="factor", values="value")
    for factor in factors:
        simu_df = pivot_df[[factor, "diff_return"]].copy()
        simu_df["signal"] = np.sign(pivot_df[factor])
        simu = CQuickSimu(simu_id=f"{pair_id}.{factor}.T{delay}", df=simu_df,
                          sig="signal", ret="diff_return", cost_rate=cost_rate)
        simu.main(run_mode=run_mode, simulations_dir=simulations_dir)
    return 0


def cal_simulations_pairs(proc_qty: int, instruments_pairs: list[tuple[str, str]], diff_ret_delays: list[int],
                          **kwargs):
    pool = mp.Pool(processes=proc_qty)
    for (pair, delay) in product(instruments_pairs, diff_ret_delays):
        pool.apply_async(cal_simulations, args=(pair, delay), kwds=kwargs)
    pool.close()
    pool.join()
    return 0

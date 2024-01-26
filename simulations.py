import datetime as dt
import multiprocessing as mp
import numpy as np
import pandas as pd
from itertools import product
from husfort.qutility import SFG
from husfort.qcalendar import CCalendar
from husfort.qsqlite import CQuickSqliteLib, CLib1Tab1, CTable
from regroups import CLibRegroups
from returns_diff import CLibDiffReturn
from mlModels import CLibPredictions


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
        simu_id = f"{pair_id}.{factor}.T{delay}"
        simu = CQuickSimu(simu_id=simu_id, df=simu_df, sig="signal", ret="diff_return", cost_rate=cost_rate)
        simu.main(run_mode=run_mode, simulations_dir=simulations_dir)
    return 0


def cal_simulations_pairs(instruments_pairs: list[tuple[str, str]], diff_ret_delays: list[int], proc_qty: int = None,
                          **kwargs):
    pool = mp.Pool(processes=proc_qty) if proc_qty else mp.Pool()
    for (pair, delay) in product(instruments_pairs, diff_ret_delays):
        pool.apply_async(cal_simulations, args=(pair, delay), kwds=kwargs)
    pool.close()
    pool.join()
    return 0


def cal_simulations_ml(ml_model_id: str, instrument_pairs: list[tuple[str, str]],
                       run_mode: str, bgn_date: str, stp_date: str, cost_rate: float,
                       calendar: CCalendar,
                       predictions_dir: str, diff_returns_dir: str, simulations_dir: str, ):
    pair_ids = ["_".join(p) for p in instrument_pairs]
    ret_dates = calendar.get_iter_list(bgn_date, stp_date)
    sig_dates = calendar.shift_iter_dates(ret_dates, -2)

    # --- load signal
    lib_pred = CLibPredictions(ml_model_id, predictions_dir).get_lib_reader()
    pred_df = lib_pred.read_by_conditions(conditions=[
        ("trade_date", ">=", sig_dates[0]),
        ("trade_date", "<=", sig_dates[-1]),
    ], value_columns=["trade_date", "pair", "value"])
    sig_df = pd.pivot_table(pred_df, index="trade_date", columns="pair", values="value")
    sig_df = sig_df[pair_ids].applymap(lambda z: 2 * z - 1).fillna(0)
    dlt_wgt_abs_sum = sig_df.abs().sum(axis=1)
    sig_df = sig_df.div(dlt_wgt_abs_sum, axis=0)

    # --- align signal
    bridge = pd.DataFrame({"ret_date": ret_dates, "sig_date": sig_dates})
    aligned_sig_df = pd.merge(
        left=bridge, right=sig_df,
        left_on="sig_date", right_index=True, how="left"
    )
    aligned_sig_df = aligned_sig_df.set_index("ret_date").drop(axis=1, labels=["sig_date"]).fillna(0)

    # --- diff weight
    dlt_wgt_df = aligned_sig_df - aligned_sig_df.shift(1).fillna(0)

    # --- load ret
    ret_dfs: list[pd.DataFrame] = []
    for pair, pair_id in zip(instrument_pairs, pair_ids):
        lib_diff_return = CLibDiffReturn(pair, diff_returns_dir).get_lib_reader()
        pair_df = lib_diff_return.read_by_conditions(
            conditions=[
                ("trade_date", ">=", ret_dates[0]),
                ("trade_date", "<=", ret_dates[-1]),
            ], value_columns=["trade_date", "diff_return"]
        )
        pair_df["pair"] = pair_id
        ret_dfs.append(pair_df)
    ret_con_df = pd.concat(ret_dfs, axis=0, ignore_index=True)
    ret_df = pd.pivot_table(data=ret_con_df, index="trade_date", columns="pair", values="diff_return")
    ret_df = ret_df[pair_ids].fillna(0)

    # --- raw ret
    if aligned_sig_df.shape != ret_df.shape:
        print(f"{dt.datetime.now()} [ERR] signal shape = {aligned_sig_df.shape}, return shape = {ret_df.shape}")
        raise ValueError

    pair_ret_df = aligned_sig_df * ret_df
    raw_ret: pd.Series = pair_ret_df.mean(axis=1)
    dlt_wgt_abs_sum: pd.Series = dlt_wgt_df.abs().sum(axis=1)
    cost: pd.Series = dlt_wgt_abs_sum * cost_rate
    net_ret: pd.Series = raw_ret - cost
    simu_df = pd.DataFrame({
        "signal": None,
        "rawRet": raw_ret,
        "dltWgt": dlt_wgt_abs_sum,
        "cost": cost,
        "netRet": net_ret,
        "cumNetRet": net_ret.cumsum()
    })

    lib_simu_writer = CLibSimu(simu_id=ml_model_id, lib_save_dir=simulations_dir).get_lib_writer(run_mode)
    lib_simu_writer.update(update_df=simu_df, using_index=True)
    lib_simu_writer.commit()
    lib_simu_writer.close()

    print(f"{dt.datetime.now()} [INF] simulation for {SFG(ml_model_id)}"
          f" from {SFG(bgn_date)} to {SFG(stp_date)} are calculated")
    return 0

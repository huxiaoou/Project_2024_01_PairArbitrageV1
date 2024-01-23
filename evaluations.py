import os
import datetime as dt
import pandas as pd
from itertools import product
from husfort.qutility import SFG
from husfort.qevaluation import CNAV
from husfort.qplot import CPlotLines
from simulations import CLibSimu


def cal_evaluations(simu_id: str, bgn_date: str, stp_date: str, simulations_dir: dir) -> dict:
    lib_simu_reader = CLibSimu(simu_id=simu_id, lib_save_dir=simulations_dir).get_lib_reader()
    net_ret_df = lib_simu_reader.read_by_conditions(conditions=[
        ("trade_date", ">=", bgn_date),
        ("trade_date", "<", stp_date),
    ], value_columns=["trade_date", "netRet"]).set_index("trade_date")
    nav = CNAV(input_srs=net_ret_df["netRet"], input_type="RET")
    nav.cal_all_indicators()
    d = nav.to_dict(save_type="eng")
    print(f"{dt.datetime.now()} [INF] {' ' + SFG(simu_id):.>48s} evaluated", end="\r")
    return d


def cal_evaluations_quick(instruments_pairs: list[tuple[str, str]], diff_ret_delays: list[int], factors: list[str],
                          evaluations_dir: str, **kwargs):
    eval_results = []
    for pair, delay, factor in product(instruments_pairs, diff_ret_delays, factors):
        pair_id = f"{pair[0]}_{pair[1]}"
        simu_id = f"{pair_id}.{factor}.T{delay}"
        d = cal_evaluations(simu_id=simu_id, **kwargs)
        d.update({
            "pair": pair_id,
            "factor": factor,
            "delay": f"T{delay}",
        })
        eval_results.append(d)
    eval_results_df = pd.DataFrame(eval_results)
    eval_results_file = "eval.quick.csv"
    eval_results_path = os.path.join(evaluations_dir, eval_results_file)
    eval_results_df.to_csv(eval_results_path, index=False, float_format="%.8f")
    return 0


def plot_factor_simu_quick(factor: str, delay: int, instruments_pairs: list[str],
                           bgn_date: str, stp_date: str, simulations_dir: str, plot_save_dir: str):
    nav_data = {}
    for pair in instruments_pairs:
        pair_id = f"{pair[0]}_{pair[1]}"
        simu_id = f"{pair_id}.{factor}.T{delay}"
        lib_simu_reader = CLibSimu(simu_id=simu_id, lib_save_dir=simulations_dir).get_lib_reader()
        net_ret_df = lib_simu_reader.read_by_conditions(conditions=[
            ("trade_date", ">=", bgn_date),
            ("trade_date", "<", stp_date),
        ], value_columns=["trade_date", "netRet"]).set_index("trade_date")
        nav_data[pair_id] = (net_ret_df["netRet"] + 1).cumprod()
    nav_df = pd.DataFrame(nav_data)
    artist = CPlotLines(
        line_width=1, fig_save_dir=plot_save_dir, fig_save_type="PNG",
        line_style=["-"] * 10 + ["-."] * 10,
        fig_name=f"simu_quick.{factor}.T{delay}", plot_df=nav_df,
    )
    artist.plot()
    return 0


def plot_factors_simu_quick(factors: list[str], diff_ret_delays: list[int], instruments_pairs: list[str], **kwargs):
    for factor, delay in product(factors, diff_ret_delays):
        plot_factor_simu_quick(factor, delay, instruments_pairs=instruments_pairs, **kwargs)
        print(f"{dt.datetime.now()} [INF] quick simulation nav curve for {SFG(f'{factor}-T{delay}'):>24s} plotted",
              end="\r")
    return 0

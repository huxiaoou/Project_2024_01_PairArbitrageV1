import argparse


def parse_project_args():
    parser_main = argparse.ArgumentParser(description="Entry point of this project")
    parsers_sub = parser_main.add_subparsers(
        title="sub argument:switch", dest="switch",
        description="use this argument to go to call different functions",
    )

    # diff return
    parser_sub = parsers_sub.add_parser(name="diff", help="Calculate difference return of pair instruments")
    parser_sub.add_argument("--mode", type=str, help="overwrite or append", choices=("o", "a"), required=True)
    parser_sub.add_argument("--bgn", type=str, help="begin date, format = [YYYYMMDD]", required=True)
    parser_sub.add_argument("--stp", type=str, help="stop  date, format = [YYYYMMDD]")

    # factor exposure
    parser_sub = parsers_sub.add_parser(name="exposure", help="Calculate factor exposure")
    parser_sub.add_argument("--factor", type=str, help="which factor to calculate", choices=(
        "lag", "ewm", "volatility", "tnr",
        "basisa", "ctp", "cvp", "csp",
        "rsbr", "rslr", "skew", "mtm", "mtms", "tsa", "tsld"
    ), required=True)
    parser_sub.add_argument("--mode", type=str, help="overwrite or append", choices=("o", "a"), required=True)
    parser_sub.add_argument("--bgn", type=str, help="begin date, format = [YYYYMMDD]", required=True)
    parser_sub.add_argument("--stp", type=str, help="stop  date, format = [YYYYMMDD]")

    # regroup: diff ret and factor exposure
    parser_sub = parsers_sub.add_parser(name="regroups", help="Regroup factor exposure and diff return")
    parser_sub.add_argument("--mode", type=str, help="overwrite or append", choices=("o", "a"), required=True)
    parser_sub.add_argument("--bgn", type=str, help="begin date, format = [YYYYMMDD]", required=True)
    parser_sub.add_argument("--stp", type=str, help="stop  date, format = [YYYYMMDD]")

    return parser_main.parse_args()


if __name__ == "__main__":
    args = parse_project_args()
    if args.switch == "diff":
        from project_config import instruments_pairs
        from project_setup import diff_returns_dir, major_return_save_dir
        from returns_diff import cal_diff_returns_pairs

        cal_diff_returns_pairs(
            instruments_pairs=instruments_pairs,
            major_return_save_dir=major_return_save_dir,
            run_mode=args.mode, bgn_date=args.bgn, stp_date=args.stp,
            diff_returns_dir=diff_returns_dir
        )
    elif args.switch == "exposure":
        from husfort.qcalendar import CCalendar
        from project_setup import (factors_exposure_dir, diff_returns_dir,
                                   instru_factor_exposure_dir, calendar_path)
        from project_config import instruments_pairs, config_factor

        calendar = CCalendar(calendar_path)
        factor_args = config_factor[args.factor]["args"]
        if args.factor == "lag":
            from exposures import CFactorExposureLagRet

            for lag in factor_args:
                factor = CFactorExposureLagRet(
                    lag=lag, diff_returns_dir=diff_returns_dir,
                    factors_exposure_dir=factors_exposure_dir,
                    instruments_pairs=instruments_pairs,
                )
                factor.main(run_mode=args.mode, bgn_date=args.bgn, stp_date=args.stp, calendar=calendar)
        elif args.factor == "ewm":
            from exposures import CFactorExposureEWM

            fix_base_date = config_factor[args.factor]["fix_base_date"]
            for (fast, slow) in factor_args:
                factor = CFactorExposureEWM(
                    fast=fast, slow=slow, diff_returns_dir=diff_returns_dir,
                    fix_base_date=fix_base_date,
                    factors_exposure_dir=factors_exposure_dir,
                    instruments_pairs=instruments_pairs,
                )
                factor.main(run_mode=args.mode, bgn_date=args.bgn, stp_date=args.stp, calendar=calendar)
        elif args.factor == "volatility":
            from exposures import CFactorExposureVolatility

            for (win, k) in factor_args:
                factor = CFactorExposureVolatility(
                    win=win, k=k, diff_returns_dir=diff_returns_dir,
                    factors_exposure_dir=factors_exposure_dir,
                    instruments_pairs=instruments_pairs,
                )
                factor.main(run_mode=args.mode, bgn_date=args.bgn, stp_date=args.stp, calendar=calendar)
        elif args.factor == "tnr":
            from exposures import CFactorExposureTNR

            for (win, k) in factor_args:
                factor = CFactorExposureTNR(
                    win=win, k=k, diff_returns_dir=diff_returns_dir,
                    factors_exposure_dir=factors_exposure_dir,
                    instruments_pairs=instruments_pairs,
                )
                factor.main(run_mode=args.mode, bgn_date=args.bgn, stp_date=args.stp, calendar=calendar)
        elif args.factor == "basisa":
            from exposures import CFactorExposureBasisa

            for win in factor_args:
                factor = CFactorExposureBasisa(
                    win=win, instru_factor_exposure_dir=instru_factor_exposure_dir,
                    factors_exposure_dir=factors_exposure_dir,
                    instruments_pairs=instruments_pairs,
                )
                factor.main(run_mode=args.mode, bgn_date=args.bgn, stp_date=args.stp, calendar=calendar)

        elif args.factor == "ctp":
            from exposures import CFactorExposureCTP

            for win in factor_args:
                factor = CFactorExposureCTP(
                    win=win, instru_factor_exposure_dir=instru_factor_exposure_dir,
                    factors_exposure_dir=factors_exposure_dir,
                    instruments_pairs=instruments_pairs,
                )
                factor.main(run_mode=args.mode, bgn_date=args.bgn, stp_date=args.stp, calendar=calendar)
        elif args.factor == "cvp":
            from exposures import CFactorExposureCVP

            for win in factor_args:
                factor = CFactorExposureCVP(
                    win=win, instru_factor_exposure_dir=instru_factor_exposure_dir,
                    factors_exposure_dir=factors_exposure_dir,
                    instruments_pairs=instruments_pairs,
                )
                factor.main(run_mode=args.mode, bgn_date=args.bgn, stp_date=args.stp, calendar=calendar)
        elif args.factor == "csp":
            from exposures import CFactorExposureCSP

            for win in factor_args:
                factor = CFactorExposureCSP(
                    win=win, instru_factor_exposure_dir=instru_factor_exposure_dir,
                    factors_exposure_dir=factors_exposure_dir,
                    instruments_pairs=instruments_pairs,
                )
                factor.main(run_mode=args.mode, bgn_date=args.bgn, stp_date=args.stp, calendar=calendar)
        elif args.factor == "rsbr":
            from exposures import CFactorExposureRSBR

            for win in factor_args:
                factor = CFactorExposureRSBR(
                    win=win, instru_factor_exposure_dir=instru_factor_exposure_dir,
                    factors_exposure_dir=factors_exposure_dir,
                    instruments_pairs=instruments_pairs,
                )
                factor.main(run_mode=args.mode, bgn_date=args.bgn, stp_date=args.stp, calendar=calendar)
        elif args.factor == "rslr":
            from exposures import CFactorExposureRSLR

            for win in factor_args:
                factor = CFactorExposureRSLR(
                    win=win, instru_factor_exposure_dir=instru_factor_exposure_dir,
                    factors_exposure_dir=factors_exposure_dir,
                    instruments_pairs=instruments_pairs,
                )
                factor.main(run_mode=args.mode, bgn_date=args.bgn, stp_date=args.stp, calendar=calendar)
        elif args.factor == "skew":
            from exposures import CFactorExposureSKEW

            for win in factor_args:
                factor = CFactorExposureSKEW(
                    win=win, instru_factor_exposure_dir=instru_factor_exposure_dir,
                    factors_exposure_dir=factors_exposure_dir,
                    instruments_pairs=instruments_pairs,
                )
                factor.main(run_mode=args.mode, bgn_date=args.bgn, stp_date=args.stp, calendar=calendar)
        elif args.factor == "mtm":
            from exposures import CFactorExposureMTM

            for win in factor_args:
                factor = CFactorExposureMTM(
                    win=win, instru_factor_exposure_dir=instru_factor_exposure_dir,
                    factors_exposure_dir=factors_exposure_dir,
                    instruments_pairs=instruments_pairs,
                )
                factor.main(run_mode=args.mode, bgn_date=args.bgn, stp_date=args.stp, calendar=calendar)
        elif args.factor == "mtms":
            from exposures import CFactorExposureMTMS

            for win in factor_args:
                factor = CFactorExposureMTMS(
                    win=win, instru_factor_exposure_dir=instru_factor_exposure_dir,
                    factors_exposure_dir=factors_exposure_dir,
                    instruments_pairs=instruments_pairs,
                )
                factor.main(run_mode=args.mode, bgn_date=args.bgn, stp_date=args.stp, calendar=calendar)
        elif args.factor == "tsa":
            from exposures import CFactorExposureTSA

            for win in factor_args:
                factor = CFactorExposureTSA(
                    win=win, instru_factor_exposure_dir=instru_factor_exposure_dir,
                    factors_exposure_dir=factors_exposure_dir,
                    instruments_pairs=instruments_pairs,
                )
                factor.main(run_mode=args.mode, bgn_date=args.bgn, stp_date=args.stp, calendar=calendar)
        elif args.factor == "tsld":
            from exposures import CFactorExposureTSLD

            for win in factor_args:
                factor = CFactorExposureTSLD(
                    win=win, instru_factor_exposure_dir=instru_factor_exposure_dir,
                    factors_exposure_dir=factors_exposure_dir,
                    instruments_pairs=instruments_pairs,
                )
                factor.main(run_mode=args.mode, bgn_date=args.bgn, stp_date=args.stp, calendar=calendar)
        else:
            print(f"... [ERR] factor = {args.factor}")
            raise ValueError
    elif args.switch == "regroups":
        from husfort.qcalendar import CCalendar
        from project_setup import (factors_exposure_dir, diff_returns_dir, regroups_dir, calendar_path)
        from project_config import instruments_pairs, factors, diff_ret_delays
        from regroups import cal_regroups_pairs

        calendar = CCalendar(calendar_path)
        cal_regroups_pairs(
            instruments_pairs=instruments_pairs, diff_ret_delays=diff_ret_delays,
            factors=factors,
            run_mode=args.mode, bgn_date=args.bgn, stp_date=args.stp,
            diff_returns_dir=diff_returns_dir,
            factors_exposure_dir=factors_exposure_dir,
            regroups_dir=regroups_dir,
            calendar=calendar,
        )

    else:
        raise ValueError

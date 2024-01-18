import argparse


def parse_project_args():
    parser_main = argparse.ArgumentParser(description="Entry point of this project")
    parsers_sub = parser_main.add_subparsers(
        title="sub argument:switch", dest="switch",
        description="use this argument to go to call different functions",
    )

    # diff return
    parser_sub = parsers_sub.add_parser(name="diff", help="Calculate difference return of pair instruments")
    parser_sub.add_argument("--mode", type=str, help="overwrite or append", choices=("o", "a"))
    parser_sub.add_argument("--bgn", type=str, help="begin date, format = [YYYYMMDD]")
    parser_sub.add_argument("--stp", type=str, help="stop  date, format = [YYYYMMDD]")

    # factor exposure
    parser_sub = parsers_sub.add_parser(name="exposure", help="Calculate factor exposure")
    parser_sub.add_argument("--factor", type=str, help="which factor to calculate",
                            choices=("lag", "ewm", "basis", "stock", "ts", "ctp"))
    parser_sub.add_argument("--mode", type=str, help="overwrite or append", choices=("o", "a"))
    parser_sub.add_argument("--bgn", type=str, help="begin date, format = [YYYYMMDD]")
    parser_sub.add_argument("--stp", type=str, help="stop  date, format = [YYYYMMDD]")

    return parser_main.parse_args()


if __name__ == "__main__":
    args = parse_project_args()
    if args.switch == "diff":
        from project_config import instruments_pairs
        from project_setup import diff_returns_dir, major_return_save_dir
        from returns_diff import cal_diff_returns_groups

        cal_diff_returns_groups(
            instruments_pairs=instruments_pairs,
            major_return_save_dir=major_return_save_dir,
            run_mode=args.mode, bgn_date=args.bgn, stp_date=args.stp,
            diff_returns_dir=diff_returns_dir
        )
    elif args.switch == "exposure":
        from husfort.qcalendar import CCalendar
        from project_setup import factors_exposure_dir, diff_returns_dir, calendar_path
        from project_config import instruments_pairs

        calendar = CCalendar(calendar_path)
        if args.factor == "lag":
            from exposures import CFactorExposureLagRet

            factor = CFactorExposureLagRet(
                lag=1, diff_returns_dir=diff_returns_dir,
                factors_exposure_dir=factors_exposure_dir,
                instruments_pairs=instruments_pairs,
            )
            factor.main(run_mode=args.mode, bgn_date=args.bgn, stp_date=args.stp, calendar=calendar)
        elif args.factor == "ewm":
            pass
        elif args.factor == "basis":
            pass
        elif args.factor == "stock":
            pass
        elif args.factor == "ts":
            pass
        elif args.factor == "ctp":
            pass
        else:
            print(f"... [ERR] factor = {args.factor}")
            raise ValueError
    else:
        raise ValueError

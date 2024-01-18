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

    return parser_main.parse_args()


if __name__ == "__main__":
    args = parse_project_args()
    if args.switch == "diff":
        from project_config import instruments_group
        from project_setup import diff_returns_dir, major_return_save_dir
        from returns_diff import cal_diff_returns_groups

        cal_diff_returns_groups(
            instruments_group=instruments_group,
            major_return_save_dir=major_return_save_dir,
            run_mode=args.mode, bgn_date=args.bgn, stp_date=args.stp,
            diff_returns_dir=diff_returns_dir
        )
    elif args.switch == "exposure":
        pass
    else:
        raise ValueError

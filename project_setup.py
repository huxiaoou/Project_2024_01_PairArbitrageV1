from husfort.qutility import get_twin_dir
import os

major_return_save_dir = r"E:\Deploy\Data\Futures\by_instrument"
calendar_path = r"E:\Deploy\Data\Calendar\cne_calendar.csv"
instru_factor_exposure_dir = r"E:\Deploy\Data\ForProjects\cta3\factors_exposure\raw"

save_root_dir = r"E:\ProjectsData"
project_save_dir = get_twin_dir(save_root_dir, src=".")
diff_returns_dir = os.path.join(project_save_dir, "diff_returns")
factors_exposure_dir = os.path.join(project_save_dir, "factors_exposure")

if __name__ == "__main__":
    from husfort.qutility import check_and_mkdir

    check_and_mkdir(project_save_dir)
    check_and_mkdir(diff_returns_dir)
    check_and_mkdir(factors_exposure_dir)

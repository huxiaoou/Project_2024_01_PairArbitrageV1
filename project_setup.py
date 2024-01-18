from husfort.qutility import get_twin_dir
import os

major_return_save_dir = r"E:\Deploy\Data\Futures\by_instrument"

save_root_dir = r"E:\ProjectsData"
project_save_dir = get_twin_dir(save_root_dir, src=".")
diff_returns_dir = os.path.join(project_save_dir, "diff_returns")

if __name__ == "__main__":
    from husfort.qutility import check_and_mkdir

    check_and_mkdir(project_save_dir)
    check_and_mkdir(diff_returns_dir)

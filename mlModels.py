import os
import datetime as dt
import numpy as np
import scipy.stats as sps
import skops.io as sio
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from husfort.qcalendar import CCalendar
from husfort.qutility import check_and_mkdir, SFG, SFY
from husfort.qsqlite import CQuickSqliteLib, CLib1Tab1, CTable
from regroups import CLibRegroups


class CLibPredictions(CQuickSqliteLib):
    def __init__(self, model_id: str, lib_save_dir: str):
        self.model_id = model_id
        lib_name = f"{self.model_id}.db"
        super().__init__(lib_name, lib_save_dir)

    def get_lib_struct(self) -> CLib1Tab1:
        return CLib1Tab1(
            lib_name=self.lib_name,
            table=CTable(
                {
                    "table_name": "predictions",
                    "primary_keys": {"trade_date": "TEXT", "pair": "TEXT"},
                    "value_columns": {"value": "REAL"},
                }
            )
        )


class CMLModel(object):
    def __init__(self, model_id: str,
                 pairs: list[tuple[str, str]], delay: int, factors: list[str], y_lbl: str,
                 ml_models_dir: str, predictions_dir: str,
                 trn_win: int, days_per_month: int = 20, normalize_alpha: float = 0.05):
        self.model_id = model_id
        self.pairs = pairs
        self.delay = delay
        self.factors, self.y_lbl = factors, y_lbl
        self.ml_models_dir = ml_models_dir
        self.predictions_dir = predictions_dir
        self.train_win = trn_win
        self.days_per_month = days_per_month
        if (normalize_alpha > 0.5) or (normalize_alpha <= 0):
            print(f"{dt.datetime.now()} [ERR] alpha = {normalize_alpha}")
            raise ValueError
        self.normalize_alpha = normalize_alpha
        self.core_data: pd.DataFrame = pd.DataFrame()
        self.model_obj = None

    @staticmethod
    def get_iter_dates(bgn_date: str, stp_date: str,
                       calendar: CCalendar, shifts: list[int] = None) -> tuple[list[str], tuple]:
        iter_dates = calendar.get_iter_list(bgn_date, stp_date)
        shift_dates = tuple(calendar.shift_iter_dates(iter_dates, s) for s in shifts)
        return iter_dates, shift_dates

    @staticmethod
    def is_model_update_date(this_date: str, next_date_1: str, next_date_2: str) -> bool:
        return (this_date[0:6] == next_date_1[0:6]) and (this_date[0:6] != next_date_2[0:6])

    @staticmethod
    def is_last_month_date(this_date: str, next_date: str) -> bool:
        return this_date[0:6] != next_date[0:6]

    @staticmethod
    def is_last_iter_date(trade_date: str, iter_dates: list[str]) -> bool:
        return trade_date == iter_dates[-1]

    @property
    def train_win_size(self) -> int:
        return self.train_win * self.days_per_month

    def _get_train_df(self, end_date: str) -> pd.DataFrame:
        return self.core_data.truncate(after=end_date).tail(n=self.train_win_size * len(self.pairs))

    def _get_predict_df(self, bgn_date: str, end_date: str) -> pd.DataFrame:
        return self.core_data.truncate(before=bgn_date, after=end_date)[self.factors]

    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        m = df.median()
        df_fill_nan = df.fillna(m)
        df_none_inf = df.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
        qu = np.quantile(df_none_inf, 1 - self.normalize_alpha / 2, axis=0)
        ql = np.quantile(df_none_inf, self.normalize_alpha / 2, axis=0)
        rng = (qu - ql) / 2
        sd = rng / sps.norm.ppf(1 - self.normalize_alpha / 2)
        ub = m + rng
        ud = m - rng
        df_fill_nan = df_fill_nan[df_fill_nan < ub].fillna(ub)
        df_fill_nan = df_fill_nan[df_fill_nan > ud].fillna(ud)
        df_norm = (df_fill_nan - m) / sd
        return df_norm

    def _transform_y(self, y_srs: pd.Series) -> pd.Series:
        pass

    def _fit(self, x: np.ndarray, y: np.ndarray):
        pass

    def _transform(self, train_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        x = self._normalize(train_df[self.factors])
        y = self._transform_y(train_df[self.y_lbl])
        return x.values, y.values

    def _apply_model(self, predict_df: pd.DataFrame) -> pd.DataFrame:
        norm_df = self._normalize(predict_df[self.factors])
        x = norm_df.values
        p = self.model_obj.predict(X=x)
        pred = pd.DataFrame(data={"pred": p}, index=predict_df.index)
        return pred

    def _save_model(self, month_id: str):
        model_file = f"{month_id}.{self.model_id}.skops"
        month_dir = os.path.join(self.ml_models_dir, month_id)
        model_path = os.path.join(month_dir, model_file)
        check_and_mkdir(month_dir)
        sio.dump(self.model_obj, model_path)
        return 0

    def _load_model(self, month_id: str) -> bool:
        model_file = f"{month_id}.{self.model_id}.skops"
        month_dir = os.path.join(self.ml_models_dir, month_id)
        model_path = os.path.join(month_dir, model_file)
        if os.path.exists(model_path):
            self.model_obj = sio.load(model_path, trusted=True)
            return True
        else:
            print(f"{dt.datetime.now()} [WRN] failed to load model for {SFY(self.model_id)} at {SFY(month_id)}")
            return False

    def load_data(self, bgn_date: str, stp_date: str, regroups_dir: str):
        dfs = []
        for pair in self.pairs:
            lib_reader = CLibRegroups(pair, self.delay, regroups_dir).get_lib_reader()
            df = lib_reader.read_by_conditions(conditions=[
                ("trade_date", ">=", bgn_date),
                ("trade_date", "<", stp_date),
            ], value_columns=["trade_date", "factor", "value"])
            df["pair"] = "_".join(pair)
            dfs.append(df)
        regroups_df = pd.concat(dfs, axis=0, ignore_index=True)
        self.core_data = pd.pivot_table(
            data=regroups_df, index=["trade_date", "pair"], columns="factor", values="value")
        self.core_data = self.core_data[self.factors + [self.y_lbl]]
        return 0

    def train(self, bgn_date: str, stp_date: str, calendar: CCalendar):
        iter_dates, (next_dates_1, next_dates_2) = self.get_iter_dates(bgn_date, stp_date, calendar, shifts=[1, 2])
        for (this_date, next_date_1, next_date_2) in zip(iter_dates, next_dates_1, next_dates_2):
            if self.is_model_update_date(this_date, next_date_1, next_date_2):
                train_df = self._get_train_df(end_date=this_date)
                x, y = self._transform(train_df)
                self._fit(x, y)
                self._save_model(month_id=this_date[0:6])
                print(f"{dt.datetime.now()} [INF] {SFG(self.model_id)} for {SFG(this_date[0:6])} trained")
        return 0

    def predict(self, bgn_date, stp_date, calendar: CCalendar) -> pd.DataFrame:
        iter_dates, (next_dates,) = self.get_iter_dates(bgn_date, stp_date, calendar, shifts=[1])
        dfs: list[pd.DataFrame] = []
        for (this_date, next_date) in zip(iter_dates, next_dates):
            if self.is_last_month_date(this_date, next_date) or self.is_last_iter_date(this_date, iter_dates):
                this_month = this_date[0:6]
                prev_month = calendar.get_next_month(this_month, s=-1)
                if self._load_model(month_id=prev_month):
                    predict_df = self._get_predict_df(bgn_date=this_month + "01", end_date=this_date)
                    pred_df = self._apply_model(predict_df)
                    dfs.append(pred_df)
                    print(f"{dt.datetime.now()} [INF] {SFG(self.model_id)} for {SFG(this_date[0:6])} predicted")
        predictions = pd.concat(dfs, axis=0, ignore_index=False).reset_index()
        return predictions

    def save_prediction(self, df: pd.DataFrame, run_mode: str):
        lib_pred_writer = CLibPredictions(self.model_id, self.predictions_dir).get_lib_writer(run_mode)
        lib_pred_writer.update(update_df=df, using_index=False)
        lib_pred_writer.commit()
        lib_pred_writer.close()
        return 0

    def main(self, run_mode: str, bgn_date: str, stp_date: str, calendar: CCalendar,
             regroups_dir: str):
        self.load_data(bgn_date, stp_date, regroups_dir)
        self.train(bgn_date, stp_date, calendar)
        predictions = self.predict(bgn_date, stp_date, calendar)
        self.save_prediction(predictions, run_mode)
        return 0


class CMLModelLogistic(CMLModel):
    def __init__(self, cs: int = 10, cv: int = None, max_iter: int = 5000,
                 fit_intercept: bool = True, penalty: str = 'l2', **kwargs):
        self.cs = cs
        self.cv = cv
        self.max_iter = max_iter
        self.fit_intercept = fit_intercept
        self.penalty = penalty
        super().__init__(**kwargs)

    def _transform_y(self, y_srs: pd.Series) -> pd.Series:
        return y_srs.map(lambda z: 1 if z >= 0 else 0)

    def _fit(self, x: np.ndarray, y: np.ndarray):
        obj_cv = LogisticRegressionCV(
            cv=self.cv, Cs=self.cs, max_iter=self.max_iter,
            fit_intercept=self.fit_intercept, penalty=self.penalty,
            random_state=0)
        self.model_obj = obj_cv.fit(X=x, y=y)
        return 0

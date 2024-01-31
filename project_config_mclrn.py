from project_config import instruments_pairs, factors
from mclrn import CMLModel, CMLModelLogistic, CMLMlp, CMLLr, CMLSvc, CMLDt, CMLKn

models_mclrn: list[CMLModel] = [
    CMLLr(
        model_id="M00", desc="Linear",
        pairs=instruments_pairs, delay=2, factors=factors, y_lbl="diff_return",
        sig_method="continuous", trn_win=3
    ),
    CMLModelLogistic(
        model_id="M01", desc="Logistic",
        pairs=instruments_pairs, delay=2, factors=factors, y_lbl="diff_return",
        sig_method="binary", trn_win=3,
    ),
    CMLMlp(
        model_id="M02", desc="MultiLayerPerception",
        pairs=instruments_pairs, delay=2, factors=factors, y_lbl="diff_return",
        sig_method="binary", trn_win=3
    ),
    CMLSvc(
        model_id="M03", desc="SupportVectorMachine",
        pairs=instruments_pairs, delay=2, factors=factors, y_lbl="diff_return",
        sig_method="binary", trn_win=3
    ),
    CMLDt(
        model_id="M04", desc="DecisionTree",
        pairs=instruments_pairs, delay=2, factors=factors, y_lbl="diff_return",
        sig_method="binary", trn_win=3
    ),
    CMLKn(
        model_id="M05", desc="KNeighbor",
        pairs=instruments_pairs, delay=2, factors=factors, y_lbl="diff_return",
        sig_method="binary", trn_win=3
    ),
]
headers_mclrn = [(m.model_id, m.desc) for m in models_mclrn]

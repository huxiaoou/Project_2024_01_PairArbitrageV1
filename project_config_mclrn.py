from project_config import instruments_pairs, factors
from mclrn import CMLModel, CMLLogistic, CMLMlp, CMLLr, CMLSvc, CMLDt, CMLKn, CMLAdaboost, CMLGb

models_mclrn: list[CMLModel] = [
    CMLLr(
        model_id="M00", desc="Linear",
        pairs=instruments_pairs, delay=2, factors=factors, y_lbl="diff_return",
        sig_method="continuous", trn_win=3
    ),
    CMLLogistic(
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
    CMLAdaboost(
        model_id="M06", desc="AdaBoost",
        pairs=instruments_pairs, delay=2, factors=factors, y_lbl="diff_return",
        sig_method="binary", trn_win=3
    ),
    CMLGb(
        model_id="M07", desc="GradientBoosting",
        pairs=instruments_pairs, delay=2, factors=factors, y_lbl="diff_return",
        sig_method="binary", trn_win=3
    ),
]

selected_models = [
    {
        "factors": ["CSP120", "CTP120", "CVP120"],
        "pairs": [
            ("C.DCE", "CS.DCE"),
            ("AL.SHF", "ZN.SHF"), ("CU.SHF", "AL.SHF"),
            ("HC.SHF", "RB.SHF"),
            ("MA.CZC", "V.DCE"), ("L.DCE", "PP.DCE"),
            ("OI.CZC", "P.DCE")
        ],
        "subId": "S0",
    },
    {
        "factors": ["BASISA060", "BASISA120", "TSA120", "TSA180", "TSLD240"],
        "pairs": [
            ("I.DCE", "RB.SHF"),
            ("M.DCE", "Y.DCE"), ("OI.CZC", "P.DCE"), ("P.DCE", "Y.DCE"), ("A.DCE", "M.DCE"),
        ],
        "subId": "S1",
    },
    {
        "factors": ["SKEW010"],
        "pairs": [
            ("AG.SHF", "AU.SHF"),
            ("A.DCE", "Y.DCE"), ("OI.CZC", "P.DCE"), ("P.DCE", "Y.DCE"),
        ],
        "subId": "S2",
    },
    {
        "factors": ["MTM", "LAG01", "F60S10"],
        "pairs": [
            ("AL.SHF", "ZN.SHF"), ("CU.SHF", "AL.SHF"),
            ("MA.CZC", "V.DCE"),
        ],
        "subId": "S3",
    },
]

for d in selected_models:
    _factors, _pairs, _subId = d["factors"], d["pairs"], d["subId"]
    models_mclrn += [
        CMLLr(
            model_id="M00" + _subId, desc="Linear",
            pairs=_pairs, delay=2, factors=_factors, y_lbl="diff_return",
            sig_method="continuous", trn_win=3
        ),
        CMLLogistic(
            model_id="M01" + _subId, desc="Logistic",
            pairs=_pairs, delay=2, factors=_factors, y_lbl="diff_return",
            sig_method="binary", trn_win=3,
        ),
        CMLMlp(
            model_id="M02" + _subId, desc="MultiLayerPerception",
            pairs=_pairs, delay=2, factors=_factors, y_lbl="diff_return",
            sig_method="binary", trn_win=3
        ),
        CMLSvc(
            model_id="M03" + _subId, desc="SupportVectorMachine",
            pairs=_pairs, delay=2, factors=_factors, y_lbl="diff_return",
            sig_method="binary", trn_win=3
        ),
        CMLDt(
            model_id="M04" + _subId, desc="DecisionTree",
            pairs=_pairs, delay=2, factors=_factors, y_lbl="diff_return",
            sig_method="binary", trn_win=3
        ),
        CMLKn(
            model_id="M05" + _subId, desc="KNeighbor",
            pairs=_pairs, delay=2, factors=_factors, y_lbl="diff_return",
            sig_method="binary", trn_win=3
        ),
        CMLAdaboost(
            model_id="M06" + _subId, desc="AdaBoost",
            pairs=_pairs, delay=2, factors=_factors, y_lbl="diff_return",
            sig_method="binary", trn_win=3
        ),
        CMLGb(
            model_id="M07" + _subId, desc="GradientBoosting",
            pairs=_pairs, delay=2, factors=_factors, y_lbl="diff_return",
            sig_method="binary", trn_win=3
        ),
    ]

headers_mclrn = [(m.model_id, m.desc) for m in models_mclrn]

if __name__ == "__main__":
    print("-" * 32)
    print(headers_mclrn)
    print("-" * 32)
    print(f"size of models:{len(headers_mclrn)}")

instruments_pairs = [
    ("A.DCE", "M.DCE"),
    ("A.DCE", "Y.DCE"),
    ("M.DCE", "Y.DCE"),
    ("M.DCE", "RM.CZC"),

    ("OI.CZC", "P.DCE"),
    ("OI.CZC", "Y.DCE"),
    ("P.DCE", "Y.DCE"),

    ("AG.SHF", "AU.SHF"),

    ("AL.SHF", "ZN.SHF"),
    ("CU.SHF", "ZN.SHF"),
    ("CU.SHF", "AL.SHF"),

    ("HC.SHF", "RB.SHF"),
    ("I.DCE", "RB.SHF"),
    ("JM.DCE", "J.DCE"),
    ("JM.DCE", "I.DCE"),

    ("BU.SHF", "TA.CZC"),
    ("L.DCE", "PP.DCE"),
    ("C.DCE", "CS.DCE"),
    ("MA.CZC", "V.DCE"),

    # ("BU.SHF", "FU.SHF"),  # "FU.SHF" since 20180716
]
ga, gb = zip(*instruments_pairs)
ga, gb = list(ga), list(gb)

config_factor = {
    "lag": {
        "args": (1, 2, 3, 4, 5)
    },
    "basisa": {
        "args": (60, 120,)
    },
    "ctp": {
        "args": (120,)
    },
    "cvp": {
        "args": (120,)
    },
    "csp": {
        "args": (120,)
    },
    "rsbr": {
        "args": (10,)
    },
    "rslr": {
        "args": (20,)
    },
    "skew": {
        "args": (10,)
    },
    "mtm": {
        "args": (1,)
    },
    "mtms": {
        "args": (240,)
    },
    "tsa": {
        "args": (120, 180)
    },
    "tsld": {
        "args": (240,)
    },
}

instruments_group = [
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

    # ("BU.SHF", "FU.SHF"),  # "FU.SHF" since 20180716
    ("BU.SHF", "TA.CZC"),
    ("L.DCE", "PP.DCE"),
    ("C.DCE", "CS.DCE"),
    ("MA.CZC", "V.DCE"),
]
ga, gb = zip(*instruments_group)
ga, gb = list(ga), list(gb)

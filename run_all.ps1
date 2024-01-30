$bgn_date_diff = "20160104"
$bgn_date_fact = $bgn_date_diff
$bgn_date_regp = "20160401"
$bgn_date_mclrn = "20160701"
$bgn_date_ictt = $bgn_date_regp
$bgn_date_simu = $bgn_date_regp
$stp_date = "20240122"

# diff return
python main.py diff     --mode o --bgn $bgn_date_diff --stp $stp_date

# factor exposure
python main.py exposure --mode o --bgn $bgn_date_fact --stp $stp_date --factor lag
python main.py exposure --mode o --bgn $bgn_date_fact --stp $stp_date --factor ewm
python main.py exposure --mode o --bgn $bgn_date_fact --stp $stp_date --factor volatility
python main.py exposure --mode o --bgn $bgn_date_fact --stp $stp_date --factor tnr
python main.py exposure --mode o --bgn $bgn_date_fact --stp $stp_date --factor basisa
python main.py exposure --mode o --bgn $bgn_date_fact --stp $stp_date --factor ctp
python main.py exposure --mode o --bgn $bgn_date_fact --stp $stp_date --factor cvp
python main.py exposure --mode o --bgn $bgn_date_fact --stp $stp_date --factor csp
python main.py exposure --mode o --bgn $bgn_date_fact --stp $stp_date --factor rsbr
python main.py exposure --mode o --bgn $bgn_date_fact --stp $stp_date --factor rslr
python main.py exposure --mode o --bgn $bgn_date_fact --stp $stp_date --factor skew
python main.py exposure --mode o --bgn $bgn_date_fact --stp $stp_date --factor mtm
python main.py exposure --mode o --bgn $bgn_date_fact --stp $stp_date --factor mtms
python main.py exposure --mode o --bgn $bgn_date_fact --stp $stp_date --factor tsa
python main.py exposure --mode o --bgn $bgn_date_fact --stp $stp_date --factor tsld

# regroup
python main.py regroups --mode o --bgn $bgn_date_regp --stp $stp_date

# ic-tests
python main.py ic-tests          --bgn $bgn_date_ictt --stp $stp_date

# quick simulations
python main.py quick-simu --mode o --bgn $bgn_date_simu --stp $stp_date

# machine learning
python main.py mclrn      --mode o --bgn $bgn_date_mclrn --stp $stp_date
python main.py simu-mclrn --mode o --bgn $bgn_date_mclrn --stp $stp_date
python main.py eval-mclrn          --bgn $bgn_date_mclrn --stp $stp_date

$bgn_date_diff = "20160104"
$bgn_date_fact = $bgn_date_diff
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

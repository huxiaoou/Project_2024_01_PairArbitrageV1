$bgn_date_diff = "20160104"
$bgn_date_fact = $bgn_date_diff
$stp_date = "20240119"
python main.py diff     --mode o --bgn $bgn_date_diff --stp $stp_date
python main.py exposure --mode o --bgn $bgn_date_fact --stp $stp_date --factor lag

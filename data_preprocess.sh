#unzip ./zp_raw_data.zip
#save *pro*
sed -i "s/\-NONE\- \*pro\*/\-NULL\- *pro*/g"  `grep "\-NONE\- \*pro\*" -rl ./zp_data`
#5 other types changed to [MASK]
 sed -i "s/\*PRO\*/\[MASK\]/g"  `grep \*PRO\* -rl ./zp_data`
 sed -i "s/\*OP\*/\[MASK\]/g"  `grep \*OP\* -rl ./zp_data`
 sed -i "s/\*T\*/\[MASK\]/g"  `grep \*T\* -rl ./zp_data`
 sed -i "s/\*RNR\*/\[MASK\]/g"  `grep \*RNR\* -rl ./zp_data`
 sed -i "s/\s\*\-/ \[MASK\]-/g"  `grep \s\*\- -rl ./zp_data`
 sed -i "s/\s\*)/ \[MASK\])/g"  `grep \s\* -rl ./zp_data`

 #del number
 sed -i "s/\*pro\*\-./*pro*/g"  `grep "\*pro\*-" -rl ./zp_data`
 sed -i "s/\[MASK\]\-./[MASK]/g"  `grep "\[MASK\]-" -rl ./zp_data`

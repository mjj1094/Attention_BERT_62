pretrained model + preprocessed data
# 1. 数据处理->dataM
将*pro*之外的5种零元素都改为[MASK]，并且6种类型全部去掉‘-d’(即后面的数字，如：-1)
 ## 留着*pro*
 sed -i "s/\-NONE\- \*pro\*/\-NULL\- *pro*/g"  `grep "\-NONE\- \*pro\*" -rl ./data/zp_data`
 
 ## 5种零元素都改为[MASK]
 sed -i "s/\*PRO\*/\[MASK\]/g"  `grep \*PRO\* -rl ./data/zp_data`
 sed -i "s/\*OP\*/\[MASK\]/g"  `grep \*OP\* -rl ./data/zp_data`
 sed -i "s/\*T\*/\[MASK\]/g"  `grep \*T\* -rl ./data/zp_data`
 sed -i "s/\*RNR\*/\[MASK\]/g"  `grep \*RNR\* -rl ./data/zp_data`
 sed -i "s/\s\*\-/ \[MASK\]-/g"  `grep \s\*\- -rl ./data/zp_data`
 sed -i "s/\s\*)/ \[MASK\])/g"  `grep \s\* -rl ./data/zp_data`
 ## 去掉数字
 sed -i "s/\*pro\*\-./*pro*/g"  `grep "\*pro\*-" -rl ./data/zp_data`
 sed -i "s/\[MASK\]\-./[MASK]/g"  `grep "\[MASK\]-" -rl ./data/zp_data`
 
 去掉5种零元素，用黄亮的代码
 https://github.com/lianghuang3/lineardpparser/blob/master/code/tree.py
 
 # 2. 模型
 ## 1）ZP
 [MASK]
 ## 2）NP
不使用RNN了，直接利用BERT的输出，原因有二：1、BERT中有position信息，不存在无序问题；2、而且存在上下文信息
运行:
python main.py --do_train >./results/result_zp_mask_np_pretrained_1214 2> logs/log_zp_mask_np_pretrained_1214
结果：
dev: 0.4383157894736842
test: 0.5539988324576766（差两个点）
Attention result：
dev: 0.5346921075455334
test: 0.5732632807939287
# 3. 存在问题
Dev和test的结果相差12个点，跟目标函数有关？算不算过拟合？

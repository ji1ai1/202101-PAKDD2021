import datetime
import lightgbm
import multiprocessing
import numpy
import pandas
import pickle

目録 =  "."
内核欄 = (0, 1, 2, 7, 8, 9, 10, 11, 12, 13, 14, 18, 19, 20, 21, 22, 23)

處理緒數 = 8
片長 = 60
特征秒數 = 1920
樣本秒數 = 960

訓練起始秒序 = -188 * 86400
訓練中間秒序 = -128 * 86400
訓練終止秒序 = -68 * 86400
第二訓練起始秒序 = -31 * 86400 + 特征秒數
第二訓練終止秒序 = 0

def 取得資料表(
		某日誌表, 某地址日誌表, 某内核日誌表
		, 某日誌片統計表, 某地址日誌片統計表, 某内核日誌片統計表
		, 某秒序
):
	某片序 = 某秒序 // 片長

	某資料表 = pandas.concat([
		某日誌片統計表.loc[某日誌片統計表.收集片序 >= 某片序 - 樣本秒數 / 片長, ["序列號", "生産商", "賣方"]]
		, 某内核日誌片統計表.loc[某内核日誌片統計表.收集片序 >= 某片序 - 樣本秒數 / 片長, ["序列號", "生産商", "賣方"]]
		, 某地址日誌片統計表.loc[某地址日誌片統計表.收集片序 >= 某片序 - 樣本秒數 / 片長, ["序列號", "生産商", "賣方"]]
	], ignore_index=True).drop_duplicates(ignore_index=True)
	某資料表["預測秒序"] = 某秒序
	某資料表 = 某資料表.set_index(["序列號", "生産商", "賣方"])

	for 甲 in [60, 30, 15]:
		某資料表[
			["近%d秒日誌數量" % 甲, "近%d秒日誌去重數量" % 甲]
			+ ["近%d秒日誌事務%s比例" % (甲, 子) for 子 in (0, 1, 2, 3)]
		] = 某日誌表.loc[某日誌表.收集秒序 >= 某秒序 - 甲] \
			.groupby(["序列號", "生産商", "賣方"]) \
			.aggregate({
				"收集秒序": ["count", "nunique"]
				, **{("事務%s" % 子): "mean" for 子 in (0, 1, 2, 3)}
			})

		某資料表[
			["近%d秒地址日誌數量" % 甲, "近%d秒地址日誌去重數量" % 甲, "近%d秒地址記憶體種數" % 甲, "近%d秒地址記憶體面種數" % 甲, "近%d秒地址記憶體面庫種數" % 甲, "近%d秒地址記憶體面庫列種數" % 甲, "近%d秒地址記憶體面庫欄種數" % 甲, "近%d秒地址記憶體面庫列欄種數" % 甲]
		] = 某地址日誌表.loc[某地址日誌表.收集秒序 >= 某秒序 - 甲] \
			.groupby(["序列號", "生産商", "賣方"]) \
			.aggregate({"收集秒序": ["count", "nunique"], "記憶體": "nunique", "記憶體面": "nunique", "記憶體面庫": "nunique", "記憶體面庫列": "nunique", "記憶體面庫欄": "nunique", "記憶體面庫列欄": "nunique"})

		某資料表[
			["近%d秒内核日誌數量" % 甲, "近%d秒内核日誌去重數量" % 甲]
			+ [丑 for 子 in 内核欄 for 丑 in ["近%d秒内核%s數量" % (甲, 子), "近%d秒内核%s比例" % (甲, 子)]]
		]= 某内核日誌表.loc[某内核日誌表.收集秒序 >= 某秒序 - 甲] \
			.groupby(["序列號", "生産商", "賣方"]) \
			.aggregate({"收集秒序": ["count", "nunique"], **{("内核%s" % 子): ["sum", "mean"] for 子 in 内核欄}})
		某資料表[["近%d秒内核%s數量" % (甲, 子) for 子 in 内核欄]] = (某資料表[["近%d秒内核%s數量" % (甲, 子) for 子 in 内核欄]] > 0).astype("float")

	for 甲 in (32, 8, 2):
		某資料表[
			["近%d片日誌數量" % 甲]
		] = 某日誌片統計表.loc[某日誌片統計表.收集片序 >= 某片序 - 甲] \
			.groupby(["序列號", "生産商", "賣方"]) \
			.aggregate({"日誌數量": "sum"})

		某資料表[
			["近%d片内核日誌數量" % 甲]
			+ ["近%d片内核%s數量" % (甲, 子) for 子 in 内核欄]
		]= 某内核日誌片統計表.loc[某内核日誌片統計表.收集片序 >= 某片序 - 甲] \
			.groupby(["序列號", "生産商", "賣方"]) \
			.aggregate({"内核日誌數量": "sum", **{("内核%s數量" % 子): "sum" for 子 in 内核欄}})

		某資料表[["近%d片内核%s數量" % (甲, 子) for 子 in 内核欄]] = (某資料表[["近%d片内核%s數量" % (甲, 子) for 子 in 内核欄]] > 0).astype("float")

	某資料表 = 某資料表.reset_index()

	return 某資料表


if __name__ == "__main__":
	訓練日誌表 = pandas.concat([
		pandas.read_csv(目録 + "/data/memory_sample_mce_log_round1_a_train.csv", header=0, names=["序列號", "機器檢査架構", "事務", "收集時間", "生産商", "賣方"])
	], ignore_index=True)
	訓練日誌表["收集秒序"] = (pandas.to_datetime(訓練日誌表["收集時間"]) - datetime.datetime.strptime("20190801", "%Y%m%d")).dt.total_seconds()
	for 甲 in ("Z", "AP", "G", "F", "BB", "E", "CC", "AF", "AE"):
		訓練日誌表["機器檢査架構%s" % 甲] = (訓練日誌表.機器檢査架構 == 甲).astype("float")
	for 甲 in (0, 1, 2, 3):
		訓練日誌表["事務%s" % 甲] = (訓練日誌表.事務 == 甲).astype("float")

	訓練地址日誌表 = pandas.concat([
		pandas.read_csv(目録 + "/data/memory_sample_address_log_round1_a_train.csv", header=0, names=["序列號", "記憶體", "面", "庫", "列", "欄", "收集時間", "生産商", "賣方"])
	], ignore_index=True)
	訓練地址日誌表["收集秒序"] = (pandas.to_datetime(訓練地址日誌表["收集時間"]) - datetime.datetime.strptime("20190801", "%Y%m%d")).dt.total_seconds()
	訓練地址日誌表["記憶體面"] = 訓練地址日誌表.記憶體.astype("str") + "_" + 訓練地址日誌表.面.astype("str")
	訓練地址日誌表["記憶體面庫"] = 訓練地址日誌表.記憶體面 + "_" + 訓練地址日誌表.庫.astype("str")
	訓練地址日誌表["記憶體面庫列"] = 訓練地址日誌表.記憶體面庫 + "_" + 訓練地址日誌表.列.astype("str")
	訓練地址日誌表["記憶體面庫欄"] = 訓練地址日誌表.記憶體面庫 + "_" + 訓練地址日誌表.欄.astype("str")
	訓練地址日誌表["記憶體面庫列欄"] = 訓練地址日誌表.記憶體面庫列 + "_" + 訓練地址日誌表.欄.astype("str")

	訓練内核日誌表 = pandas.concat([
		pandas.read_csv(目録 + "/data/memory_sample_kernel_log_round1_a_train.csv", header=0, names=["收集時間"] + ["内核%s" % 子 for 子 in range(24)] + ["序列號", "生産商", "賣方"])
	], ignore_index=True)
	訓練内核日誌表["收集秒序"] = (pandas.to_datetime(訓練内核日誌表["收集時間"]) - datetime.datetime.strptime("20190801", "%Y%m%d")).dt.total_seconds()

	訓練故障表 = pandas.read_csv(目録 + "/data/memory_sample_failure_tag_round1_a_train.csv", header=0, names=["序列號", "故障時間", "故障類型", "生産商", "賣方"])
	訓練故障表["故障秒序"] = (pandas.to_datetime(訓練故障表["故障時間"]) - datetime.datetime.strptime("20190801", "%Y%m%d")).dt.total_seconds()

	第二訓練日誌表 = pandas.concat([
		pandas.read_csv(目録 + "/data/memory_sample_mce_log_round1_b_test.csv", header=0, names=["序列號", "機器檢査架構", "事務", "收集時間", "生産商", "賣方"])
	], ignore_index=True)
	第二訓練日誌表["收集秒序"] = (pandas.to_datetime(第二訓練日誌表["收集時間"]) - datetime.datetime.strptime("20190801", "%Y%m%d")).dt.total_seconds()
	for 甲 in ["Z", "AP", "G", "F", "BB", "E", "CC", "AF", "AE"]:
		第二訓練日誌表["機器檢査架構%s" % 甲] = (第二訓練日誌表.機器檢査架構 == 甲).astype("float")
	for 甲 in [0, 1, 2, 3]:
		第二訓練日誌表["事務%s" % 甲] = (第二訓練日誌表.事務 == 甲).astype("float")

	第二訓練地址日誌表 = pandas.concat([
		pandas.read_csv(目録 + "/data/memory_sample_address_log_round1_b_test.csv", header=0, names=["序列號", "記憶體", "面", "庫", "列", "欄", "收集時間", "生産商", "賣方"])
	], ignore_index=True)
	第二訓練地址日誌表["收集秒序"] = (pandas.to_datetime(第二訓練地址日誌表["收集時間"]) - datetime.datetime.strptime("20190801", "%Y%m%d")).dt.total_seconds()
	第二訓練地址日誌表["記憶體面"] = 第二訓練地址日誌表.記憶體.astype("str") + "_" + 第二訓練地址日誌表.面.astype("str")
	第二訓練地址日誌表["記憶體面庫"] = 第二訓練地址日誌表.記憶體面 + "_" + 第二訓練地址日誌表.庫.astype("str")
	第二訓練地址日誌表["記憶體面庫列"] = 第二訓練地址日誌表.記憶體面庫 + "_" + 第二訓練地址日誌表.列.astype("str")
	第二訓練地址日誌表["記憶體面庫欄"] = 第二訓練地址日誌表.記憶體面庫 + "_" + 第二訓練地址日誌表.欄.astype("str")
	第二訓練地址日誌表["記憶體面庫列欄"] = 第二訓練地址日誌表.記憶體面庫列 + "_" + 第二訓練地址日誌表.欄.astype("str")

	第二訓練内核日誌表 = pandas.concat([
		pandas.read_csv(目録 + "/data/memory_sample_kernel_log_round1_b_test.csv", header=0, names=["收集時間"] + ["内核%s" % 子 for 子 in range(24)] + ["序列號", "生産商", "賣方", "故障時間", "故障類型"])
	], ignore_index=True)
	第二訓練内核日誌表["收集秒序"] = (pandas.to_datetime(第二訓練内核日誌表["收集時間"]) - datetime.datetime.strptime("20190801", "%Y%m%d")).dt.total_seconds()

	第二訓練故障表 = 第二訓練内核日誌表.loc[~第二訓練内核日誌表.故障時間.isna(), ["序列號", "故障時間", "故障類型", "生産商", "賣方"]].drop_duplicates(subset=["序列號", "生産商", "賣方"], ignore_index=True)
	第二訓練故障表["故障秒序"] = (pandas.to_datetime(第二訓練故障表["故障時間"]) - datetime.datetime.strptime("20190801", "%Y%m%d")).dt.total_seconds()
	第二訓練内核日誌表 = 第二訓練内核日誌表.drop(["故障時間", "故障類型"], axis=1)

	訓練日誌表 = pandas.concat([訓練日誌表.loc[訓練日誌表.收集秒序 > 訓練起始秒序 - 特征秒數], 第二訓練日誌表], ignore_index=True)
	訓練地址日誌表 = pandas.concat([訓練地址日誌表.loc[訓練地址日誌表.收集秒序 > 訓練起始秒序 - 特征秒數], 第二訓練地址日誌表], ignore_index=True)
	訓練内核日誌表 = pandas.concat([訓練内核日誌表.loc[訓練内核日誌表.收集秒序 > 訓練起始秒序 - 特征秒數], 第二訓練内核日誌表], ignore_index=True)
	空日誌表 = 訓練日誌表[:0]
	空地址日誌表 = 訓練地址日誌表[:0]
	空内核日誌表 = 訓練内核日誌表[:0]

	for 甲表 in (訓練日誌表, 訓練地址日誌表, 訓練内核日誌表):
		甲表["收集片序"] = 甲表["收集秒序"] // 片長
		甲表["收集日序"] = 甲表["收集秒序"] // 86400
	訓練日誌片表字典 = {子[0]:子[1] for 子 in 訓練日誌表.groupby("收集片序")}
	訓練地址日誌片表字典 = {子[0]:子[1] for 子 in 訓練地址日誌表.groupby("收集片序")}
	訓練内核日誌片表字典 = {子[0]:子[1] for 子 in 訓練内核日誌表.groupby("收集片序")}
	空日誌片表 = 訓練日誌片表字典[list(訓練日誌片表字典.keys())[0]][0:0]
	空地址日誌片表 = 訓練地址日誌片表字典[list(訓練地址日誌片表字典.keys())[0]][0:0]
	空内核日誌片表 = 訓練内核日誌片表字典[list(訓練内核日誌片表字典.keys())[0]][0:0]

	訓練日誌片統計表 = 訓練日誌表.groupby(["序列號", "生産商", "賣方", "收集片序"]).aggregate({
		"收集秒序": ["count", "nunique"]
		, **{("事務%s" % 子): "sum" for 子 in (0, 1, 2, 3)}
		, **{("機器檢査架構%s" % 子): "sum" for 子 in ("Z", "AP", "G", "F", "BB", "E", "CC", "AF", "AE")}
	}).reset_index()
	訓練日誌片統計表.columns = ["序列號", "生産商", "賣方", "收集片序", "日誌數量", "日誌去重數量"] + ["事務%s數量" % 子 for 子 in (0, 1, 2, 3)] + ["機器檢査架構%s數量" % 子 for 子 in ("Z", "AP", "G", "F", "BB", "E", "CC", "AF", "AE")]
	訓練地址日誌片統計表 = 訓練地址日誌表.groupby(["序列號", "生産商", "賣方", "收集片序"]).aggregate({
		"收集秒序": ["count", "nunique"], "記憶體": "nunique", "記憶體面": "nunique", "記憶體面庫": "nunique", "記憶體面庫列": "nunique", "記憶體面庫欄": "nunique", "記憶體面庫列欄": "nunique"
	}).reset_index()
	訓練地址日誌片統計表.columns = ["序列號", "生産商", "賣方", "收集片序", "地址日誌數量", "地址日誌去重數量", "地址記憶體種數", "地址記憶體面種數", "地址記憶體面庫種數", "地址記憶體面庫列種數", "地址記憶體面庫欄種數", "地址記憶體面庫列欄種數"]
	訓練内核日誌片統計表 = 訓練内核日誌表.groupby(["序列號", "生産商", "賣方", "收集片序"]).aggregate({
		"收集秒序": ["count", "nunique"], **{("内核%s" % 子): "sum" for 子 in 内核欄}
	}).reset_index()
	訓練内核日誌片統計表.columns = ["序列號", "生産商", "賣方", "收集片序", "内核日誌數量", "内核日誌去重數量"] + ["内核%d數量" % 子 for 子 in 内核欄]
	訓練日誌片統計表字典 = {子[0]:子[1] for 子 in 訓練日誌片統計表.groupby("收集片序")}
	訓練地址日誌片統計表字典 = {子[0]:子[1] for 子 in 訓練地址日誌片統計表.groupby("收集片序")}
	訓練内核日誌片統計表字典 = {子[0]:子[1] for 子 in 訓練内核日誌片統計表.groupby("收集片序")}
	空日誌片統計表 = 訓練日誌片統計表字典[list(訓練日誌片統計表字典.keys())[0]][0:0]
	空地址日誌片統計表 = 訓練地址日誌片統計表字典[list(訓練地址日誌片統計表字典.keys())[0]][0:0]
	空内核日誌片統計表 = 訓練内核日誌片統計表字典[list(訓練内核日誌片統計表字典.keys())[0]][0:0]

	with open(目録 + "/user_data/空表元組", "wb") as 档案:
		pickle.dump((空日誌表, 空地址日誌表, 空内核日誌表, 空日誌片表, 空地址日誌片表, 空内核日誌片表, 空日誌片統計表, 空地址日誌片統計表, 空内核日誌片統計表), 档案)
	print(str(datetime.datetime.now()) + "\t匯入完畢！")



	池 = multiprocessing.Pool(處理緒數)
	結果 = []
	for 甲秒序 in list(range(訓練起始秒序, 訓練終止秒序, 片長)) + list(range(第二訓練起始秒序, 第二訓練終止秒序, 片長)):
		甲選中片序 = [甲秒序 // 片長 - 1 - 子 for 子 in range(特征秒數 // 片長)]
		結果 += [池.apply_async(取得資料表, (
			訓練日誌片表字典.get(甲選中片序[0], 空日誌片表)
			, 訓練地址日誌片表字典.get(甲選中片序[0], 空地址日誌片表)
			, 訓練内核日誌片表字典.get(甲選中片序[0], 空内核日誌片表)
			, pandas.concat([訓練日誌片統計表字典.get(子, 空日誌片統計表) for 子 in 甲選中片序], ignore_index=True)
			, pandas.concat([訓練地址日誌片統計表字典.get(子, 空地址日誌片統計表) for 子 in 甲選中片序], ignore_index=True)
			, pandas.concat([訓練内核日誌片統計表字典.get(子, 空内核日誌片統計表) for 子 in 甲選中片序], ignore_index=True)
			, 甲秒序
		))]

	池.close()
	池.join()
	訓練資料表 = pandas.concat([子.get() for 子 in 結果], ignore_index=True)

	del 訓練日誌表
	del 訓練地址日誌表
	del 訓練内核日誌表
	del 第二訓練日誌表
	del 第二訓練地址日誌表
	del 第二訓練内核日誌表
	del 訓練日誌片表字典
	del 訓練地址日誌片表字典
	del 訓練内核日誌片表字典
	del 訓練日誌片統計表字典
	del 訓練地址日誌片統計表字典
	del 訓練内核日誌片統計表字典








	訓練故障表 = pandas.concat([訓練故障表.loc[訓練故障表.故障秒序 > 訓練起始秒序 - 特征秒數], 第二訓練故障表], ignore_index=True)
	
	訓練資料表 = 訓練資料表.merge(訓練故障表.loc[:, ["序列號", "生産商", "賣方", "故障秒序"]], on=["序列號", "生産商", "賣方"], how="left")
	for 甲, 甲日數 in enumerate([1, 3, 5, 7, 9, 12, 16]):
		訓練資料表["標籤%d" % 甲] = ((訓練資料表.預測秒序 < 訓練資料表.故障秒序) & (訓練資料表.預測秒序 >= -甲日數 * 86400 + 訓練資料表.故障秒序)).astype("float")
	訓練資料表 = 訓練資料表.drop("故障秒序", axis=1)
	訓練資料表 = 訓練資料表.loc[:, ["序列號", "生産商", "賣方", "預測秒序"] + ["標籤%d" % 子 for 子 in range(7)]  + [子 for 子 in 訓練資料表.columns if 子 not in ["序列號", "生産商", "賣方", "預測秒序"] + ["標籤%d" % 子 for 子 in range(7)]]]
	print(str(datetime.datetime.now()) + "\t已生成%d列訓練資料！" % len(訓練資料表))


	for 甲 in range(7):
		輕模型 = lightgbm.train(train_set=lightgbm.Dataset(訓練資料表.iloc[:, 11:], label=訓練資料表["標籤%d" % 甲])
			, num_boost_round=1000, params={"objective": "binary", "learning_rate": 0.05, "max_depth": 6, "num_leaves": 127, "feature_fraction": 0.7, "bagging_fraction": 0.7, "verbose": -1})
		with open(目録 + "user_data/輕模型%d" % 甲, "wb") as 档案:
			pickle.dump(輕模型, 档案)
		
	訓練資料表 = 訓練資料表.loc[訓練資料表.預測秒序 < 訓練終止秒序].reset_index(drop=True)
	for 甲 in range(7):
		輕模型 = lightgbm.train(train_set=lightgbm.Dataset(訓練資料表.iloc[:, 11:], label=訓練資料表["標籤%d" % 甲])
			, num_boost_round=500, params={"objective": "binary", "learning_rate": 0.05, "max_depth": 6, "num_leaves": 127, "feature_fraction": 0.7, "bagging_fraction": 0.7, "verbose": -1})
		with open(目録 + "user_data/輕模型%d" % (7 + 甲), "wb") as 档案:
			pickle.dump(輕模型, 档案)

	print(str(datetime.datetime.now()) + "\t 訓練完畢！")

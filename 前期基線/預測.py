#python 3.9.2
#python套件 lightgbm 3.1.1
#python套件 numpy 1.20.1
#python套件 pandas 1.2.2
#輸入：
#	memory_sample_mce_log_round1_a_train.csv
#	memory_sample_mce_log_round1_a_test.csv
#	memory_sample_kernel_log_round1_a_train.csv
#	memory_sample_kernel_log_round1_a_test.csv
#	memory_sample_failure_tag_round1_a_train.csv
#輸出：
#	result.csv
#	result.zip
#
import datetime
import gc
import lightgbm
import multiprocessing
import pandas
import zipfile

訓練起始秒序 = -120 * 86400
訓練終止秒序 = -7 * 86400
測試起始秒序 = 0 * 86400
測試終止秒序 = 30 * 86400
處理緒數 = 16
片長 = 240
特征窗口長度 = 480


def 取得資料表(某日誌表, 某內核日誌表, 某秒序):
	某資料表 = 某日誌表.loc[:, ["序列號", "生產商", "賣方"]].drop_duplicates()
	某資料表["預測秒序"] = 某秒序
	某資料表 = 某資料表.set_index(["序列號", "生產商", "賣方"])

	for 甲 in [480, 240, 120, 60, 30, 15]:
		某資料表[
			["近%d秒日誌數量" % 甲]
			+ ["近%d秒事務%s" % (甲, 子) for 子 in [0, 1, 2, 3]]
			+ ["近%d秒機器檢查架構%s" % (甲, 子) for 子 in ["Z", "AP", "G", "F", "BB", "E", "CC", "AF", "AE"]]
		] = 某日誌表.loc[某日誌表.收集秒序 >= 某秒序 - 甲] \
			.groupby(["序列號", "生產商", "賣方"]) \
			.aggregate({"收集秒序": "count"
				, **{("事務%s" % 子): "sum" for 子 in [0, 1, 2, 3]}
				, **{("機器檢查架構%s" % 子): "sum" for 子 in ["Z", "AP", "G", "F", "BB", "E", "CC", "AF", "AE"]}
			})
		某資料表[
			["近%d秒內核日誌數量" % 甲]
			+ ["近%d秒列%s" % (甲, 子) for 子 in range(24)]
		]= 某內核日誌表.loc[某內核日誌表.收集秒序 >= 某秒序 - 甲] \
			.groupby(["序列號", "生產商", "賣方"]) \
			.aggregate({"收集秒序": "count"
				, **{("列%s" % 子): "sum" for 子 in range(24)}
			})

	某資料表 = 某資料表.reset_index()

	return 某資料表

def 取得訓練資料表(某日誌表, 某內核日誌表, 訓練故障日誌表, 某秒序):
	某訓練資料表 = 取得資料表(某日誌表, 某內核日誌表, 某秒序)
	某訓練資料表 = 某訓練資料表.merge(訓練故障日誌表.loc[:, ["序列號", "生產商", "賣方", "故障秒序"]], on=["序列號", "生產商", "賣方"], how="left")
	某訓練資料表["標籤"] = ((某訓練資料表.預測秒序 < 某訓練資料表.故障秒序) & (某訓練資料表.預測秒序 >= -7 * 86400 + 某訓練資料表.故障秒序)).astype("float")
	某訓練資料表 = 某訓練資料表.drop("故障秒序", axis=1)
	某訓練資料表 = 某訓練資料表.loc[:, ["序列號", "生產商", "賣方", "預測秒序", "標籤"] + [子 for 子 in 某訓練資料表.columns if 子 not in ["序列號", "生產商", "賣方", "預測秒序", "標籤"]]]

	return 某訓練資料表

def 取得測試資料表(某日誌表, 某內核日誌表, 某秒序):
	某測試資料表 = 取得資料表(某日誌表, 某內核日誌表, 某秒序)
	某測試資料表["標籤"] = -1
	某測試資料表 = 某測試資料表.loc[:, ["序列號", "生產商", "賣方", "預測秒序", "標籤"] + [子 for 子 in 某測試資料表.columns if 子 not in ["序列號", "生產商", "賣方", "預測秒序", "標籤"]]]
	
	return 某測試資料表
	

if __name__ == "__main__":
	訓練日誌表 = pandas.concat([
		pandas.read_csv("memory_sample_mce_log_round1_a_train.csv", header=0, names=["序列號", "機器檢查架構", "事務", "收集時間", "生產商", "賣方"])
		, pandas.read_csv("memory_sample_mce_log_round1_a_test.csv", header=0, names=["序列號", "機器檢查架構", "事務", "收集時間", "生產商", "賣方"])
	], ignore_index=True)
	訓練日誌表["收集秒序"] = [(datetime.datetime.strptime(子, "%Y-%m-%d %H:%M:%S") - datetime.datetime.strptime("20190601", "%Y%m%d")).total_seconds() for 子 in 訓練日誌表.收集時間]
	for 甲 in ["Z", "AP", "G", "F", "BB", "E", "CC", "AF", "AE"]:
		訓練日誌表["機器檢查架構%s" % 甲] = (訓練日誌表.機器檢查架構 == 甲).astype("float")
	for 甲 in [0, 1, 2, 3]:
		訓練日誌表["事務%s" % 甲] = (訓練日誌表.事務 == 甲).astype("float")

	訓練內核日誌表 = pandas.concat([
		pandas.read_csv("memory_sample_kernel_log_round1_a_train.csv", header=0, names=["收集時間"] + ["列%d" % 子 for 子 in range(24)] + ["序列號", "生產商", "賣方"])
		, pandas.read_csv("memory_sample_kernel_log_round1_a_test.csv", header=0, names=["收集時間"] + ["列%d" % 子 for 子 in range(24)] + ["序列號", "生產商", "賣方"])
	], ignore_index=True)
	訓練內核日誌表["收集秒序"] = [(datetime.datetime.strptime(子, "%Y-%m-%d %H:%M:%S") - datetime.datetime.strptime("20190601", "%Y%m%d")).total_seconds() for 子 in 訓練內核日誌表.收集時間]

	訓練故障日誌表 = pandas.read_csv("memory_sample_failure_tag_round1_a_train.csv", header=0, names=["序列號", "故障時間", "故障類型", "生產商", "賣方"])
	訓練故障日誌表["故障秒序"] = [(datetime.datetime.strptime(子, "%Y-%m-%d %H:%M:%S") - datetime.datetime.strptime("20190601", "%Y%m%d")).total_seconds() for 子 in 訓練故障日誌表.故障時間]
	
	訓練日誌表分割字典 = {子[0]:子[1] for 子 in 訓練日誌表.groupby(訓練日誌表.收集秒序 // 片長)}
	訓練內核日誌表分割字典 = {子[0]:子[1] for 子 in 訓練內核日誌表.groupby(訓練內核日誌表.收集秒序 // 片長)}

	print(str(datetime.datetime.now()) + "\t匯入完畢！")
	
	池 = multiprocessing.Pool(處理緒數)
	結果 = []
	for 甲秒序 in range(訓練起始秒序, 訓練終止秒序, 片長):
		甲選中片序 = [甲秒序 // 片長 - 1 - 子 for 子 in range(特征窗口長度 // 片長)]
		結果 += [池.apply_async(取得訓練資料表, (
			pandas.concat([訓練日誌表分割字典[子] for 子 in 甲選中片序 if 子 in 訓練日誌表分割字典], ignore_index=True)
			, pandas.concat([訓練內核日誌表分割字典[子] for 子 in 甲選中片序 if 子 in 訓練內核日誌表分割字典], ignore_index=True)
			, 訓練故障日誌表
			, 甲秒序
		))]
	
	池.close()
	池.join()
	訓練資料表 = pandas.concat([子.get() for 子 in 結果], ignore_index=True)
	print(str(datetime.datetime.now()) + "\t已生成%d列訓練資料！" % len(訓練資料表))

	輕模型 = lightgbm.train(train_set=lightgbm.Dataset(訓練資料表.iloc[:, 5:], label=訓練資料表.標籤)
		, num_boost_round=500, params={"objective": "binary", "learning_rate": 0.05, "max_depth": 6, "num_leaves": 127, "verbose": -1})

	
	
	
	池 = multiprocessing.Pool(處理緒數)
	結果 = []
	for 甲秒序 in range(測試起始秒序, 測試終止秒序, 片長):
		甲選中片序 = [甲秒序 // 片長 - 1 - 子 for 子 in range(特征窗口長度 // 片長)]
		結果 += [池.apply_async(取得測試資料表, (
			pandas.concat([訓練日誌表分割字典[子] for 子 in 甲選中片序 if 子 in 訓練日誌表分割字典], ignore_index=True)
			, pandas.concat([訓練內核日誌表分割字典[子] for 子 in 甲選中片序 if 子 in 訓練內核日誌表分割字典], ignore_index=True)
			, 甲秒序
		))]
		
	池.close()
	池.join()
	測試資料表 = pandas.concat([子.get() for 子 in 結果], ignore_index=True)
	預測打分表 = 測試資料表.loc[:, ["序列號", "生產商", "賣方", "預測秒序", "標籤"]].copy()
	預測打分表["預測打分"] = 輕模型.predict(測試資料表.iloc[:, 5:])
	
	預測打分表 = 預測打分表.loc[:, ["序列號", "生產商", "賣方", "預測秒序", "預測打分"]]

	print(str(datetime.datetime.now()) + "\t已生成%d列預測打分！" % len(預測打分表))




	原始預測表 = 預測打分表.loc[預測打分表.預測打分 > 0.5].sort_values("預測秒序", ignore_index=True)
	預測秒序字典 = {}
	預測表 = None
	for 乙, 乙元組 in enumerate(原始預測表.itertuples()):
		乙標識 = "%s_%s_%s" % (乙元組.序列號, 乙元組.生產商, 乙元組.賣方)
		if 乙標識 not in 預測秒序字典 or 乙元組.預測秒序 - 預測秒序字典[乙標識] > 7 * 86400:
			預測表 = pandas.concat([預測表, 原始預測表[乙:(1 + 乙)]], ignore_index=True)
			預測秒序字典[乙標識] = 乙元組.預測秒序
	print(str(datetime.datetime.now()) + "\t已生成%d列預測！" % len(預測打分表))

	提交表 = 預測表.loc[:, ["序列號"]]
	提交表["預測時間"] = [(datetime.datetime.strptime("20190601", "%Y%m%d") + datetime.timedelta(seconds=子)).strftime("%Y-%m-%d %H:%M:%S") for 子 in 預測表.預測秒序]
	
	提交表.to_csv("result.csv", header=None, index=None)
	壓縮檔案 = zipfile.ZipFile("result.zip", mode="w")
	壓縮檔案.write("result.csv")
	壓縮檔案.close()

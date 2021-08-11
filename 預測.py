import ai_hub
import datetime
import lightgbm
import numpy
import pandas
import pickle
import traceback
import logging
logging.getLogger("werkzeug").setLevel(logging.ERROR)

内核欄 = (0, 1, 2, 7, 8, 9, 10, 11, 12, 13, 14, 18, 19, 20, 21, 22, 23)

片長 = 60
特征秒數 = 960
樣本秒數 = 1920
刪除片數 = -1 - 特征秒數 // 片長

測試日誌片表字典 = {}
測試地址日誌片表字典 = {}
測試内核日誌片表字典 = {}
測試日誌片統計表字典 = {}
測試地址日誌片統計表字典 = {}
測試内核日誌片統計表字典 = {}
with open("資料/空表元組", "rb") as 档案:
	空日誌表, 空地址日誌表, 空内核日誌表, 空日誌片表, 空地址日誌片表, 空内核日誌片表, 空日誌片統計表, 空地址日誌片統計表, 空内核日誌片統計表 = pickle.load(档案)
try:
	歴史日誌表 = pandas.concat([
		pandas.read_csv("/tcdata/memory_sample_mce_log_round2_b_his.csv", header=0, names=["序列號", "機器檢査架構", "事務", "收集時間", "生産商", "賣方"])
	], ignore_index=True)
	歴史日誌表["收集秒序"] = (pandas.to_datetime(歴史日誌表["收集時間"]) - datetime.datetime.strptime("20190801", "%Y%m%d")).dt.total_seconds()

	歴史地址日誌表 = pandas.concat([
		pandas.read_csv("/tcdata/memory_sample_address_log_round2_b_his.csv", header=0, names=["序列號", "記憶體", "面", "庫", "列", "欄", "收集時間", "生産商", "賣方"])
	], ignore_index=True)
	歴史地址日誌表["收集秒序"] = (pandas.to_datetime(歴史地址日誌表["收集時間"]) - datetime.datetime.strptime("20190801", "%Y%m%d")).dt.total_seconds()

	歴史内核日誌表 = pandas.concat([
		pandas.read_csv("/tcdata/memory_sample_kernel_log_round2_b_his.csv", header=0, names=["收集時間"] + ["内核%d" % 子 for 子 in range(24)] + ["序列號", "生産商", "賣方"])
	], ignore_index=True)
	歴史内核日誌表["收集秒序"] = (pandas.to_datetime(歴史内核日誌表["收集時間"]) - datetime.datetime.strptime("20190801", "%Y%m%d")).dt.total_seconds()

	print("%s\t%s\t%s" % (len(歴史日誌表), len(歴史地址日誌表), len(歴史内核日誌表)))

	最小秒序 = min(歴史日誌表.收集秒序.min(), 歴史地址日誌表.收集秒序.min(), 歴史内核日誌表.收集秒序.min()) // 60 * 60
	最大秒序 = 60 + max(歴史日誌表.收集秒序.max(), 歴史地址日誌表.收集秒序.max(), 歴史内核日誌表.收集秒序.max()) // 60 * 60

	歴史日誌表 = 歴史日誌表.loc[歴史日誌表.收集秒序 >= 最大秒序 - 特征秒數]
	歴史地址日誌表 = 歴史地址日誌表.loc[歴史地址日誌表.收集秒序 >= 最大秒序 - 特征秒數]
	歴史内核日誌表 = 歴史内核日誌表.loc[歴史内核日誌表.收集秒序 >= 最大秒序 - 特征秒數]

	歴史日誌表 = 歴史日誌表.reset_index(drop=True)
	for 甲 in ["Z", "AP", "G", "F", "BB", "E", "CC", "AF", "AE"]:
		歴史日誌表["機器檢査架構%s" % 甲] = (歴史日誌表.機器檢査架構 == 甲).astype("float")
	for 甲 in [0, 1, 2, 3]:
		歴史日誌表["事務%s" % 甲] = (歴史日誌表.事務 == 甲).astype("float")
	歴史地址日誌表 = 歴史地址日誌表.reset_index(drop=True)


	歴史地址日誌表["記憶體面"] = 歴史地址日誌表.記憶體.astype("str") + "_" + 歴史地址日誌表.面.astype("str")
	歴史地址日誌表["記憶體面庫"] = 歴史地址日誌表.記憶體面 + "_" + 歴史地址日誌表.庫.astype("str")
	歴史地址日誌表["記憶體面庫列"] = 歴史地址日誌表.記憶體面庫 + "_" + 歴史地址日誌表.列.astype("str")
	歴史地址日誌表["記憶體面庫欄"] = 歴史地址日誌表.記憶體面庫 + "_" + 歴史地址日誌表.欄.astype("str")
	歴史地址日誌表["記憶體面庫列欄"] = 歴史地址日誌表.記憶體面庫列 + "_" + 歴史地址日誌表.欄.astype("str")
	歴史内核日誌表 = 歴史内核日誌表.reset_index(drop=True)


	for 甲表 in (歴史日誌表, 歴史地址日誌表, 歴史内核日誌表):
		甲表["收集片序"] = 甲表["收集秒序"] // 片長
		甲表["收集日序"] = 甲表["收集秒序"] // 86400

	測試日誌片表字典 = {子[0]: 子[1] for 子 in 歴史日誌表.groupby("收集片序")}
	測試地址日誌片表字典 = {子[0]: 子[1] for 子 in 歴史地址日誌表.groupby("收集片序")}
	測試内核日誌片表字典 = {子[0]: 子[1] for 子 in 歴史内核日誌表.groupby("收集片序")}

	歴史日誌片統計表 = 歴史日誌表.groupby(["序列號", "生産商", "賣方", "收集片序"]).aggregate({
		"收集秒序": ["count", "nunique"]
		, **{("事務%s" % 子): "sum" for 子 in (0, 1, 2, 3)}
		, **{("機器檢査架構%s" % 子): "sum" for 子 in ("Z", "AP", "G", "F", "BB", "E", "CC", "AF", "AE")}
	}).reset_index()
	歴史日誌片統計表.columns = ["序列號", "生産商", "賣方", "收集片序", "日誌數量", "日誌去重數量"] + ["事務%s數量" % 子 for 子 in (0, 1, 2, 3)] + ["機器檢査架構%s數量" % 子 for 子 in ("Z", "AP", "G", "F", "BB", "E", "CC", "AF", "AE")]
	歴史地址日誌片統計表 = 歴史地址日誌表.groupby(["序列號", "生産商", "賣方", "收集片序"]).aggregate({
		"收集秒序": ["count", "nunique"], "記憶體": "nunique", "記憶體面": "nunique", "記憶體面庫": "nunique", "記憶體面庫列": "nunique", "記憶體面庫欄": "nunique", "記憶體面庫列欄": "nunique"
	}).reset_index()
	歴史地址日誌片統計表.columns = ["序列號", "生産商", "賣方", "收集片序", "地址日誌數量", "地址日誌去重數量", "地址記憶體種數", "地址記憶體面種數", "地址記憶體面庫種數", "地址記憶體面庫列種數", "地址記憶體面庫欄種數", "地址記憶體面庫列欄種數"]
	歴史内核日誌片統計表 = 歴史内核日誌表.groupby(["序列號", "生産商", "賣方", "收集片序"]).aggregate({
		"收集秒序": ["count", "nunique"], **{("内核%s" % 子): "sum" for 子 in 内核欄}
	}).reset_index()
	歴史内核日誌片統計表.columns = ["序列號", "生産商", "賣方", "收集片序", "内核日誌數量", "内核日誌去重數量"] + ["内核%d數量" % 子 for 子 in 内核欄]
	測試日誌片統計表字典 = {子[0]: 子[1] for 子 in 歴史日誌片統計表.groupby("收集片序")}
	測試地址日誌片統計表字典 = {子[0]: 子[1] for 子 in 歴史地址日誌片統計表.groupby("收集片序")}
	測試内核日誌片統計表字典 = {子[0]: 子[1] for 子 in 歴史内核日誌片統計表.groupby("收集片序")}
	del 歴史日誌表
	del 歴史地址日誌表
	del 歴史内核日誌表
	del 歴史日誌片統計表
	del 歴史地址日誌片統計表
	del 歴史内核日誌片統計表
except Exception as 異常:
	traceback.print_exc()

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







輕模型 = []
for 甲 in range(14):
	with open("資料/輕模型%d" % 甲, "rb") as 档案:
		輕模型 += [pickle.load(档案)]

總測試資料表 = None
class 類別_推斷伺服器(ai_hub.inferServer):
	def __init__(self):
		super(類別_推斷伺服器, self).__init__(None)
		self.分鐘數 = -1
		self.日誌數 = 0
		self.地址日誌數 = 0
		self.内核日誌數 = 0
		self.候選數 = 0
		self.預測數 = 0
		self.預測序列號 = set()

	def pre_process(self, request):
		return request.get_json()

	def predict(self, data):
		try:
			self.分鐘數 += 1

			global 測試日誌片表字典, 測試地址日誌片表字典, 測試内核日誌片表字典
			global 測試日誌片統計表字典, 測試地址日誌片統計表字典, 測試内核日誌片統計表字典
			global 總測試資料表

			try:
				if len(data["mce_log"]) == 0:
					測試日誌表 = 空日誌表.copy()
				else:
					測試日誌表 = pandas.DataFrame(data["mce_log"], columns=["序列號", "機器檢査架構", "事務", "收集時間", "生産商", "賣方"])
					測試日誌表["收集秒序"] = (pandas.to_datetime(測試日誌表["收集時間"]) - datetime.datetime.strptime("20190801", "%Y%m%d")).dt.total_seconds()
					for 甲 in ["Z", "AP", "G", "F", "BB", "E", "CC", "AF", "AE"]:
						測試日誌表["機器檢査架構%s" % 甲] = (測試日誌表.機器檢査架構 == 甲).astype("float")
					for 甲 in [0, 1, 2, 3]:
						測試日誌表["事務%s" % 甲] = (測試日誌表.事務 == 甲).astype("float")
			except:
				測試日誌表 = 空日誌表.copy()

			try:
				if len(data["address_log"]) == 0:
					測試地址日誌表 = 空地址日誌表.copy()
				else:
					測試地址日誌表 = pandas.DataFrame(data["address_log"], columns=["序列號", "記憶體", "面", "庫", "列", "欄", "收集時間", "生産商", "賣方"])
					測試地址日誌表["收集秒序"] = (pandas.to_datetime(測試地址日誌表["收集時間"]) - datetime.datetime.strptime("20190801", "%Y%m%d")).dt.total_seconds()
					測試地址日誌表["記憶體面"] = 測試地址日誌表.記憶體.astype("str") + "_" + 測試地址日誌表.面.astype("str")
					測試地址日誌表["記憶體面庫"] = 測試地址日誌表.記憶體面 + "_" + 測試地址日誌表.庫.astype("str")
					測試地址日誌表["記憶體面庫列"] = 測試地址日誌表.記憶體面庫 + "_" + 測試地址日誌表.列.astype("str")
					測試地址日誌表["記憶體面庫欄"] = 測試地址日誌表.記憶體面庫 + "_" + 測試地址日誌表.欄.astype("str")
					測試地址日誌表["記憶體面庫列欄"] = 測試地址日誌表.記憶體面庫列 + "_" + 測試地址日誌表.欄.astype("str")
			except:
				測試地址日誌表 = 空地址日誌表.copy()

			try:
				if len(data["kernel_log"]) == 0:
					測試内核日誌表 = 空内核日誌表.copy()
				else:
					測試内核日誌表 = pandas.DataFrame(data["kernel_log"], columns=["收集時間"] + ["内核%d" % 子 for 子 in range(24)] + ["序列號", "生産商", "賣方"])
					測試内核日誌表["收集秒序"] = (pandas.to_datetime(測試内核日誌表["收集時間"]) - datetime.datetime.strptime("20190801", "%Y%m%d")).dt.total_seconds()
			except:
				測試内核日誌表 = 空内核日誌表.copy()

			if len(測試日誌表) > 0:
				當前秒序 = 測試日誌表.收集秒序[0]
			elif len(測試地址日誌表) > 0:
				當前秒序 = 測試地址日誌表.收集秒序[0]
			elif len(測試内核日誌表) > 0:
				當前秒序 = 測試内核日誌表.收集秒序[0]
			else:
				return "[]"

			self.日誌數 += len(測試日誌表)
			self.地址日誌數 += len(測試地址日誌表)
			self.内核日誌數 += len(測試内核日誌表)

			當前秒序 = 片長 + 當前秒序 // 片長 * 片長
			當前片序 = 當前秒序 // 片長

			for 甲表 in (測試日誌表, 測試地址日誌表, 測試内核日誌表):
				甲表["收集片序"] = 甲表["收集秒序"] // 片長
				甲表["收集日序"] = 甲表["收集秒序"] // 86400
			刪除片序 = 當前片序 + 刪除片數

			if 刪除片序 in 測試日誌片表字典:
				測試日誌片表字典.pop(刪除片序)
				測試地址日誌片表字典.pop(刪除片序)
				測試内核日誌片表字典.pop(刪除片序)
				測試日誌片統計表字典.pop(刪除片序)
				測試地址日誌片統計表字典.pop(刪除片序)
				測試内核日誌片統計表字典.pop(刪除片序)

			測試日誌片表字典[當前片序 - 1] = 測試日誌表
			測試地址日誌片表字典[當前片序 - 1] = 測試地址日誌表
			測試内核日誌片表字典[當前片序 - 1] = 測試内核日誌表

			測試日誌片統計表 = 測試日誌表.groupby(["序列號", "生産商", "賣方", "收集片序"]).aggregate({
				"收集秒序": ["count", "nunique"]
				, **{("事務%s" % 子): "sum" for 子 in (0, 1, 2, 3)}
				, **{("機器檢査架構%s" % 子): "sum" for 子 in ("Z", "AP", "G", "F", "BB", "E", "CC", "AF", "AE")}
			}).reset_index()
			測試日誌片統計表.columns = ["序列號", "生産商", "賣方", "收集片序", "日誌數量", "日誌去重數量"] + ["事務%s數量" % 子 for 子 in (0, 1, 2, 3)] + ["機器檢査架構%s數量" % 子 for 子 in ("Z", "AP", "G", "F", "BB", "E", "CC", "AF", "AE")]
			測試地址日誌片統計表 = 測試地址日誌表.groupby(["序列號", "生産商", "賣方", "收集片序"]).aggregate({
				"收集秒序": ["count", "nunique"], "記憶體": "nunique", "記憶體面": "nunique", "記憶體面庫": "nunique", "記憶體面庫列": "nunique", "記憶體面庫欄": "nunique", "記憶體面庫列欄": "nunique"
			}).reset_index()
			測試地址日誌片統計表.columns = ["序列號", "生産商", "賣方", "收集片序", "地址日誌數量", "地址日誌去重數量", "地址記憶體種數", "地址記憶體面種數", "地址記憶體面庫種數", "地址記憶體面庫列種數", "地址記憶體面庫欄種數", "地址記憶體面庫列欄種數"]
			測試内核日誌片統計表 = 測試内核日誌表.groupby(["序列號", "生産商", "賣方", "收集片序"]).aggregate({
				"收集秒序": ["count", "nunique"], **{("内核%s" % 子): "sum" for 子 in 内核欄}
			}).reset_index()
			測試内核日誌片統計表.columns = ["序列號", "生産商", "賣方", "收集片序", "内核日誌數量", "内核日誌去重數量"] + ["内核%d數量" % 子 for 子 in 内核欄]
			測試日誌片統計表字典[當前片序 - 1] = 測試日誌片統計表
			測試地址日誌片統計表字典[當前片序 - 1] = 測試地址日誌片統計表
			測試内核日誌片統計表字典[當前片序 - 1] = 測試内核日誌片統計表

			選中片序 = [當前片序 - 1 - 子 for 子 in range(特征秒數 // 片長)]
			測試資料表 = 取得資料表(
				測試日誌片表字典.get(選中片序[0], 空日誌片表)
				, 測試地址日誌片表字典.get(選中片序[0], 空地址日誌片表)
				, 測試内核日誌片表字典.get(選中片序[0], 空内核日誌片表)
				, pandas.concat([測試日誌片統計表字典.get(子, 空日誌片統計表) for 子 in 選中片序], ignore_index=True)
				, pandas.concat([測試地址日誌片統計表字典.get(子, 空地址日誌片統計表) for 子 in 選中片序], ignore_index=True)
				, pandas.concat([測試内核日誌片統計表字典.get(子, 空内核日誌片統計表) for 子 in 選中片序], ignore_index=True)
				, 當前秒序
			)
			if len(測試資料表) == 0:
				return "[]"

			測試資料表["標籤"] = -1
			測試資料表 = 測試資料表.loc[:, ["序列號", "生産商", "賣方", "預測秒序", "標籤"] + [子 for 子 in 測試資料表.columns if 子 not in ["序列號", "生産商", "賣方", "預測秒序", "標籤"]]]
			總測試資料表 = pandas.concat([總測試資料表, pandas.DataFrame(測試資料表.iloc[:, 5:].sum(axis=0)).T], ignore_index=True)
			self.候選數 += len(測試資料表)

			預測打分表 = 測試資料表.loc[:, ["序列號"]].copy()
			預測打分表["預測打分"] = numpy.mean([輕模型[子].predict(測試資料表.iloc[:, 5:]) for 子 in range(14)], axis=0)

			預測表 = 預測打分表.loc[預測打分表.預測打分 > 0.6, ["序列號"]].assign(預測分鐘數 = 5)
			self.預測數 += len(預測表)
			self.預測序列號 = self.預測序列號.union(預測表.序列號.to_list())
			if self.分鐘數 % 1440 == 1439:
				print("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s" % (當前秒序, self.分鐘數, self.日誌數, self.地址日誌數, self.内核日誌數, self.候選數, self.預測數, len(self.預測序列號), min(self.預測序列號), max(self.預測序列號)))

			return pandas.DataFrame({"serial_number": 預測表.序列號, "pti": 預測表.預測分鐘數}).to_json(orient="records")
		except Exception as 異常:
			traceback.print_exc()
			return "[]"


推斷伺服器 = 類別_推斷伺服器()
推斷伺服器.run()
print(最大秒序)
print(最小秒序)
總測試資料表 = pandas.DataFrame(總測試資料表.sum(axis=0)).T / 推斷伺服器.候選數
for 甲 in range(總測試資料表.shape[1]):
	print("%s\t%f" % (總測試資料表.columns.to_list()[甲], 總測試資料表.iloc[0, 甲]))

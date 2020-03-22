# 二手车交易价格预测

<left><img src="https://tianchi-public.oss-cn-hangzhou.aliyuncs.com/public/files/image/admin/15844106056484384%E8%81%94%E5%90%88logo.png" alt="img" style="zoom:25%;" /></left>

[零基础入门数据挖掘 - 二手车交易价格预测](https://tianchi.aliyun.com/competition/entrance/231784/information)

## 一、赛题数据

赛题以预测二手车的交易价格为任务，数据集报名后可见并可下载，该数据来自某交易平台的二手车交易记录，总数据量超过40w，包含31列变量信息，其中15列为匿名变量。为了保证比赛的公平性，将会从中抽取15万条作为训练集，5万条作为测试集A，5万条作为测试集B，同时会对name、model、brand和regionCode等信息进行脱敏。

### 字段表

| **Field**         | **Description**                                              |
| ----------------- | ------------------------------------------------------------ |
| SaleID            | 交易ID，唯一编码                                             |
| name              | 汽车交易名称，已脱敏                                         |
| regDate           | 汽车注册日期，例如20160101，2016年01月01日                   |
| model             | 车型编码，已脱敏                                             |
| brand             | 汽车品牌，已脱敏                                             |
| bodyType          | 车身类型：豪华轿车：0，微型车：1，厢型车：2，大巴车：3，敞篷车：4，双门汽车：5，商务车：6，搅拌车：7 |
| fuelType          | 燃油类型：汽油：0，柴油：1，液化石油气：2，天然气：3，混合动力：4，其他：5，电动：6 |
| gearbox           | 变速箱：手动：0，自动：1                                     |
| power             | 发动机功率：范围 [ 0, 600 ]                                  |
| kilometer         | 汽车已行驶公里，单位万km                                     |
| notRepairedDamage | 汽车有尚未修复的损坏：是：0，否：1                           |
| regionCode        | 地区编码，已脱敏                                             |
| seller            | 销售方：个体：0，非个体：1                                   |
| offerType         | 报价类型：提供：0，请求：1                                   |
| creatDate         | 汽车上线时间，即开始售卖时间                                 |
| price             | 二手车交易价格（预测目标）                                   |
| v系列特征         | 匿名特征，包含v0-14在内15个匿名特征                          |

## 二、评测标准

评价标准为MAE(Mean Absolute Error)。
![enter image description here](https://tianchi-public.oss-cn-hangzhou.aliyuncs.com/public/files/forum/158401047136251171584010471248.png)
MAE越小，说明模型预测得越准确。
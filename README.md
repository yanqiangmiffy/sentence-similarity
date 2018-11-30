# sentence-similarity
问题句子相似度计算，即给定客服里用户描述的两句话，用算法来判断是否表示了相同的语义。
## 句子相似度判定
今年和去年前后相继出现了多个关于句子相似度判定的比赛，即得定两个句子，用算法判断是否表示了相同的语义或者意思。下面是比赛的列表：
- [Quora Question Pairs](https://www.kaggle.com/c/quora-question-pairs)

> The goal of this competition is to predict which of the provided pairs of questions contain two questions with the same meaning. 

> [数据集](https://www.kaggle.com/c/quora-question-pairs/data)不是匿名的,用真实的英文单词标识

- [ ATEC学习赛：NLP之问题相似度计算](https://dc.cloud.alipay.com/index#/topic/intro?id=8)
> 问题相似度计算，即给定客服里用户描述的两句话，用算法来判断是否表示了相同的语义。
示例：
1. “花呗如何还款” --“花呗怎么还款”：同义问句
2. “花呗如何还款” -- “我怎么还我的花被呢”：同义问句
3. “花呗分期后逾期了如何还款”-- “花呗分期后逾期了哪里还款”：非同义问句
对于例子a，比较简单的方法就可以判定同义；对于例子b，包含了错别字、同义词、词序变换等问题，两个句子乍一看并不类似，想正确判断比较有挑战；对于例子c，两句话很类似，仅仅有一处细微的差别 “如何”和“哪里”，就导致语义不一致。
- [CCKS 2018 微众银行智能客服问句匹配大赛](https://biendata.com/competition/CCKS2018_3/leaderboard/)
- []()

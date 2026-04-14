# BrowseComp-Plus Search Skills

从96个正确案例中提取的搜索策略skills。共20个skill。

## 通用搜索策略 (7个)

| Skill | 从正确case中观察到的模式 |
|-------|------------------------|
| [knowledge-inference](knowledge-inference/) | 用世界知识将抽象描述翻译为具体实体名，直接命中文档 |
| [constraint-decomposition](constraint-decomposition/) | 拆解问题 + 翻译每个约束为文档侧词汇 + 按稀有度排序 |
| [multi-angle-reformulation](multi-angle-reformulation/) | 先失败的查询 vs 后成功的查询：换约束、换词汇、搜容器 |
| [entity-pivot](entity-pivot/) | 发现候选实体后以其名字为锚点搜索验证 |
| [phrase-targeting](phrase-targeting/) | 引号使用率约42%，仅用于标题/实体名/术语 |
| [year-enumeration](year-enumeration/) | 日期范围展开为年份列表 |
| [verification-cascade](verification-cascade/) | 58%的搜索在发现候选后用于验证 |

## 领域专属检索策略 (7个)

| Skill | 适用场景 |
|-------|---------|
| [domain-thesis-acknowledgment](domain-thesis-acknowledgment/) | 论文致谢：先定位作者→找论文→提取致谢 |
| [domain-paper-author-chain](domain-paper-author-chain/) | 论文链追踪：机构约束→定位一人→co-author链 |
| [domain-article-blog-trace](domain-article-blog-trace/) | 文章/博客：搜最独特的内容细节而非元描述 |
| [domain-actor-identification](domain-actor-identification/) | 演员全名：奖项/家族/死亡细节→年份枚举→验证 |
| [domain-movie-identification](domain-movie-identification/) | 电影识别：先识别演员/导演→filmography→交叉验证 |
| [domain-person-bio](domain-person-bio/) | 人物传记：找"最可搜索"事实→世界知识→验证 |
| [domain-science-nature](domain-science-nature/) | 科学/自然：世界知识翻译物种→用物种名精确搜索 |

## 推理/检索模式 (6个)

| Skill | 适用场景 |
|-------|---------|
| [domain-multi-hop-chain](domain-multi-hop-chain/) | A→B→C→D推理链：解最容易的hop→carry结果到下一hop |
| [domain-doi-paper-detail](domain-doi-paper-detail/) | 找DOI/论文内数字：搜方法学细节（样本量/周数/表结构） |
| [domain-cultural-niche](domain-cultural-niche/) | 文化/艺术/文学：搜最怪异的短语（exact quote命中率最高） |
| [domain-quick-hit](domain-quick-hit/) | 2-6次搜索解决：识别极独特事实→直接搜→快速确认 |
| [domain-historical-event](domain-historical-event/) | 历史事件锚定：先解析相对日期→转绝对年份→搜索 |
| [domain-interview-quote-trace](domain-interview-quote-trace/) | 采访/引语追踪：搜独特观点原文→interview包含原话 |

## 关系/结构模式 (2个 — 暂未分到上面的)

| Skill | 适用场景 |
|-------|---------|
| [domain-family-chain](domain-family-chain/) | 家族链：先找更出名的家族成员→追踪关系到目标 |
| [domain-institution-anchor](domain-institution-anchor/) | 机构锚定：从建校年份/校训/地点识别机构→用机构名搜人 |

## 正确case的数据特征

| 指标 | 正确 (n=96) |
|------|-------------|
| 平均搜索次数 | 15.3 |
| 引号使用率 | ~42% |
| 首次命中GT的搜索位置 | 第0-2次: 36个, 第3-5次: 13个, 第6+次: 35个 |
| Quick-hit (≤6次搜索) | 12个case |

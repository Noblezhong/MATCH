# Seen 284 分层抽样子集

- 原索引：`/HDD2/TravelUAV_dataset/TravelUAV_data/data/uav_dataset/seen_valset.json`
- 子集索引：`/HDD2/TravelUAV_dataset/TravelUAV_data/data/uav_dataset/seen_valset_stratified284_seed20261117.json`
- 抽样清单：`/HDD2/TravelUAV_dataset/TravelUAV_data/data/uav_dataset/seen_valset_stratified284_seed20261117.selection.json`
- 原集 1418 条，子集 284 条（20.028%），索引含 15019 条帧记录。
- 固定随机种子：20261117；按场景比例采用最大余数法分配，再按场景内短、中、长轨迹分层无放回抽样。
- 轨迹长度为 trajectory_raw_detailed 相邻 position 的三维距离之和；同长度按 UUID 排序。三组尽量等大，多余条目优先分给短、中组。
- 分层配额同样使用最大余数法，余数相同按短、中、长排序；场景余数相同按名称排序。
- 所有系统使用同一索引；本文件仅供快速评估，不能据此保证完整集上的小幅性能差异。

| UE 场景 | 原集 | 子集 | 短 / 中 / 长 |
|---|---:|---:|---|
| Carla_Town01 | 165 | 33 | 11 / 11 / 11 |
| Carla_Town04 | 106 | 21 | 7 / 7 / 7 |
| Carla_Town05 | 73 | 15 | 5 / 5 / 5 |
| Carla_Town06 | 47 | 10 | 4 / 3 / 3 |
| Carla_Town07 | 60 | 12 | 4 / 4 / 4 |
| Carla_Town10HD | 141 | 28 | 10 / 9 / 9 |
| Carla_Town15 | 111 | 22 | 8 / 7 / 7 |
| ModernCityMap | 124 | 25 | 9 / 8 / 8 |
| NYCEnvironmentMegapa | 281 | 56 | 19 / 19 / 18 |
| NewYorkCity | 145 | 29 | 10 / 10 / 9 |
| TropicalIsland | 165 | 33 | 11 / 11 / 11 |
| **合计** | **1418** | **284** | |

长度统计单位：米。

| 集合 | 最短 | 中位数 | 平均 | 最长 |
|---|---:|---:|---:|---:|
| 原集 | 12.636 | 232.174 | 247.619 | 1092.758 |
| 子集 | 45.856 | 230.303 | 245.712 | 1072.797 |

静态核验通过：284 条唯一 UUID、11 个场景配额正确、全部元数据引用存在、所选轨迹原索引帧记录完整保留。没有复制图像或轨迹数据，没有启动模型、UE 或评估。

原索引 SHA256：`af5b7b660ed3b1dea95b53aa72a98f74ae7a37cd381aa0e818ee2a343cf54552`
子集索引 SHA256：`9498b9e79fa17698dbe44e98a693d30ea1b87b1f7f31eaae40301521bef0391c`

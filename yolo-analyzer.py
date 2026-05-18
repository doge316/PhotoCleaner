from pathlib import Path

import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results
import cv2


# 这个脚本的目标：
# 1. 找出图片中的所有人物
# 2. 给每个人物打一个“像不像主体”的分数
# 3. 自动保留主体人物，得到路人掩码
# 4. 后续可以把路人掩码交给 LaMa 这类修复模型去抹除

# results / res 里常用的数据说明：
# - boxes: 预测框信息，包含坐标、置信度、类别
# - masks: 分割掩码信息，和 boxes 一一对应
# - names: 类别名称字典


def compute_subject_score(box: np.ndarray, image_w: int, image_h: int, conf: float) -> float:
	"""给单个人物打分。

	分数越高，越像“主体人物”。

	评分思路：
	- 框面积越大，越可能是主体
	- 框越接近图片中心，越可能是主体
	- 模型置信度越高，越可信
	"""
	x1, y1, x2, y2 = box

	# 1) 面积得分：框面积 / 图片面积，归一化到 0~1
	box_area = max(0.0, (x2 - x1) * (y2 - y1))
	image_area = float(image_w * image_h)
	area_score = box_area / image_area if image_area > 0 else 0.0

	# 2) 中心得分：框中心离图片中心越近越好
	box_cx = (x1 + x2) / 2.0
	box_cy = (y1 + y2) / 2.0
	image_cx = image_w / 2.0
	image_cy = image_h / 2.0

	# 归一化后的中心距离，数值越小越靠近中心
	center_dist = np.sqrt((box_cx - image_cx) ** 2 + (box_cy - image_cy) ** 2)
	max_dist = np.sqrt(image_cx**2 + image_cy**2)
	center_score = 1.0 - (center_dist / max_dist if max_dist > 0 else 0.0)
	center_score = float(np.clip(center_score, 0.0, 1.0))

	# 3) 置信度得分：直接使用模型输出的 conf
	conf_score = float(np.clip(conf, 0.0, 1.0))

	# 最终总分：权重可以根据自己的图片风格继续调整
	# 当前只使用“面积 + 中心位置 + 置信度”三项
	score = (
		0.45 * area_score
		+ 0.35 * center_score
		+ 0.20 * conf_score
	)
	return float(score)


def build_union_mask(masks: np.ndarray, indices: np.ndarray) -> np.ndarray:
	"""把多个实例掩码合并成一个总掩码。

	参数：
	- masks: shape = (N, H, W)
	- indices: 要合并的实例索引

	返回：
	- shape = (H, W) 的布尔数组，True 表示该像素属于这些实例
	"""
	if masks.size == 0 or len(indices) == 0:
		return np.zeros((0, 0), dtype=bool)

	union_mask = np.zeros(masks.shape[1:], dtype=bool)
	for idx in indices:
		# 把单个人物的 mask 合并进去
		union_mask |= (masks[idx] > 0.5)
	return union_mask


# 加载分割模型
model = YOLO("yolov8s-seg.pt")

# 只检测 person 类别（COCO 里 person 的类别 id = 0）
results = model("./消除路人/训练集/test1.png", classes=[0], save=True, show=True)

# model(...) 返回的是列表；如果输入一张图片，列表里通常只有一个 Results
res: Results = results[0]

# 原图尺寸，格式是 (height, width)
image_h, image_w = res.orig_shape

# 获取所有人物框信息
boxes = res.boxes

# xyxy 是二维数组，每一行代表一个人：
# [左上角x, 左上角y, 右下角x, 右下角y]
xyxy = boxes.xyxy.cpu().numpy()

# conf: 每个人物框的置信度
conf = boxes.conf.cpu().numpy()

# cls: 每个人物框对应的类别 id
cls = boxes.cls.cpu().numpy()

# 获取人物分割掩码
# masks 的形状通常是 (N, H, W)
masks = res.masks.data.cpu().numpy() if res.masks is not None else np.array([])

# 如果没有检测到人，直接结束
if len(xyxy) == 0:
	print("没有检测到人物")
	raise SystemExit(0)

# 计算每个人的主体分数
scores = []
for i in range(len(xyxy)):
	score = compute_subject_score(xyxy[i], image_w, image_h, float(conf[i]))
	scores.append(score)

scores = np.array(scores, dtype=np.float32)

# 分数最高的人，通常最像主体
main_idx = int(np.argmax(scores))
main_score = float(scores[main_idx])

# 如果有多个主体（比如合照、情侣照），保留分数接近最高分的人
# 这里的规则是：分数 >= 最高分 * 0.75
subject_indices = np.where(scores >= main_score * 0.75)[0]

# 再加一个简单规则：太小的人通常更像路人
# 这里把框面积小于整图 3% 的人过滤掉
areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
area_keep = np.where(areas >= image_w * image_h * 0.03)[0]

# 取“分数足够高”并且“面积不太小”的交集
subject_indices = np.intersect1d(subject_indices, area_keep)

# 如果过滤完一个都不剩，那至少保留分数最高的人
if len(subject_indices) == 0:
	subject_indices = np.array([main_idx], dtype=int)

# 路人索引 = 所有人 - 主体人物
all_indices = np.arange(len(xyxy))
stray_indices = np.setdiff1d(all_indices, subject_indices)

# 把主体 mask 和路人 mask 分别合并出来
subject_mask = build_union_mask(masks, subject_indices)
stray_mask = build_union_mask(masks, stray_indices)

# 输出分析结果，方便观察当前规则是否合理
print(f"一共检测到 {len(xyxy)} 个人")
print("每个人的框坐标：")
for i, box in enumerate(xyxy):
	print(
		f"  {i}: box={box.tolist()}, conf={float(conf[i]):.3f}, "
		f"score={float(scores[i]):.3f}, area={float(areas[i]):.0f}"
	)

print(f"主体人物索引：{subject_indices.tolist()}")
print(f"路人索引：{stray_indices.tolist()}")

# 下面这两个结果可以直接交给修复模型：
# - subject_mask：主体区域
# - stray_mask：需要消除的路人区域
#
# 如果后面要接 LaMa，一般就是把 stray_mask 保存成黑白图，
# 然后把原图 + 这个 mask 送进去做 inpaint。


#输出掩码图，白色部分是对应的区域
cv2.imwrite("./mask/subject_mask.png", (subject_mask * 255).astype(np.uint8))
cv2.imwrite("./mask/stray_mask.png", (stray_mask * 255).astype(np.uint8))







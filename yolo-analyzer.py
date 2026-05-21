from photo_cleaner_core import run_cli


if __name__ == "__main__":
	run_cli(
		input_path="./消除路人/训练集/test1.png",
		output_dir="./消除路人/结果集",
		model_path="yolov8s-seg.pt",
	)







from types import SimpleNamespace


config = SimpleNamespace(**{})
#
config.device = "cuda"
config.crops_dir = "./intermediate_crops_dir"
config.results_dir = "./results"
config.run_name = "./image_inpainting"
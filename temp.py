from generate_scrolls_dist import gather_results_from_each_node, combine_partial_results, save_json
from pathlib import Path
save_dir = Path('tmp_gen')
json_save_dir = Path('tmp_gen_tmp')
save_dir.mkdir(exist_ok=True)
partial_results = gather_results_from_each_node(4, json_save_dir, 600)
final_results = combine_partial_results(partial_results)
save_path = save_dir.joinpath(f"adape.json")
print(f"Saving aggregated results at {save_path}, intermediate in {json_save_dir}/")
save_json(final_results, save_path)
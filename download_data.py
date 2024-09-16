from datasets import load_dataset, load_from_disk
import argparse


parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset-cache-dir",
    type=str,
    required=True,
    help="Path to save the pile dataset",
)

args = parser.parse_args()

raw_datasets = load_dataset('monology/pile-test-val')

raw_datasets.save_to_disk(args.dataset_cache_dir, num_proc=48)


# filtered data for evaluation

# pg19
pg19 = raw_datasets.filter(lambda item: item['meta']['pile_set_name'] == 'Gutenberg (PG-19)', num_proc=48)
pg19.save_to_disk(args.dataset_cache_dir+"_pg19", num_proc=48)

# github
github = raw_datasets.filter(lambda item: item['meta']['pile_set_name'] == 'Github', num_proc=48)
github.save_to_disk(args.dataset_cache_dir+"_github", num_proc=48)

# aixiv
arxiv = raw_datasets.filter(lambda item: item['meta']['pile_set_name'] == 'ArXiv', num_proc=48)
arxiv.save_to_disk(args.dataset_cache_dir+"_arxiv", num_proc=48)

# if __name__ == '__main__':
#     # main()
#     raw_datasets = load_dataset('/zhujiajun/data/wikitext/wikitext-103-raw-v1', num_proc=30)
#     raw_datasets.save_to_disk('/zhujiajun/BiPE/data/wikitext', num_proc=48)
from python_research.experiments.band_selection_algorithms.BS_IC.improved_class_map import \
    generate_pseudo_ground_truth_map
from python_research.experiments.band_selection_algorithms.BS_IC.select_bands import select_bands
from python_research.experiments.band_selection_algorithms.utils import arg_parser


def main():
    arguments = arg_parser()
    generated_map = generate_pseudo_ground_truth_map(arguments, save_map=False)
    select_bands(generated_map, arguments)


if __name__ == '__main__':
    main()

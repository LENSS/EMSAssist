
import argparse
import Utils as util
from datetime import datetime

def attach_label(file_no_label, file_with_label, file_labeled, separator = "~|~"):

    lines_no_label = util.readFile(file_no_label)
    lines_with_label = util.readFile(file_with_label)
    assert len(lines_no_label) == len(lines_with_label)

    lines_labeled = []
    for line_no_label, line_with_label in zip(lines_no_label, lines_with_label):
        label = line_with_label.split(separator)[-1]
        line_labeled = line_no_label + separator + label
        lines_labeled.append(line_labeled)
    
    util.writeListFile(file_labeled, lines_labeled)


if __name__ == "__main__":

    t_start = datetime.now()

    parser = argparse.ArgumentParser(description = "control the functions for EMSBert")
    parser.add_argument("--file_no_label", action='store', type=str, help="file that needs labeling", required=True)
    parser.add_argument("--file_with_label", action='store', type=str, help="file that comes with label", required=True)
#    parser.add_argument("--input_label_separator", action='store', type=str, help="separator used to separate sample and label", required=False)
    parser.add_argument("--file_labeled", action='store', type=str, help="file after labeling will output", required=True)
#    parser.add_argument("--output_label_separator", action='store', type=str, help="separator used to separate sample and label", required=False)

    args = parser.parse_args()

    attach_label(args.file_no_label, args.file_with_label, args.file_labeled)

    t_total = datetime.now() - t_start
    print("\nThis run takes time: %s" % t_total)

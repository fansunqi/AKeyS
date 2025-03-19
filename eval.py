import json
import argparse

def main(filepath):
    data = json.load(open(filepath))

    no_ans = []
    accs = []
    frames = []
    for key in data:
        if data[key]["answer"] == -1:
            no_ans.append(key)
            continue
        else:
            acc = data[key]["answer"] == data[key]["label"]
            accs.append(acc)

        frame = data[key]["count_frame"]
        frames.append(frame)

    print("Total: ", len(data))
    print("No answer: ", len(no_ans))
    print("Have answer: ", len(accs))
    print("Mean accuracy (included no answer): {:.2f}%".format((sum(accs) + len(no_ans) * 0.2) / (len(accs) + len(no_ans)) * 100))
    print("Mean accuracy (excluded no answer): {:.2f}%".format(sum(accs) / len(accs) * 100))
    print("Mean frame: {:.2f}".format(sum(frames) / len(frames)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process timestamp for JSON file.")
    parser.add_argument("filepath", default="", type=str)
    args = parser.parse_args()
    main(args.filepath)
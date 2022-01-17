import os

trainroot = './../LibriSpeech/train-clean-100/' #, 'train-clean-360/', 'train-other-500/'
#devroot = './../LibriSpeech/dev-other/' #, './../LibriSpeech/dev-other/'
#testroot = './../LibriSpeech/test-clean/'

def generate_list(root_dir, fn):

    # get the utterance ids
    utterance_ids = []
    for subdir, _, files in os.walk(root_dir):
        for filename in [f for f in files if f.endswith(".txt")]:
            with open(os.path.join(subdir, filename)) as f:
                ids = [l.split(" ")[0] + "\n" for l in f.readlines()]
                utterance_ids.extend(ids)

    # write them
    with open(fn, "w") as of:
        of.writelines(utterance_ids)

if __name__ == "__main__":
    generate_list(trainroot, "./list/train1.txt")
    #generate_list(testroot, "./list/eval.txt")
    #generate_list(devroot, "./list/validation2.txt")

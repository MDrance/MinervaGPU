import time
import os
from collections import  defaultdict
class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Timer(object):
    """Computes elapsed time."""

    def __init__(self):
        self.running = True
        self.total = 0
        self.start = time.time()

    def reset(self):
        self.running = True
        self.total = 0
        self.start = time.time()
        return self

    def resume(self):
        if not self.running:
            self.running = True
            self.start = time.time()
        return self

    def stop(self):
        if self.running:
            self.running = False
            self.total += time.time() - self.start
        return self

    def time(self):
        if self.running:
            return self.total + time.time() - self.start
        return self.total


class Printer:
    def __init__(self, args):
        self.per_relation_paths = defaultdict(list)
        self.just_answers = defaultdict(list)
        self.print_dir = os.path.join(args.output_dir,"printed_paths")



    def save_paths(self, args, e1, r, e2, to_print, KB, blp, print_reward=False, reward=None):
        e1 = e1.cpu().numpy().tolist()
        e2 = e2.cpu().numpy().tolist()

        to_print = [[t[0].detach().cpu().numpy().tolist(), t[1].detach().cpu().numpy().tolist()] for t in to_print]
        blp = blp.detach().cpu().numpy().tolist()

        if print_reward:
            reward = reward.tolist()
        r = r.cpu().numpy().tolist()
        self.ev = KB.e_vocab
        self.rv = KB.r_vocab


        print("Printing file at {}".format(self.print_dir))
        if not os.path.exists(args.output_dir+"/printed_paths"):
            os.makedirs(args.output_dir+"/printed_paths")

        # print(len(e1))

        rank = 1
        for i in range(len(e1)):
            solved = False
            self.just_answers[(self.ev[e1[i]], self.rv[r[i]])].append(self.ev[to_print[-1][1][i]])
            if i % args.beam_size == 0:
                self.per_relation_paths[r[i]].append("#######")
                if e2[i] == to_print[-1][1][i]:
                    solved = True
                self.per_relation_paths[r[i]].append(str(solved))
                rank = 1

            self.per_relation_paths[r[i]].append(self.ev[e1[i]] + "\t" + self.rv[r[i]] + "\t" + self.ev[e2[i]])
            self.per_relation_paths[r[i]].append(str(blp[i]))
            self.per_relation_paths[r[i]].append("Reward:{}".format(str(reward[i])))
            self.per_relation_paths[r[i]].append("Rank:{}".format(str(rank)))
            for j in range(KB.args.num_steps):
                try:
                    self.per_relation_paths[r[i]].append(self.rv[to_print[j][0][i]] + "\t" + self.ev[to_print[j][1][i]])
                except:
                    import pdb
                    pdb.set_trace()
            self.per_relation_paths[r[i]].append("----------")
            rank += 1


    def print(self):
        for rel in self.per_relation_paths:
            rel_str = self.rv[rel]
            with open(self.print_dir+ "/"+rel_str.replace("/", "_"), "w") as print_out_file:
                print_out_file.write("\n".join(self.per_relation_paths[rel]))

        with open(self.print_dir + "/" + "all_answers.txt", "w") as ans_file:
            for q in self.just_answers:
                ans = self.just_answers[q]
                e, rel = q
                ans_file.write(e + "\t" + rel + "\t" + "\t".join([a for a in ans]) + "\n")

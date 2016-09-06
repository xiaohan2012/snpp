# Runtime status storage


class Status():
    def __init__(self):
        # for each iteration
        self.pred_cnt_list = []  # the accumulated number of predictions
        self.acc_list = []  # accuracy
        self.part_list = []  # partitions
        self.pred_list = []  # list of prediction bunches

    def update(self, predictions, acc, part):
        self.part_list.append(part)
        self.pred_list.append(predictions)
        if len(self.pred_cnt_list):
            self.pred_cnt_list.append(len(predictions) + self.pred_cnt_list[-1])
        else:
            self.pred_cnt_list.append(len(predictions))
        self.acc_list.append(acc)
